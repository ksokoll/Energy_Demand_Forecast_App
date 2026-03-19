### ADR-001: LightGBM as Primary Model

**Status:** Accepted

**Context:** Need to choose a model for 24h-ahead energy demand forecasting. Candidates: LightGBM, XGBoost, Random Forest, LSTM, AutoML (H2O).

**Decision:** LightGBM as the single production model. No ensemble, no stacking.

**Reasoning:**
- Handles missing values natively, works well with correlated features (important for city-specific weather columns)
- Fast training enables rapid iteration during feature engineering
- Default hyperparameters already near-optimal for this problem (Optuna tuning over 100 trials produced 0% improvement)
- Easy to serialize, load, and serve via API. No complex dependency chain like LSTM (TensorFlow) or H2O (Java runtime)

**Alternatives considered:**
- XGBoost: viable alternative, but H2O's best single GBM (comparable to XGBoost) scored 2,337. No reason to switch.
- LSTM: standard in academic energy forecasting literature. But adds framework complexity (TensorFlow/PyTorch), longer training, harder to debug, and rarely beats gradient boosting on structured tabular data with well-engineered features.

---

### ADR-002: 24-Hour-Ahead Forecast Horizon

**Status:** Accepted

**Context:** Need to define the prediction horizon. Options: 1h, 24h, 7 days, 1 year.

**Decision:** Predict the next 24 hours from a single forecast trigger.

**Reasoning:**
- Matches the real-world use case: TSOs issue day-ahead forecasts daily for grid planning and energy market bidding
- Allows usage of lag_24h (same hour yesterday) as the strongest lag feature
- 24h is short enough that calendar features (hour, weekday) are highly predictive, but long enough to be operationally useful

**Consequences:**
- All lag features must be >= 24 hours. Enforced via `assert lag >= FORECAST_HORIZON` in code.
- The API must have access to at least 168 hours (7 days) of historical data to compute lag and rolling features at serving time.

---

### ADR-003: Chronological Train/Test Split

**Status:** Accepted

**Context:** Need a validation strategy for time series data.

**Decision:** Fixed chronological split: 2015-2017 training, 2018 test. Additionally, TimeSeriesSplit (3 folds) on training data for hyperparameter tuning.

**Reasoning:**
- Random splitting would leak future patterns into training. This is the most common mistake in time series projects and causes artificially low error metrics.
- Full year as test set covers all seasons, holidays, and weekday patterns. A shorter test period could miss seasonal effects.
- No refit on all data before final evaluation. The test metrics are honest out-of-sample scores, not optimistic resubstitution estimates.
- TimeSeriesSplit for tuning ensures hyperparameters are selected without touching the test set.

**Alternatives considered:**
- Walk-forward validation (train on expanding window, test on next block, repeat): more robust, but computationally expensive. The dataset creator used this approach. For our scope and timeline, fixed split is sufficient.
- Single train/validation/test split (three-way): would reduce training data. With only 4 years of data, every year matters.

---

### ADR-004: Generation and Price Columns Excluded

**Status:** Accepted

**Context:** The dataset contains 20+ generation columns (fossil gas, solar, wind, etc.), price columns, and TSO forecast columns. Should these be used as features?

**Decision:** All generation, price, and forecast columns excluded from features.

**Reasoning:**
- **Generation** is a consequence of demand. The grid operator dispatches power in response to load. Using generation as a feature is circular and constitutes target leakage.
- **Price** is co-determined with load via the market clearing mechanism. Same leakage issue.
- **Day-ahead forecasts** (solar, wind) are theoretically available at prediction time. Excluded for V1 to keep the model self-contained with no external forecast dependencies. Can be revisited in V2.
- **TSO load forecast** would make our model a copy of the existing forecast. Excluded as feature, used as benchmark instead.

**Validation:** This approach is consistent with the dataset creator's own forecasting project (github.com/kolasniwash/short-term-energy-demand-forecasting), which also excluded generation and price columns entirely.

---

### ADR-005: Weather Features Kept Despite Leakage Nuance

**Status:** Accepted

**Context:** The weather data in this dataset represents observed conditions, not forecasts. At serving time (predicting T+24), we would only have weather forecasts, not actual observations.

**Decision:** Keep weather features in the model. Document the limitation.

**Reasoning:**
- In production, day-ahead weather forecasts are widely available and reasonably accurate (typical error 1-2°C for temperature). The gap between observed and forecasted weather is small.
- The model should learn the real relationship between weather and demand. Training on observed weather is correct. The serving-time approximation (using forecasts instead of observations) introduces minor additional error but does not invalidate the learned patterns.
- The dataset creator made the same decision in his project.

**Trade-off:** Test set metrics are slightly optimistic because they use perfect weather. Documented as known limitation in the evaluation section.

---

### ADR-006: Weather Data Pivoted by City, Not Aggregated

**Status:** Accepted

**Context:** Weather data covers 5 cities. Need to decide how to merge with the national-level energy data. Options: mean across cities, pivot (one column per city per feature), or pick a single representative city.

**Decision:** Hybrid approach. Temperature averaged (low inter-city variance, CV 1-2%). High-variance features (pressure, humidity, wind, rain, clouds) pivoted by city.

**Reasoning:**
- Quantitative analysis showed that temperature is nearly identical across all 5 cities (CV 1-2%). Averaging loses almost no information.
- Wind speed (CV 62%), rain (CV 181%), and cloud cover (CV 104%) differ massively between cities. Correlation analysis confirmed: most city pairs show r < 0.3. Averaging these would destroy signal.
- LightGBM handles correlated and irrelevant features well (splits only where useful, ignores the rest). No risk of model degradation from additional columns.
- Feature importance confirmed: city-specific weather features rank #11-18, contributing supporting signal.

**Alternatives considered:**
- Full aggregation (mean all features): simpler, but quantifiably worse. Would destroy wind/rain/cloud signal.
- Single city (e.g. Madrid): arbitrary and ignores coastal vs inland weather differences.
- PCA on weather: the dataset creator used this approach. Valid, but less interpretable. Explainability matters for consulting deliverables.

---

### ADR-007: Feature Engineering Over Model Complexity

**Status:** Accepted

**Context:** Two strategies to improve performance: (a) invest in feature engineering with a simple model, or (b) use complex model architectures (ensembles, deep learning) with raw features.

**Decision:** Strategy (a). Invest heavily in feature engineering. Keep the model simple.

**Reasoning:**
- Empirical evidence from this project: 7 engineered features occupy 6 of the top 10 importance slots. Fourier terms for seasonality (#1, #2), diff features for trend (#4), rolling means for smoothing (#5, #6).
- Hyperparameter tuning (100 Optuna trials) produced 0% improvement. The feature set, not the model config, determines performance.
- H2O AutoML with model stacking scored worse (RMSE 2,321) than our single default-param LightGBM (RMSE 2,284).
- Simpler models are easier to explain to stakeholders, easier to debug, easier to deploy, and cheaper to maintain. In a consulting context, explainability is a requirement, not a nice-to-have.

**Consequence:** The `create_calendar_features()`, `create_weather_features()`, and `create_lag_features()` functions are the core intellectual property of this project. They must be well-tested and well-documented.

---

### ADR-008: Stateful Prediction Service

**Status:** Accepted

**Context:** The car damage detection API was stateless: image in, classification out. The energy forecast API requires historical data to compute lag features at serving time.

**Decision:** The prediction service requires access to a data store with at least 168 hours (7 days) of historical load data.

**Reasoning:**
- Lag features (lag_24h, lag_48h, lag_168h) and rolling means (3-day, 7-day) require historical values.
- Calendar and weather features can be computed from the request timestamp and an external weather forecast API. But lag features can only come from past observations.
- The API does not expect the caller to send historical data. The caller sends a forecast trigger (timestamp), and the service fetches history internally.

**Implementation for portfolio project:** CSV or Parquet file as static data store. In production, this would be a database or real-time API feed from the grid operator.

**Trade-off:** Adds architectural complexity compared to a stateless service. But this complexity is inherent to time series forecasting and worth demonstrating in a portfolio context.

### ADR-009: Holiday Features Kept Despite No Aggregate RMSE Improvement

**Status:** Accepted

**Context:** Holiday features were added late in the feature engineering process (inspired by strong results from previous learnings). Four features were created: `is_holiday`, `holiday_density_7d`, `days_since_holiday`, `is_bridge_day`. Overall RMSE did not improve (2,310 vs 2,284 without).

**Decision:** Keep all holiday features in the model.

**Reasoning:**
- Holidays have a measurable load effect: -2,847 MW vs normal weekdays, slightly stronger than weekends (-2,659 MW). The signal is real.
- The model's error on weekday holidays is double the normal weekday error (MAE 2,968 vs 1,452 MW). The model needs these features.
- No aggregate improvement because only 8 weekday holidays exist in the 2018 test set (192 of 8,758 hours = 2.2%). The fix is invisible in the overall metric. This is a sample size problem, not a signal problem.
- `days_since_holiday` ranked #10 in feature importance, confirming LightGBM actively uses it. This feature captures the "backlog effect" (post-holiday demand recovery), same pattern observed in previous projects regarding visitor forecasting.
- The other three features (`is_holiday`, `holiday_density_7d`, `is_bridge_day`) ranked low due to sparsity (binary features with 95%+ zeros). Kept anyway because they are cheap to compute and domain-correct.
- In production, every holiday forecast matters for grid planning. Aggregate RMSE is the wrong metric to evaluate holiday features. Per-holiday accuracy is what counts operationally.

**V2 improvements:** Add regional holidays (Catalonia, Andalusia, Basque Country, etc.). Spain's regional calendar is more granular than the national one and could further reduce errors on regionally observed holidays.

## ADR-010: Error Handling and Logging at Production-Ready Level

**Status:** Accepted

**Context:** Error handling and logging range from minimal (silent 
failures, no logs) to enterprise-grade (structured JSON logging, 
correlation IDs, circuit breakers, alerting integration). Need to 
choose the right level for a portfolio project.

**Decision:** Implement "Production-Ready" logging (Level 2 of 3). 
Strategic log statements at context boundaries (request in, store 
access, prediction out). Specific exception classes for domain errors. 
Request tracing via log context. No logging inside pure feature 
engineering functions.

**Reasoning:**
- Level 1 (MVP, silent) would make debugging in Azure impossible 
  without SSH access. Not acceptable for a deployed service.
- Level 2 (Production-Ready) adds ~30 lines across 3 modules. 
  Enough to diagnose any production issue from logs alone.
- Level 3 (Enterprise) would include structured JSON logging for 
  log aggregation tools (ELK, Datadog), correlation IDs across 
  services, circuit breaker patterns, retry logic with exponential 
  backoff, rate limiting, input sanitization, and alerting 
  integration (PagerDuty, OpsGenie). This is the right choice for 
  regulated environments (finance, healthcare, grid operators) but 
  over-engineering for a single-service portfolio project.

**What Level 3 would add (not implemented):**
- Structured JSON logging (for Elasticsearch/CloudWatch ingestion)
- Correlation IDs across service boundaries
- Circuit breaker on DataStore access
- Retry with exponential backoff for transient failures
- Rate limiting on the API
- Health checks with dependency status (store reachable, model age)
- Alerting integration on error rate thresholds
- Audit logging (who requested what, when)

**Trade-off:** A reviewer familiar with enterprise systems will notice 
the absence of these patterns. The ADR demonstrates awareness of what 
is missing and why it was excluded, which is more valuable than 
implementing it poorly under time pressure.

---

### ADR-011: Data Engineering and Serving as Separate Bounded Contexts

**Status:** Accepted

**Context:** The system has two distinct workflows: (1) cleaning and merging raw data (Data Engineering), and (2) building features and serving predictions (ML Serving). Both could live in the same Python package, or be separated.

**Decision:** Separate them. Data Engineering lives in `data_engineering/` outside the installable Python package. ML Serving lives in `src/energy_forecast/`. The Parquet file in `data/` is the contract between them.

**Reasoning:**
- Different lifecycles: the pipeline runs once daily (or once for initial setup), the API runs continuously. Different deployment targets (batch job vs. long-running service).
- Different dependencies: DuckDB is only needed by the pipeline, not by the API. Separating them keeps the Docker image for the API smaller.
- Independent testability: pipeline tests verify data quality, API tests verify prediction correctness. Neither depends on the other.
- Clear ownership boundary: if the data source changes (new API, new format), only the pipeline changes. The serving code never touches raw data.

**Trade-off:** Shared constants (TARGET, DATA_PATH) create a coupling via `config.py`. Acceptable for a single-repo project. In a multi-repo setup, these would move to a shared config package.

---

### ADR-012: Protocol Interface for DataStore

**Status:** Accepted

**Context:** The serving API needs historical load data to compute lag features. This data could come from a local file (development), S3 (cloud), or a database (production). Need to decide how to abstract this.

**Decision:** Define a `DataStore` Protocol with a single `get_history()` method. Implement `FileDataStore` for local development and `S3DataStore` for cloud deployment. The serving code depends only on the Protocol, never on a specific implementation.

**Reasoning:**
- The Protocol pattern (PEP 544) enables structural subtyping: any class with a matching `get_history()` signature satisfies the contract, without explicit inheritance. This is idiomatic Python.
- Swapping the data source requires changing one line in `main.py` (which store to instantiate), not touching any serving or feature code.
- Testability: tests inject a mock or a FileDataStore with small fixtures. No S3 credentials needed in CI.
- Matches the DDD principle: domain logic (features, prediction) is decoupled from infrastructure (where data lives).

**Alternatives considered:**
- Abstract Base Class (ABC): more explicit, but requires `class MyStore(DataStore)` inheritance. Protocol is lighter and more Pythonic.
- No abstraction (hardcode file reads): works initially but forces a rewrite when the data source changes. The Protocol costs 10 lines of code and prevents that.

---

### ADR-013: Feature Engineering in Python, Not SQL

**Status:** Accepted

**Context:** The DuckDB ingestion pipeline already uses SQL for data cleaning and merging. Feature engineering (calendar, weather, lag, holiday features) could also be implemented in SQL within the same pipeline.

**Decision:** Feature engineering stays in Python (`src/energy_forecast/features/`). The DuckDB pipeline only produces clean, merged raw data. Features are built at serving time by the Python feature pipeline.

**Reasoning:**
- Training-serving skew prevention: the exact same Python functions that were validated in the training notebook are used in the serving API. If features were rebuilt in SQL, there would be two implementations of the same logic with no guarantee of consistency. This is the highest-risk bug in production ML systems (Google Rules of ML #32, #37).
- Testability: 39 small tests verify the Python feature functions with synthetic data. SQL-based features would require a running DuckDB instance for every test.
- The feature pipeline is the core intellectual property of the project (ADR-007). It must be explicit, readable, and testable. SQL is powerful for data transformation but harder to unit test and version.

**Consequence:** The DuckDB pipeline and the feature pipeline have a clear boundary: DuckDB outputs clean rows with raw columns, Python adds engineered features on top. No feature logic crosses this boundary.