## Business Case

### The Problem: From Notebook to Production

Most time series forecasting exists only in Jupyter notebooks. Models are trained once, evaluated on a screenshot of metrics, and never leave the data scientist's laptop. The gap between "model works in a notebook" and "model runs as a service that a team can operate" is where most ML projects fail.

This project closes that gap. It takes a common forecasting problem (24h-ahead demand prediction with weather and calendar influences) and delivers it as a production-grade serving system: containerized, tested, documented, and deployable.

### Why Energy Demand?

Energy demand forecasting is the demonstration, not the destination. The methodology applies to any time series where demand is shaped by weather, calendar, and historical patterns:

| Domain | Target Variable | Same Feature Categories |
|---|---|---|
| **Energy** (this project) | Hourly load (MW) | Seasonality, weekday/weekend, temperature, lags |
| Water utilities | Daily consumption (m³) | Seasonality, weekday/weekend, temperature, lags |
| Retail | Daily revenue (EUR) | Seasonality, weekday/weekend, weather, lags |
| Transport | Hourly ridership | Seasonality, weekday/weekend, weather, lags |
| Healthcare | Daily bed occupancy | Seasonality, weekday/weekend, flu season proxy, lags |

The feature architecture is the same across domains: cyclical encoding for time, deviation-based weather features, lag/rolling features with leakage prevention, and holiday effects. What changes between domains is the data source, the forecast horizon, and the domain-specific interaction terms.

### Results

The model achieves an RMSE of **2,284 MW** on a full year of out-of-sample data (2018), a **40% improvement** over the naive seasonal baseline (3,815 MW). For context, Spain's actual grid operator (REE) achieves an RMSE of 389 MW using proprietary models and real-time telemetry.

| Metric | Value |
|---|---|
| Test RMSE | 2,284 MW |
| Test MAE | 1,578 MW |
| Baseline RMSE (naive seasonal) | 3,815 MW |
| Improvement over baseline | 40.1% |
| TSO benchmark (REE, with live data) | 389 MW |

Two deliberate decisions shaped these numbers. First, features over model complexity: 7 of the top 10 features by importance are engineered (cyclical encoding, diffs, rolling means). Hyperparameter tuning across 100 Optuna trials produced 0% improvement. H2O AutoML with stacking scored worse (RMSE 2,321) than a single LightGBM with default parameters. Second, honest evaluation: chronological split (train 2015-2017, test 2018), no refit on test data, no random shuffling. The test metrics are genuine out-of-sample scores.

### What This Project Demonstrates

**For consulting delivery**, this project shows the ability to take a client problem from raw data to a running API in a structured, documented way:

- **Reproducible feature engineering.** Four modular transformers (calendar, weather, lag, holiday) that run identically in training and serving. No training-serving skew.
- **Leakage-proof design.** Every lag feature enforces `lag >= forecast_horizon` with runtime assertions. The contract test verifies feature column order against the trained model config.
- **Production architecture.** Bounded contexts separating domain logic, data access, and HTTP interface. Protocol-based data store for swappable backends. Factory-pattern API for testable dependency injection.
- **Documented trade-offs.** 10 Architecture Decision Records explain not just what was built, but why alternatives were rejected, backed by quantitative evidence.
- **Tested.** 59 tests covering all feature modules, the data store, the prediction service, and the API layer. Every test uses synthetic data with no I/O dependencies.