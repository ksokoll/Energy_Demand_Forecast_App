# Energy Demand Forecast App

> **A production-grade 24h-ahead energy demand forecasting system for Spain, covering the full pipeline from data engineering (DuckDB) through ML training (LightGBM) to a deployed serving API (FastAPI + Docker + AWS).**

---

## Purpose & Context

Time-Series Forecasting always has something magical to me: We are looking into the crystal ball and are somewhat able to see what is coming.

This system predicts the next 24 hours of electricity demand for Spain using historical load data and weather observations. This is not an academic exercise: day-ahead load forecasting is a real operational task performed daily by Transmission System Operators across Europe. The methodology (lag features, cyclical encoding, gradient boosting) transfers directly to adjacent problems like visitor forecasting, retail demand planning, or building energy management.

The project is structured as two complementary parts: a Kaggle notebook handling EDA, feature engineering, and model training, and this repository handling everything needed to serve predictions in production.

Most ML portfolio projects stop at the Jupyter notebook. Model trained, metrics reported, done. This project exists to demonstrate the step that comes after: turning a working model into a deployable system with proper architecture, testing, and infrastructure.

**Kaggle Notebook (Training):** [[Link to Kaggle Notebook](https://www.kaggle.com/code/dantronic/energy-forecast-project)]

---

## What It Does

A user (or scheduler) sends a timestamp to the API. The service fetches historical load data from the data store, builds 51 features (calendar, weather, lag, holiday), runs the trained LightGBM model, and returns 24 hourly predictions. The entire flow takes under 2 seconds.

```
POST /forecast
{ "forecast_from": "2018-06-15T00:00:00+00:00" }

→ 24 hourly predictions (MW), model version, feature count
```

The system was deployed to AWS EC2 via Terraform and Docker, verified with a live forecast request, then torn down to avoid costs. The Terraform configuration remains in the repo for reproducibility.

---

## Architecture

The application follows Domain-Driven Design with five bounded contexts. The key architectural decision: Data Engineering and ML Serving are separated, connected only by a Parquet file.

```
[Data Engineering]                    [ML Serving]

data_engineering/     data/*.parquet     serving/api.py
  ingestion.py   →    (contract)    →     serving/predict.py → features/pipeline.py
  run_ingestion.py                                           → data/store.py
```

The Data Engineering pipeline (DuckDB) handles raw data cleaning: weather deduplication, city-level pivoting, null handling. It outputs a clean Parquet file. The Serving side reads that file, builds features at request time using the exact same Python functions that were validated during training (preventing training-serving skew), and returns predictions.

This separation exists because the two sides have different lifecycles. The pipeline runs once (or daily when new data arrives). The API runs continuously. They have different dependencies (DuckDB vs. LightGBM), different deployment targets (batch job vs. long-running container), and can be tested independently.

For the full module breakdown, see [`docs/BOUNDED_CONTEXTS.md`](docs/BOUNDED_CONTEXTS.md).

---

## Model Results

The model was trained on 2015-2017 and evaluated on 2018 (full year, chronological split). Feature engineering was the primary driver of performance, not model complexity or hyperparameter tuning.

| Model | RMSE (MW) | MAE (MW) |
|---|---|---|
| LightGBM on raw features (no engineering) | 4,018 | 3,268 |
| Naive baseline (same hour last week) | 3,815 | 2,647 |
| LightGBM with engineered features | 2,284 | 1,578 |
| TSO forecast (Red Eléctrica) | 389 | 270 |

The same model with the same hyperparameters went from worse than naive (4,018) to 40% better than naive (2,284). The only difference was feature engineering. Hyperparameter tuning with Optuna (100 trials) produced 0% improvement. H2O AutoML with Stacked Ensembles scored worse (2,321) than our single default-param LightGBM.

The TSO forecast from Red Eléctrica (389 RMSE) is 6x better than our model, which is expected. They operate dedicated infrastructure with real-time telemetry and proprietary data. The comparison provides realistic context, not a competition target.

Feature importance analysis confirmed the engineering approach: all top 10 features are engineered. Sine/cosine encoded day-of-year ranked #1 and #2, while raw `month` as integer didn't make the top 20.

For the detailed evaluation, feature importance analysis, and known limitations, see the Kaggle notebook.

---

## Key Technical Decisions

Every non-obvious architectural choice is documented in an Architecture Decision Record. Here are the most significant ones:

**Generation columns excluded (ADR-004):** The dataset contains 20+ generation columns (fossil gas, solar, wind). These are a consequence of demand, not a cause. The grid dispatches power in response to load. Using them as features would be target leakage. This decision is consistent with the dataset creator's own approach.

**Feature engineering over model complexity (ADR-007):** Rather than stacking models or using deep learning, I invested in cyclical encoding, lag features with leakage guards, rolling means, and holiday indicators. The empirical result: features drove 40% improvement, model tuning drove 0%.

**Weather kept despite leakage nuance (ADR-005):** The weather data represents observed conditions, not forecasts. In production, day-ahead weather forecasts would substitute. Tested impact: removing weather costs only 2.5% RMSE. Documented as known limitation.

**Protocol interface for DataStore (ADR-012):** The serving code depends on a Protocol, not a specific implementation. Swapping from local Parquet to S3 required changing one line in `main.py`. FileDataStore for development, S3DataStore for cloud. Same code, same tests.

All 13 ADRs are in [`docs/ADR.md`](docs/ADR.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | LightGBM (native .lgb format) |
| Feature Pipeline | pandas, numpy, scikit-learn, holidays |
| Serving API | FastAPI, Pydantic, uvicorn |
| Data Engineering | DuckDB (SQL-based ingestion pipeline) |
| Data Store | Parquet (local), S3 (cloud) |
| Containerization | Docker (multi-stage build) |
| Infrastructure | Terraform (ECR + EC2) |
| CI | GitHub Actions (ruff + pytest + docker build) |
| Testing | pytest (61 tests, 96% coverage) |

---

## Project Structure

```
Energy_Demand_Forecast_App/
├── src/energy_forecast/              ← ML Serving Package
│   ├── config.py                     ← Central constants, paths, feature lists
│   ├── models.py                     ← Pydantic schemas, custom exceptions
│   ├── main.py                       ← Application entrypoint
│   ├── data/
│   │   └── store.py                  ← DataStore Protocol, FileDataStore, S3DataStore
│   ├── features/
│   │   ├── calendar.py               ← Cyclical encoding, weekday, season
│   │   ├── weather.py                ← Temperature deviation, interactions
│   │   ├── lag.py                    ← Lag features with leakage prevention
│   │   ├── holiday.py                ← Spanish holidays, bridge days
│   │   └── pipeline.py              ← Single entry point for all features
│   └── serving/
│       ├── api.py                    ← FastAPI endpoints
│       └── predict.py                ← Prediction orchestrator
├── data_engineering/                 ← DuckDB-based ingestion pipeline
│   ├── ingestion.py
│   └── run_ingestion.py
├── tests/                            ← 61 tests, 96% coverage
├── terraform/                        ← ECR + EC2 deployment
├── docs/                             ← ADRs, bounded contexts, devlog
├── artifacts/                        ← Trained model + config
├── Dockerfile
└── pyproject.toml
```

---

## How to Run

**Prerequisites:** Python 3.11+, Docker Desktop (optional for containerized run)

**Local development:**
```bash
git clone https://github.com/ksokoll/Energy_Demand_Forecast_App.git
cd Energy_Demand_Forecast_App
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e ".[dev]"

# Run the data engineering pipeline (requires raw CSVs in data/raw/)
pip install -e ".[ingestion]"
python data_engineering/run_ingestion.py

# Start the API
uvicorn energy_forecast.main:app --reload

# Run tests
pytest tests/ -v
```

**Docker:**
```bash
docker build -t energy-forecast .
docker run -p 8000:8000 energy-forecast
```

**API endpoints:**
- `GET /health` : Service status and model info
- `POST /forecast` : 24h-ahead prediction (body: `{"forecast_from": "2018-06-15T00:00:00+00:00"}`)
- `GET /docs` : Interactive Swagger documentation

---

## Testing

61 tests across three levels following the test pyramid from "Software Engineering at Google":

- **Small tests (39):** Pure feature functions with synthetic DataFrames. No I/O, no external dependencies. Calendar encoding, lag correctness, leakage prevention, holiday detection.
- **Medium tests (17):** FastAPI TestClient, FileDataStore with temp files, prediction orchestrator with a dummy LightGBM model.
- **Integration tests (5):** Full DuckDB pipeline with CSV fixtures, end-to-end data flow.

A dedicated contract test verifies that the feature pipeline output matches the exact 51 columns the trained model expects, in the correct order. This prevents training-serving skew (the most dangerous bug in production ML systems).

---

## Documentation

| Document | Description |
|---|---|
| [`docs/ADR.md`](docs/ADR.md) | 13 Architecture Decision Records |
| [`docs/BOUNDED_CONTEXTS.md`](docs/BOUNDED_CONTEXTS.md) | Module responsibilities and boundaries |
| [`docs/DEVLOG.md`](docs/DEVLOG.md) | Key learnings and pivotal decisions |
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Market context and use case |

---

## Known Limitations

**Weather data uses observations, not forecasts.** Test metrics are slightly optimistic. In production, day-ahead weather forecasts (1-2°C typical error) would substitute. Quantified impact: 2.5% RMSE.

**No regional holidays.** Only national Spanish holidays are included. Regional holidays (Catalonia, Andalusia, etc.) could reduce errors on regionally observed days.

**Single model for all hours.** The same model predicts 3 AM (low, stable) and 7 PM (peak, volatile). Per-hour models might improve peak accuracy at the cost of complexity.

**Error handling at production-ready, not enterprise level.** Strategic logging at context boundaries, specific exception classes, but no structured JSON logging, correlation IDs, or circuit breakers. See ADR-010 for what enterprise-level would add.

**No permanent cloud deployment.** Deployed to AWS EC2 via Terraform, verified with live requests, then torn down. The infrastructure code remains reproducible.

**No Return Objects for Pipeline Orchestration**
The ingestion pipeline uses the pattern success = no exception. 
A PipelineResult dataclass (rows_written, columns, rows_dropped, 
duration_seconds) would bring consistency across projects and enable 
downstream monitoring without log parsing.
