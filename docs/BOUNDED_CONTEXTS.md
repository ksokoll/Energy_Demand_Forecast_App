# Bounded Contexts

## Overview

The application is structured around five bounded contexts, each with
a single responsibility. The ML Serving side (Contexts 1-4) lives in
`src/energy_forecast/`. The Data Engineering side (Context 5) lives in
`data_engineering/`. The Parquet file in `data/` is the contract between
them (see ADR-011).
```
[Data Engineering]                          [ML Serving]

data_engineering/        data/*.parquet     serving/api.py
  ingestion.py     →     (contract)    →     serving/predict.py  →  features/pipeline.py
  run_ingestion.py                                               →  data/store.py
```

## Context 1: Features (Domain Logic)

**Responsibility:** Transform raw data into model-ready feature vectors.

**Modules:**
- `features/calendar.py`: Cyclical encoding, weekday/weekend, season
- `features/weather.py`: Temperature deviation, interactions
- `features/lag.py`: Lag features with leakage prevention
- `features/holiday.py`: Spanish holidays, bridge days, backlog effect
- `features/pipeline.py`: Single entry point combining all transformers

**Key properties:**
- Pure functions. No I/O, no database, no API calls.
- Input: pandas DataFrame. Output: pandas DataFrame.
- Identical code in training (notebook) and serving (API). This
  prevents training-serving skew (ADR-013).
- Any divergence between training and serving here causes silent
  prediction errors.

**Boundary rule:** This context never imports from `serving/` or `data/`.
It depends only on `config.py` for constants (FORECAST_HORIZON,
COMFORT_TEMPERATURE, TARGET).

## Context 2: Data Store (Infrastructure)

**Responsibility:** Provide access to historical load data for lag
feature computation at serving time.

**Modules:**
- `data/store.py`: Protocol definition, FileDataStore, S3DataStore

**Key properties:**
- The serving API does not receive historical data from the caller.
  The caller sends a forecast trigger (timestamp), and the service
  fetches the required history internally.
- Minimum 168 hours (7 days) of historical data required for
  lag_168h and rolling_7d_same_hour features.
- Protocol interface enables swapping implementations without
  changing consuming code (ADR-012).
- FileDataStore for local development (reads Parquet/CSV from disk).
- S3DataStore for cloud deployment (reads Parquet from S3).

**Boundary rule:** This context knows how to read data but knows
nothing about features, models, or the API. It returns raw DataFrames,
not feature vectors.

## Context 3: Serving (Interface)

**Responsibility:** HTTP interface for forecast requests.

**Modules:**
- `serving/api.py`: FastAPI route definitions, health check

**Key properties:**
- Thin layer. Validates input, delegates to orchestration, returns output.
- No business logic. No feature computation. No model loading.
- Pydantic schemas from `models.py` enforce the API contract.

**Boundary rule:** This context imports from `serving/predict.py` only.
It never directly accesses `features/` or `data/`.

## Context 4: Prediction (Orchestration)

**Responsibility:** Coordinate the full prediction flow: fetch data,
build features, run model, return results.

**Modules:**
- `serving/predict.py`: Prediction orchestrator

**Key properties:**
- This is the equivalent of `handler.py` in the car damage project.
- Orchestrates the sequence: DataStore → Feature Pipeline → Model → Response.
- Loads the trained model artifact from disk (or S3 temp file) on startup.
- Constructs 24 feature vectors (one per forecast hour) from
  historical data and calendar information.

**Boundary rule:** This is the only module that imports from both
`features/` and `data/`. It is the integration point.

## Context 5: Data Engineering (Ingestion Pipeline)

**Responsibility:** Transform raw CSV data into clean, merged Parquet
files ready for the serving layer.

**Modules:**
- `data_engineering/ingestion.py`: DuckDB-based pipeline
- `data_engineering/run_ingestion.py`: Entrypoint script

**Key properties:**
- Lives outside the Python package (`data_engineering/`, not `src/`).
  The API Docker image does not include this code.
- Uses SQL (DuckDB) for all transformations: deduplication, pivot,
  merge, null handling.
- Feature engineering is explicitly NOT part of this pipeline (ADR-013).
- Output is a single Parquet file that the DataStore reads.
- Includes output validation (row count, null check, column count)
  to catch data quality issues before they reach the serving layer.

**Boundary rule:** This context imports only `TARGET` and `DATA_PATH`
from `config.py`. It knows nothing about features, models, or the API.

## Cross-Cutting: Config and Contracts

**`config.py`:** Central constants, feature lists, file paths.
Imported by all contexts. The FEATURE_COLS list is the single
source of truth for feature order and must match the trained model
exactly.

**`models.py`:** Pydantic schemas shared across contexts.
ForecastRequest/ForecastResponse define the API contract.
HourlyPrediction defines the output structure. Custom exception
classes (InsufficientHistoryError, ModelNotFoundError) for
domain-specific error handling.