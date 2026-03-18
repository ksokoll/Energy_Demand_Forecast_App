# Bounded Contexts

## Overview

The serving application is structured around four bounded contexts,
each with a single responsibility. Dependencies flow inward: the API
layer depends on the orchestration layer, which depends on domain
logic and infrastructure. Domain logic (features) has no external
dependencies beyond pandas and numpy.
```
serving/api.py  →  serving/predict.py  →  features/pipeline.py  (domain)
                                       →  data/store.py         (infrastructure)
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
- Identical code in training (notebook) and serving (API).
- Any divergence between training and serving here causes silent
  prediction errors (training-serving skew).

**Boundary rule:** This context never imports from `serving/` or `data/`.
It depends only on `config.py` for constants (FORECAST_HORIZON,
COMFORT_TEMPERATURE, TARGET).

## Context 2: Data Store (Infrastructure)

**Responsibility:** Provide access to historical load data for lag
feature computation at serving time.

**Modules:**
- `data/store.py`: Protocol definition and implementation

**Key properties:**
- The serving API does not receive historical data from the caller.
  The caller sends a forecast trigger (timestamp), and the service
  fetches the required history internally.
- Minimum 168 hours (7 days) of historical data required for
  lag_168h and rolling_7d_same_hour features.
- Currently implemented as a CSV/Parquet reader. Planned migration
  to DuckDB for SQL-based access (ADR pending).

**Boundary rule:** This context knows how to read data but knows
nothing about features, models, or the API. It returns raw DataFrames,
not feature vectors.

## Context 3: Serving (Interface)

**Responsibility:** HTTP interface for forecast requests.

**Modules:**
- `serving/api.py`: FastAPI route definitions, health check
- `serving/schemas.py`: Request/Response Pydantic models (re-exports
  from models.py for API-specific documentation)

**Key properties:**
- Thin layer. Validates input, delegates to orchestration, returns output.
- No business logic. No feature computation. No model loading.
- Pydantic schemas enforce the API contract.

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
- Loads the trained model artifact from disk on startup.
- Constructs 24 feature vectors (one per forecast hour) from
  historical data and calendar information.

**Boundary rule:** This is the only module that imports from both
`features/` and `data/`. It is the integration point.

## Cross-Cutting: Config and Contracts

**`config.py`:** Central constants, feature lists, file paths.
Imported by all contexts. The FEATURE_COLS list is the single
source of truth for feature order and must match the trained model
exactly.

**`models.py`:** Pydantic schemas shared across contexts.
ForecastRequest/ForecastResponse define the API contract.
HourlyPrediction defines the output structure.

## Comparison to Car Damage Detection API

| Aspect | Car Damage | Energy Forecast |
|---|---|---|
| Trigger | User sends image | User sends timestamp |
| State | Stateless (image is self-contained) | Stateful (needs historical data) |
| Domain Logic | Image quality validation | Feature engineering |
| Inference | ONNX Runtime | LightGBM native |
| Storage | DynamoDB (write results) | DuckDB (read history) |
| Orchestrator | handler.py | predict.py |

The key architectural difference: the car damage API receives
everything it needs in the request. The energy forecast API must
fetch historical data from an internal store before it can build
features and predict.