# Repository Structure
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
│
├── data_engineering/                 ← Data Engineering (DuckDB-based)
│   ├── ingestion.py                  ← SQL-based cleaning and merging pipeline
│   └── run_ingestion.py              ← Pipeline entrypoint script
│
├── data/
│   ├── raw/                          ← Pipeline input (not in git)
│   └── energy_weather_merged.parquet ← Contract between DE and ML (not in git)
│
├── artifacts/
│   ├── model_v1.lgb                  ← Trained LightGBM model
│   └── model_v1_config.json          ← Feature list, params, metrics
│
├── tests/
│   ├── test_features/                ← Small tests (pure functions, no I/O)
│   ├── test_serving/                 ← Medium tests (API, store, predict)
│   └── test_data_engineering/        ← Pipeline tests (DuckDB with fixtures)
│
├── terraform/
│   └── main.tf                       ← ECR + EC2 deployment
│
├── docs/
│   ├── ADR.md                        ← Architecture Decision Records (13)
│   ├── BOUNDED_CONTEXTS.md           ← Module responsibilities and boundaries
│   ├── devlog.md                     ← Key learnings and decisions
│   ├── project_plan.md               ← Timeline, scope, MoSCoW priorities
│   ├── business_case.md              ← Use case and market context
│   └── repo_structure.md             ← This file
│
├── .github/workflows/ci.yml          ← CI: ruff + pytest + docker build
├── Dockerfile                        ← Multi-stage build
├── pyproject.toml                    ← Package config, dependencies, tooling
└── README.md                         ← Project overview (you are here)
```
