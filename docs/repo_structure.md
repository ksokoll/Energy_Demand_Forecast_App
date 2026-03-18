energy_forecast_serving/
├── src/energy_forecast/              ← ML Serving Package
│   ├── config.py
│   ├── models.py
│   ├── data/
│   │   └── store.py
│   ├── features/
│   └── serving/
│
├── data_engineering/                 ← Data Engineering (DuckDB-based)
│   ├── ingestion.py
│   └── run_ingestion.py
│
├── data/
│   ├── raw/                          ← Pipeline Input
│   └── energy_weather_merged.parquet ← Contract between DE and ML
│
├── artifacts/
├── tests/
├── docs/
├── Dockerfile
└── pyproject.toml