"""Central configuration for the energy forecast serving application.

All constants, feature lists, and paths are defined here.
Feature lists must match exactly what the model was trained on
to prevent training-serving skew (Hyrum's Law).
"""

import os
from pathlib import Path

# --- Paths ---

# Use environment variable if set (Docker), otherwise resolve from file location
PROJECT_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent.parent.parent))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_v1.lgb"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_v1_config.json"
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "energy_weather_merged.parquet"
RAW_DIR = DATA_DIR / "raw"

# --- Forecast settings ---
FORECAST_HORIZON = 24  # hours
MINIMUM_HISTORY_HOURS = 168  # 7 days, required for lag_168h and rolling_7d
COMFORT_TEMPERATURE = 18.0  # °C, center of U-shape

# --- Feature lists (must match training notebook exactly) ---
CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "day_of_year",
    "season",
    "sin_day_of_year",
    "cos_day_of_year",
    "sin_hour",
    "cos_hour",
]

HOLIDAY_FEATURES = [
    "is_holiday",
    "holiday_density_7d",
    "days_since_holiday",
    "is_bridge_day",
]

WEATHER_FEATURES = [
    "temp_celsius",
    "temp_deviation",
    "temp_deviation_sq",
    "temp_dev_x_weekend",
    "temp_dev_x_hour_sin",
]

CITIES = ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]

CITY_WEATHER_FEATURES = [
    f"{feature}_{city}"
    for feature in ["pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]
    for city in CITIES
]

LAG_FEATURES = [
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_7d_same_hour",
    "rolling_3d_same_hour",
    "diff_24h_vs_168h",
    "diff_24h_vs_48h",
]

# Complete feature list in training order
FEATURE_COLS = (
    CITY_WEATHER_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES + HOLIDAY_FEATURES
)

TARGET = "total load actual"
