"""Feature engineering pipeline combining all transformer modules.

This is the single entry point for feature construction, used by
both training (notebook) and serving (API). Using the same pipeline
in both contexts prevents training-serving skew.
"""

import pandas as pd

from energy_forecast.features.calendar import create_calendar_features
from energy_forecast.features.holiday import create_holiday_features
from energy_forecast.features.lag import create_lag_features
from energy_forecast.features.weather import create_weather_features


def build_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order.

    Order matters: weather features depend on calendar features
    (is_weekend, sin_hour), and lag features require the DataFrame
    to be sorted by time.

    Args:
        df: Input DataFrame with timestamp, weather, and target columns.
        time_col: Name of the timestamp column.

    Returns:
        Copy of the input DataFrame with all engineered features added.
        Adds 51 feature columns total: 10 calendar, 5 weather,
        7 lag, 4 holiday, plus 25 existing city-weather columns.
    """
    result = df.sort_values(time_col).reset_index(drop=True)

    result = create_calendar_features(result, time_col=time_col)
    result = create_weather_features(result)
    result = create_lag_features(result)
    result = create_holiday_features(result, time_col=time_col)

    return result
