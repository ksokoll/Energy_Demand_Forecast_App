"""Weather-based feature engineering.

Transforms raw weather observations into model-ready features.
Key insight from EDA: temperature and load have a U-shaped relationship
(both heating and cooling drive demand). The temp_deviation feature
linearizes this relationship.
"""

import pandas as pd

from energy_forecast.config import COMFORT_TEMPERATURE


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weather-derived features.

    Assumes temperature columns are in Kelvin and that calendar
    features (is_weekend, sin_hour) already exist in the DataFrame.

    Args:
        df: Input DataFrame with temp column in Kelvin and
            pre-computed calendar features.

    Returns:
        Copy of the input DataFrame with 5 additional columns:
        temp_celsius, temp_deviation, temp_deviation_sq,
        temp_dev_x_weekend, temp_dev_x_hour_sin.
    """
    result = df.copy()

    result["temp_celsius"] = result["temp"] - 273.15

    # Distance from comfort zone captures the U-shape:
    # both cold (heating) and hot (cooling) increase demand
    result["temp_deviation"] = (result["temp_celsius"] - COMFORT_TEMPERATURE).abs()
    result["temp_deviation_sq"] = result["temp_deviation"] ** 2

    # Interactions with calendar features
    result["temp_dev_x_weekend"] = result["temp_deviation"] * result["is_weekend"]
    result["temp_dev_x_hour_sin"] = result["temp_deviation"] * result["sin_hour"]

    return result
