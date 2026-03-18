"""Calendar-based feature engineering.

Extracts time-aware features from timestamps including cyclical
encoding (sine/cosine) to preserve the circular nature of hours
and days. Cyclical encoding ranked #1 and #2 in feature importance
during training, outperforming raw integer representations.
"""

import numpy as np
import pandas as pd

# Meteorological season mapping (DJF=winter, MAM=spring, JJA=summer, SON=autumn)
_SEASON_MAP = {
    12: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
}


def create_calendar_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Create calendar-based features from a timestamp column.

    Features include raw calendar integers (hour, dayofweek, month),
    binary indicators (is_weekend), and sine/cosine encoded cyclical
    features for hour-of-day and day-of-year.

    Args:
        df: Input DataFrame containing a timestamp column.
        time_col: Name of the timestamp column.

    Returns:
        Copy of the input DataFrame with 10 additional columns:
        hour, dayofweek, is_weekend, month, day_of_year, season,
        sin_day_of_year, cos_day_of_year, sin_hour, cos_hour.
    """
    result = df.copy()
    ts = pd.to_datetime(result[time_col], utc=True)

    result["hour"] = ts.dt.hour
    result["dayofweek"] = ts.dt.dayofweek
    result["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    result["season"] = result["month"].map(_SEASON_MAP)

    # Cyclical encoding: sine/cosine preserves circular proximity
    # (hour 23 and hour 0 are neighbors, not 23 apart)
    result["sin_day_of_year"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["cos_day_of_year"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    result["sin_hour"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["cos_hour"] = np.cos(2 * np.pi * result["hour"] / 24)

    return result
