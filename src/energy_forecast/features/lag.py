"""Lag-based feature engineering for time series forecasting.

All lag features enforce a minimum lag equal to the forecast horizon
to prevent data leakage. For 24h-ahead forecasting, the minimum
allowed lag is 24 hours.

The assert guard catches accidental leakage if someone adds a
shorter lag in the future.
"""

import pandas as pd

from energy_forecast.config import FORECAST_HORIZON, TARGET


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features for time series forecasting.

    All lags are >= FORECAST_HORIZON (24h) to prevent data leakage.
    The DataFrame must be sorted by time before calling this function.

    Args:
        df: Input DataFrame sorted by time, containing the target column.

    Returns:
        Copy of the input DataFrame with 7 additional columns:
        lag_24h, lag_48h, lag_168h, rolling_7d_same_hour,
        rolling_3d_same_hour, diff_24h_vs_168h, diff_24h_vs_48h.

    Raises:
        AssertionError: If any lag is shorter than FORECAST_HORIZON.
    """
    result = df.copy()

    # Primary lags (from autocorrelation analysis)
    primary_lags = [24, 48, 168]
    for lag in primary_lags:
        assert lag >= FORECAST_HORIZON, (
            f"Lag {lag} < forecast horizon {FORECAST_HORIZON}. This would cause data leakage."
        )
        result[f"lag_{lag}h"] = result[TARGET].shift(lag)

    # Rolling means: shift first, then roll to prevent leakage
    result["rolling_7d_same_hour"] = (
        result[TARGET].shift(FORECAST_HORIZON).rolling(window=7 * 24).mean()
    )
    result["rolling_3d_same_hour"] = (
        result[TARGET].shift(FORECAST_HORIZON).rolling(window=3 * 24).mean()
    )

    # Difference features: trend direction
    result["diff_24h_vs_168h"] = result["lag_24h"] - result["lag_168h"]
    result["diff_24h_vs_48h"] = result["lag_24h"] - result["lag_48h"]

    return result
