"""Holiday-based feature engineering for Spain.

Holidays show a load reduction of ~2,847 MW vs weekdays (slightly
stronger than weekends at ~2,659 MW). The model's error on weekday
holidays is double the normal weekday error (MAE 2,968 vs 1,452 MW),
confirming the need for these features.

days_since_holiday ranked #10 in feature importance, capturing the
post-holiday demand recovery ("backlog effect").
"""

import holidays as holidays_lib
import pandas as pd

from energy_forecast.config import FORECAST_HORIZON


def _days_since_holiday(date, holiday_list: list) -> int:
    """Return days since the last holiday, capped at 30.

    Args:
        date: The date to check.
        holiday_list: Sorted list of holiday dates.

    Returns:
        Number of days since the last holiday, capped at 30.
    """
    past = [h for h in holiday_list if h <= date]
    if not past:
        return 30
    return min((date - past[-1]).days, 30)


def create_holiday_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Create holiday-based features for Spain.

    Includes national holidays. Regional holidays (Catalonia,
    Andalusia, etc.) are planned for V2.

    Args:
        df: Input DataFrame containing a timestamp column.
        time_col: Name of the timestamp column.

    Returns:
        Copy of the input DataFrame with 4 additional columns:
        is_holiday, holiday_density_7d, days_since_holiday,
        is_bridge_day.
    """
    result = df.copy()
    ts = pd.to_datetime(result[time_col], utc=True)
    dates = ts.dt.date

    years = sorted(ts.dt.year.unique())
    es_holidays = holidays_lib.Spain(years=years)

    result["is_holiday"] = dates.isin(es_holidays).astype(int)

    # Holiday density: count of holidays in a ±7 day window
    holiday_dates = pd.Series(sorted(es_holidays.keys()))
    unique_dates = pd.Series(dates.unique())

    density_map = {
        d: (
            (holiday_dates >= d - pd.Timedelta(days=7))
            & (holiday_dates <= d + pd.Timedelta(days=7))
        ).sum()
        for d in unique_dates
    }
    result["holiday_density_7d"] = dates.map(density_map)

    # Days since last holiday (capped at 30)
    holiday_list = sorted(es_holidays.keys())
    days_since_map = {d: _days_since_holiday(d, holiday_list) for d in unique_dates}
    result["days_since_holiday"] = dates.map(days_since_map)

    # Bridge day: weekday between holiday and weekend
    result["_dow"] = ts.dt.dayofweek
    result["is_bridge_day"] = (
        (~result["is_holiday"].astype(bool))
        & (result["_dow"].isin([0, 4]))
        & (
            result["is_holiday"].shift(FORECAST_HORIZON).astype(bool)
            | result["is_holiday"].shift(-FORECAST_HORIZON).astype(bool)
        )
    ).astype(int)

    result = result.drop(columns=["_dow"])

    return result
