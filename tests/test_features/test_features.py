"""Tests for lag feature engineering.

Small tests focused on leakage prevention and correct shifting.
"""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import FORECAST_HORIZON, TARGET
from energy_forecast.features.lag import create_lag_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Two weeks of hourly data with synthetic load values."""
    n_hours = 14 * 24  # 336 hours
    return pd.DataFrame(
        {
            "time": pd.date_range("2018-01-01", periods=n_hours, freq="h", tz="UTC"),
            TARGET: np.random.default_rng(42).normal(28000, 4000, n_hours),
        }
    )


class TestLagFeatures:
    """Tests for create_lag_features()."""

    def test_adds_all_expected_columns(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        expected = [
            "lag_24h",
            "lag_48h",
            "lag_168h",
            "rolling_7d_same_hour",
            "rolling_3d_same_hour",
            "diff_24h_vs_168h",
            "diff_24h_vs_48h",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_lag_24h_is_correct_value(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        # Row 24 should have lag_24h equal to row 0's target
        assert result.loc[24, "lag_24h"] == sample_df.loc[0, TARGET]

    def test_lag_168h_is_correct_value(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        # Row 168 should have lag_168h equal to row 0's target
        assert result.loc[168, "lag_168h"] == sample_df.loc[0, TARGET]

    def test_no_leakage_all_lags_ge_horizon(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        # First non-null lag value must be at index >= FORECAST_HORIZON
        for col in ["lag_24h", "lag_48h", "lag_168h"]:
            first_valid = result[col].first_valid_index()
            assert first_valid >= FORECAST_HORIZON, (
                f"{col} has data at index {first_valid}, "
                f"before forecast horizon {FORECAST_HORIZON}"
            )

    def test_diff_features_are_correct(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        valid = result.dropna()
        row = valid.iloc[0]
        assert row["diff_24h_vs_48h"] == row["lag_24h"] - row["lag_48h"]
        assert row["diff_24h_vs_168h"] == row["lag_24h"] - row["lag_168h"]

    def test_first_168_rows_have_nulls(self, sample_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_df)
        # lag_168h needs 168 rows of history, so first 168 must be null
        assert result["lag_168h"].iloc[:168].isna().all()

    def test_does_not_modify_input(self, sample_df: pd.DataFrame) -> None:
        original_len = len(sample_df.columns)
        create_lag_features(sample_df)
        assert len(sample_df.columns) == original_len
