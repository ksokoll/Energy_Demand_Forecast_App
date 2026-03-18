"""Tests for calendar feature engineering.

Small tests: no I/O, no external dependencies, synthetic data only.
Tests verify behaviors, not methods (SE at Google, Ch. 12).
"""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.calendar import create_calendar_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """One week of hourly timestamps for testing."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2018-06-11", periods=168, freq="h", tz="UTC"),
        }
    )


class TestCalendarFeatures:
    """Tests for create_calendar_features()."""

    def test_adds_all_expected_columns(self, sample_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_df)
        expected = [
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
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_input(self, sample_df: pd.DataFrame) -> None:
        original_cols = sample_df.columns.tolist()
        create_calendar_features(sample_df)
        assert sample_df.columns.tolist() == original_cols

    def test_hour_range_is_0_to_23(self, sample_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_df)
        assert result["hour"].min() == 0
        assert result["hour"].max() == 23

    def test_weekend_flag_matches_saturday_sunday(self, sample_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_df)
        weekends = result[result["is_weekend"] == 1]
        # dayofweek 5=Saturday, 6=Sunday
        assert set(weekends["dayofweek"].unique()) == {5, 6}

    def test_sin_cos_hour_preserves_circularity(self, sample_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_df)
        hour_0 = result[result["hour"] == 0].iloc[0]
        hour_23 = result[result["hour"] == 23].iloc[0]
        hour_1 = result[result["hour"] == 1].iloc[0]

        dist_23_to_0 = np.sqrt(
            (hour_0["sin_hour"] - hour_23["sin_hour"]) ** 2
            + (hour_0["cos_hour"] - hour_23["cos_hour"]) ** 2
        )
        dist_0_to_1 = np.sqrt(
            (hour_1["sin_hour"] - hour_0["sin_hour"]) ** 2
            + (hour_1["cos_hour"] - hour_0["cos_hour"]) ** 2
        )
        # Hour 23→0 should be the same distance as 0→1
        assert abs(dist_23_to_0 - dist_0_to_1) < 0.001

    def test_season_mapping_is_correct(self, sample_df: pd.DataFrame) -> None:
        # June = summer = season 2
        result = create_calendar_features(sample_df)
        assert (result["season"] == 2).all()

    def test_row_count_unchanged(self, sample_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_df)
        assert len(result) == len(sample_df)
