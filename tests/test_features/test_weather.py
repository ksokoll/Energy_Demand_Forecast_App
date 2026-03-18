"""Tests for weather feature engineering.

Focuses on Kelvin conversion, U-shape encoding, and the implicit
dependency on calendar features (is_weekend, sin_hour must exist).
"""

import pandas as pd
import pytest

from energy_forecast.config import COMFORT_TEMPERATURE
from energy_forecast.features.calendar import create_calendar_features
from energy_forecast.features.weather import create_weather_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """48 hours of data with calendar features pre-applied."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2018-06-15", periods=48, freq="h", tz="UTC"),
            "temp": [290.0] * 48,  # 16.85°C
        }
    )
    return create_calendar_features(df)


class TestWeatherFeatures:
    """Tests for create_weather_features()."""

    def test_adds_all_expected_columns(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        expected = [
            "temp_celsius",
            "temp_deviation",
            "temp_deviation_sq",
            "temp_dev_x_weekend",
            "temp_dev_x_hour_sin",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_kelvin_to_celsius_conversion(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        expected_celsius = 290.0 - 273.15
        assert abs(result["temp_celsius"].iloc[0] - expected_celsius) < 0.001

    def test_temp_deviation_from_comfort_zone(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        celsius = 290.0 - 273.15  # 16.85
        expected_deviation = abs(celsius - COMFORT_TEMPERATURE)
        assert abs(result["temp_deviation"].iloc[0] - expected_deviation) < 0.001

    def test_temp_deviation_is_always_positive(self) -> None:
        """Both hot and cold should produce positive deviation."""
        cold = pd.DataFrame(
            {
                "time": pd.date_range("2018-01-15", periods=24, freq="h", tz="UTC"),
                "temp": [268.15] * 24,  # -5°C
            }
        )
        hot = pd.DataFrame(
            {
                "time": pd.date_range("2018-07-15", periods=24, freq="h", tz="UTC"),
                "temp": [313.15] * 24,  # 40°C
            }
        )
        cold = create_calendar_features(cold)
        hot = create_calendar_features(hot)

        cold_result = create_weather_features(cold)
        hot_result = create_weather_features(hot)

        assert (cold_result["temp_deviation"] > 0).all()
        assert (hot_result["temp_deviation"] > 0).all()

    def test_deviation_squared_is_correct(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        row = result.iloc[0]
        assert abs(row["temp_deviation_sq"] - row["temp_deviation"] ** 2) < 0.001

    def test_weekend_interaction_zero_on_weekdays(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        weekday_rows = result[result["is_weekend"] == 0]
        assert (weekday_rows["temp_dev_x_weekend"] == 0).all()

    def test_weekend_interaction_nonzero_on_weekends(self, sample_df: pd.DataFrame) -> None:
        result = create_weather_features(sample_df)
        weekend_rows = result[result["is_weekend"] == 1]
        assert (weekend_rows["temp_dev_x_weekend"] > 0).all()

    def test_fails_without_calendar_features(self) -> None:
        """Weather features depend on calendar features existing."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2018-06-15", periods=24, freq="h", tz="UTC"),
                "temp": [290.0] * 24,
            }
        )
        with pytest.raises(KeyError):
            create_weather_features(df)

    def test_does_not_modify_input(self, sample_df: pd.DataFrame) -> None:
        original_cols = sample_df.columns.tolist()
        create_weather_features(sample_df)
        assert sample_df.columns.tolist() == original_cols
