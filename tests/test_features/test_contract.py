"""Contract test for feature pipeline output.

Verifies that the feature pipeline produces exactly the columns
the trained model expects, in the correct order. This is the
Hyrum's Law protection test (SE at Google, Ch. 1).

If this test fails, the model will produce wrong predictions
without any error message.
"""

import numpy as np
import pandas as pd

from energy_forecast.config import FEATURE_COLS, TARGET
from energy_forecast.features.pipeline import build_features


def _create_minimal_dataset() -> pd.DataFrame:
    """Create a minimal dataset with all required columns.

    Must include enough rows for lag features (168+ hours)
    and all columns the pipeline expects.
    """
    n_hours = 14 * 24
    rng = np.random.default_rng(42)
    cities = ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]

    data = {
        "time": pd.date_range("2018-01-01", periods=n_hours, freq="h", tz="UTC"),
        TARGET: rng.normal(28000, 4000, n_hours),
        "temp": rng.normal(290, 5, n_hours),
    }

    for feature in ["pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]:
        for city in cities:
            data[f"{feature}_{city}"] = rng.normal(100, 10, n_hours)

    return pd.DataFrame(data)


class TestFeatureContract:
    """Tests that the pipeline output matches the model's expectations."""

    def test_output_contains_all_feature_columns(self) -> None:
        df = _create_minimal_dataset()
        result = build_features(df)
        result_clean = result.dropna()

        missing = [c for c in FEATURE_COLS if c not in result_clean.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_feature_column_order_matches_config(self) -> None:
        df = _create_minimal_dataset()
        result = build_features(df)
        result_clean = result.dropna()

        actual_order = [c for c in result_clean.columns if c in FEATURE_COLS]
        for expected, actual in zip(FEATURE_COLS, actual_order):
            assert expected == actual, f"Column order mismatch: expected {expected}, got {actual}"

    def test_feature_count_matches_config(self) -> None:
        df = _create_minimal_dataset()
        result = build_features(df)
        result_clean = result.dropna()

        available = [c for c in FEATURE_COLS if c in result_clean.columns]
        assert len(available) == len(FEATURE_COLS), (
            f"Expected {len(FEATURE_COLS)} features, got {len(available)}"
        )

    def test_no_nulls_in_feature_columns_after_dropna(self) -> None:
        df = _create_minimal_dataset()
        result = build_features(df)
        result_clean = result.dropna(subset=FEATURE_COLS)

        null_counts = result_clean[FEATURE_COLS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"Null values in feature columns: {cols_with_nulls.to_dict()}"
        )
