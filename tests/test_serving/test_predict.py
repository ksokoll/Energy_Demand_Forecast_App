"""Tests for the prediction orchestrator.

Medium tests: uses real feature pipeline but mocked store and model.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import FORECAST_HORIZON, TARGET
from energy_forecast.models import ModelNotFoundError
from energy_forecast.serving.predict import ForecastService


def _create_mock_history(n_hours: int = 192) -> pd.DataFrame:
    """Create synthetic history data matching the merged schema."""
    rng = np.random.default_rng(42)
    cities = ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]

    data = {
        "time": pd.date_range("2018-06-07", periods=n_hours, freq="h", tz="UTC"),
        TARGET: rng.normal(28000, 4000, n_hours),
        "temp": rng.normal(290, 5, n_hours),
    }

    for feature in ["pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]:
        for city in cities:
            data[f"{feature}_{city}"] = rng.normal(100, 10, n_hours)

    return pd.DataFrame(data)


@pytest.fixture
def mock_store() -> MagicMock:
    """DataStore mock returning synthetic history."""
    store = MagicMock()
    store.get_history.return_value = _create_mock_history()
    return store


@pytest.fixture
def service(mock_store: MagicMock, tmp_path: Path) -> ForecastService:
    """ForecastService with mocked store and a dummy model."""
    import lightgbm as lgb

    from energy_forecast.config import FEATURE_COLS

    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLS))),
        columns=FEATURE_COLS,
    )
    y = rng.normal(28000, 4000, n)

    model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
    model.fit(X, y)

    model_path = tmp_path / "test_model.lgb"
    model.booster_.save_model(str(model_path))

    return ForecastService(store=mock_store, model_path=model_path)


class TestForecastService:
    """Tests for ForecastService."""

    def test_predict_returns_24_predictions(self, service: ForecastService) -> None:
        result = service.predict(pd.Timestamp("2018-06-15", tz="UTC"))
        assert len(result.predictions) == FORECAST_HORIZON

    def test_predictions_have_correct_timestamps(self, service: ForecastService) -> None:
        forecast_from = pd.Timestamp("2018-06-15", tz="UTC")
        result = service.predict(forecast_from)

        expected_hours = pd.date_range(start=forecast_from, periods=24, freq="h")
        actual_hours = [p.timestamp for p in result.predictions]
        for expected, actual in zip(expected_hours, actual_hours):
            assert expected == actual

    def test_predictions_are_plausible_range(self, service: ForecastService) -> None:
        result = service.predict(pd.Timestamp("2018-06-15", tz="UTC"))
        for p in result.predictions:
            # Spanish load roughly 15,000 to 45,000 MW
            assert 5000 < p.load_mw < 60000

    def test_model_version_in_response(self, service: ForecastService) -> None:
        result = service.predict(pd.Timestamp("2018-06-15", tz="UTC"))
        assert result.model_version == "v1"

    def test_raises_on_missing_model(self, mock_store: MagicMock) -> None:
        with pytest.raises(ModelNotFoundError):
            ForecastService(
                store=mock_store,
                model_path=Path("/nonexistent/model.lgb"),
            )
