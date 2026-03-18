"""Prediction orchestrator for 24h-ahead energy demand forecasting.

Coordinates the full prediction flow: fetch historical data from
the store, build feature vectors for each forecast hour, run the
model, and return structured results.

This is the equivalent of handler.py in the car damage project.
"""

import logging
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_forecast.config import (
    FEATURE_COLS,
    FORECAST_HORIZON,
    MINIMUM_HISTORY_HOURS,
    MODEL_PATH,
)
from energy_forecast.data.store import DataStore
from energy_forecast.features.pipeline import build_features
from energy_forecast.models import (
    ForecastResponse,
    HourlyPrediction,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


class ForecastService:
    """Orchestrates the prediction pipeline.

    Loads the trained model on initialization and provides
    a predict method that fetches history, builds features,
    and returns 24 hourly predictions.

    Attributes:
        model: The loaded LightGBM booster.
        store: The data store for historical data access.
    """

    def __init__(self, store: DataStore, model_path: Path = MODEL_PATH) -> None:
        """Initialize the service with a data store and model.

        Args:
            store: Implementation of the DataStore protocol.
            model_path: Path to the trained LightGBM model file.

        Raises:
            ModelNotFoundError: If the model file does not exist.
        """
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        self.model = lgb.Booster(model_file=str(model_path))
        self.store = store
        logger.info("ForecastService initialized, model loaded from %s", model_path)

    def predict(self, forecast_from: pd.Timestamp) -> ForecastResponse:
        """Generate a 24h-ahead forecast.

        Args:
            forecast_from: Start timestamp for the forecast.
                Historical data must be available for at least
                168 hours before this timestamp.

        Returns:
            ForecastResponse with 24 hourly predictions.

        Raises:
            InsufficientHistoryError: If not enough historical data.
        """
        logger.info("Forecast requested for %s", forecast_from)

        history = self.store.get_history(
            before=forecast_from,
            hours=MINIMUM_HISTORY_HOURS + FORECAST_HORIZON,
        )
        logger.info("Retrieved %d rows of history", len(history))

        forecast_rows = self._build_forecast_rows(history, forecast_from)
        features_df = build_features(forecast_rows)

        features_df = features_df.dropna(subset=FEATURE_COLS)

        forecast_mask = features_df["time"] >= forecast_from
        forecast_features = features_df.loc[forecast_mask, FEATURE_COLS]
        raw_predictions = self.model.predict(forecast_features)

        forecast_timestamps = pd.date_range(
            start=forecast_from,
            periods=FORECAST_HORIZON,
            freq="h",
        )

        predictions = [
            HourlyPrediction(
                timestamp=ts.to_pydatetime(),
                load_mw=round(float(pred), 1),
            )
            for ts, pred in zip(forecast_timestamps, raw_predictions)
        ]

        logger.info(
            "Forecast complete: %d predictions, range %.0f - %.0f MW",
            len(predictions),
            min(p.load_mw for p in predictions),
            max(p.load_mw for p in predictions),
        )

        return ForecastResponse(
            forecast_from=forecast_from.to_pydatetime(),
            predictions=predictions,
        )

    def _build_forecast_rows(
        self,
        history: pd.DataFrame,
        forecast_from: pd.Timestamp,
    ) -> pd.DataFrame:
        """Combine history with placeholder rows for forecast hours.

        Creates a DataFrame with historical data followed by 24
        rows for the forecast period. Weather values for forecast
        hours use the values from one week ago as proxy.

        Args:
            history: Historical data from the data store.
            forecast_from: Start of the forecast period.

        Returns:
            Combined DataFrame ready for feature engineering.
        """
        forecast_timestamps = pd.date_range(
            start=forecast_from,
            periods=FORECAST_HORIZON,
            freq="h",
        )

        one_week_ago = forecast_from - timedelta(hours=168)
        weather_proxy = history[history["time"] >= one_week_ago].head(FORECAST_HORIZON)

        forecast_rows = weather_proxy.copy()
        forecast_rows["time"] = forecast_timestamps
        forecast_rows["total load actual"] = np.nan

        return pd.concat([history, forecast_rows], ignore_index=True)
