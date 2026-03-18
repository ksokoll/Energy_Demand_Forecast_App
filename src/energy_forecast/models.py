"""Pydantic schemas and domain models.

These schemas serve as contracts between components,
preventing training-serving skew and ensuring API consistency.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class HourlyPrediction(BaseModel):
    """Single hourly load prediction.

    Attributes:
        timestamp: UTC timestamp for this prediction hour.
        load_mw: Predicted electrical load in megawatts.
    """

    timestamp: datetime
    load_mw: float = Field(description="Predicted load in megawatts")


class ForecastRequest(BaseModel):
    """Request for a 24h-ahead forecast.

    The API will predict the next 24 hours starting from
    the provided timestamp. Historical load data must be
    available in the data store for at least 168 hours
    before this timestamp (required for lag features).

    Attributes:
        forecast_from: Start timestamp for the forecast.
    """

    forecast_from: datetime = Field(
        description="Start timestamp for the forecast. "
        "The API will predict the next 24 hours from this point."
    )


class ForecastResponse(BaseModel):
    """Response containing 24 hourly predictions.

    Attributes:
        forecast_from: The requested forecast start timestamp.
        predictions: List of 24 hourly predictions.
        model_version: Version identifier of the model used.
        feature_count: Number of features used by the model.
    """

    forecast_from: datetime
    predictions: list[HourlyPrediction]
    model_version: str = "v1"
    feature_count: int = Field(default=51)


class HealthResponse(BaseModel):
    """Health check response for monitoring.

    Attributes:
        status: Service status string.
        model_loaded: Whether the model artifact is loaded.
        model_version: Version of the loaded model.
    """

    status: str = "ok"
    model_loaded: bool
    model_version: str


class ForecastError(Exception):
    """Base exception for forecast service errors."""


class InsufficientHistoryError(ForecastError):
    """Raised when the data store has not enough historical data.

    Attributes:
        required: Number of hours required.
        available: Number of hours available.
    """

    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(f"Insufficient history: need {required} hours, got {available}")


class ModelNotFoundError(ForecastError):
    """Raised when the model artifact cannot be loaded.

    Attributes:
        path: Path where the model was expected.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Model not found: {path}")
