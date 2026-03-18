"""FastAPI application for energy demand forecasting.

Provides two endpoints: /forecast for 24h-ahead predictions
and /health for monitoring.
"""

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

from energy_forecast.models import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    InsufficientHistoryError,
)
from energy_forecast.serving.predict import ForecastService

logger = logging.getLogger(__name__)


def create_app(service: ForecastService) -> FastAPI:
    """Create and configure the FastAPI application.

    Uses a factory function so the service can be injected,
    making testing easier.

    Args:
        service: Initialized ForecastService instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Energy Demand Forecast API",
        description="24h-ahead energy demand forecasting for Spain",
        version="0.1.0",
    )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        """Generate a 24h-ahead energy demand forecast.

        Args:
            request: ForecastRequest with forecast_from timestamp.

        Returns:
            ForecastResponse with 24 hourly predictions.
        """
        try:
            timestamp = pd.Timestamp(request.forecast_from).tz_convert("UTC")
            return service.predict(timestamp)
        except InsufficientHistoryError as e:
            logger.warning("Insufficient history: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Prediction failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Check service health and model status."""
        return HealthResponse(
            status="ok",
            model_loaded=service.model is not None,
            model_version="v1",
        )

    return app
