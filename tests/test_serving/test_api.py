"""Tests for the FastAPI serving endpoints.

Medium tests: uses FastAPI TestClient (local HTTP), no external
services, synthetic data via fixtures.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from energy_forecast.models import ForecastResponse, HourlyPrediction
from energy_forecast.serving.api import create_app
from energy_forecast.serving.predict import ForecastService


@pytest.fixture
def mock_service() -> MagicMock:
    """ForecastService mock that returns a valid response."""
    service = MagicMock(spec=ForecastService)
    service.model = True  # model_loaded check

    predictions = [
        HourlyPrediction(
            timestamp=datetime(2018, 6, 15, h, tzinfo=timezone.utc),
            load_mw=25000.0 + h * 500,
        )
        for h in range(24)
    ]
    service.predict.return_value = ForecastResponse(
        forecast_from=datetime(2018, 6, 15, tzinfo=timezone.utc),
        predictions=predictions,
    )
    return service


@pytest.fixture
def client(mock_service: MagicMock) -> TestClient:
    """FastAPI test client with mocked service."""
    app = create_app(mock_service)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_reports_model_loaded(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["model_version"] == "v1"


class TestForecastEndpoint:
    """Tests for POST /forecast."""

    def test_returns_200_with_valid_request(self, client: TestClient) -> None:
        response = client.post(
            "/forecast",
            json={"forecast_from": "2018-06-15T00:00:00+00:00"},
        )
        assert response.status_code == 200

    def test_returns_24_predictions(self, client: TestClient) -> None:
        data = client.post(
            "/forecast",
            json={"forecast_from": "2018-06-15T00:00:00+00:00"},
        ).json()
        assert len(data["predictions"]) == 24

    def test_returns_422_with_invalid_body(self, client: TestClient) -> None:
        response = client.post("/forecast", json={"wrong_field": "value"})
        assert response.status_code == 422

    def test_returns_400_on_insufficient_history(
        self, mock_service: MagicMock, client: TestClient
    ) -> None:
        from energy_forecast.models import InsufficientHistoryError

        mock_service.predict.side_effect = InsufficientHistoryError(required=192, available=50)
        response = client.post(
            "/forecast",
            json={"forecast_from": "2015-01-01T00:00:00+00:00"},
        )
        assert response.status_code == 400
        assert "Insufficient history" in response.json()["detail"]

    def test_returns_500_on_unexpected_error(
        self, mock_service: MagicMock, client: TestClient
    ) -> None:
        mock_service.predict.side_effect = RuntimeError("Something broke")
        response = client.post(
            "/forecast",
            json={"forecast_from": "2018-06-15T00:00:00+00:00"},
        )
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]

    def test_health_reports_model_not_loaded(self) -> None:
        service = MagicMock(spec=ForecastService)
        service.model = None
        app = create_app(service)
        client = TestClient(app)

        data = client.get("/health").json()
        assert data["model_loaded"] is False
