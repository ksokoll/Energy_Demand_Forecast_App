"""Application entrypoint.

Initializes the data store, forecast service, and FastAPI app.

Local development:
    uvicorn energy_forecast.main:app --reload

Cloud deployment (ECS / App Runner):
    Set S3_DATA_BUCKET, S3_MODEL_BUCKET env vars to enable S3 I/O.
    Optional: S3_DATA_KEY, S3_MODEL_KEY (default to standard filenames).
"""

import logging
import os
import tempfile
from pathlib import Path

from energy_forecast.config import DATA_PATH, MODEL_PATH
from energy_forecast.data.store import FileDataStore, S3DataStore
from energy_forecast.serving.api import create_app
from energy_forecast.serving.predict import ForecastService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def _create_store() -> FileDataStore | S3DataStore:
    """Select data store based on environment.

    Returns S3DataStore if S3_DATA_BUCKET is set, otherwise
    falls back to FileDataStore for local development.
    """
    if bucket := os.environ.get("S3_DATA_BUCKET"):
        key = os.environ.get("S3_DATA_KEY", "energy_weather_merged.parquet")
        logger.info("Cloud mode: loading data from s3://%s/%s", bucket, key)
        return S3DataStore(bucket=bucket, key=key)

    logger.info("Local mode: loading data from %s", DATA_PATH)
    return FileDataStore(path=DATA_PATH)


def _resolve_model_path() -> Path:
    """Download model from S3 to a temp file if S3_MODEL_BUCKET is set.

    Returns the local MODEL_PATH in development, or a temporary file
    path after downloading from S3 in cloud deployment. ForecastService
    remains unchanged: it always receives a local Path.

    Note: The temp file is intentionally not deleted after creation.
    ForecastService loads it on startup and keeps the Booster in memory.
    The file is cleaned up on process exit via atexit handler.
    """
    if bucket := os.environ.get("S3_MODEL_BUCKET"):
        key = os.environ.get("S3_MODEL_KEY", "model_v1.lgb")
        logger.info("Cloud mode: downloading model from s3://%s/%s", bucket, key)

        import atexit

        import boto3

        tmp = tempfile.NamedTemporaryFile(suffix=".lgb", delete=False)
        try:
            boto3.client("s3").download_fileobj(bucket, key, tmp)
            tmp.flush()
        finally:
            tmp.close()

        # Clean up temp file when process exits
        atexit.register(os.unlink, tmp.name)

        return Path(tmp.name)

    logger.info("Local mode: loading model from %s", MODEL_PATH)
    return MODEL_PATH


store = _create_store()
service = ForecastService(store=store, model_path=_resolve_model_path())
app = create_app(service)
