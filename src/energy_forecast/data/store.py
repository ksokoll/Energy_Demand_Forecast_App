"""Data store for historical load and weather data.

Provides access to historical data required for lag feature
computation at serving time. The store is defined as a Protocol
so the implementation can be swapped (File now, S3 for cloud)
without changing the consuming code.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from energy_forecast.config import MINIMUM_HISTORY_HOURS
from energy_forecast.models import InsufficientHistoryError

logger = logging.getLogger(__name__)


class DataStore(Protocol):
    """Protocol for historical data access.

    Any implementation must provide a method to retrieve
    historical load and weather data for a given time range.
    """

    def get_history(
        self,
        before: pd.Timestamp,
        hours: int = MINIMUM_HISTORY_HOURS,
    ) -> pd.DataFrame:
        """Retrieve historical data up to a given timestamp.

        Args:
            before: Fetch data before this timestamp.
            hours: Number of hours of history to retrieve.

        Returns:
            DataFrame with at least `hours` rows of historical
            data, sorted by time ascending.
        """
        ...


class _InMemoryHistoryMixin:
    """Shared get_history implementation for in-memory data stores.

    Both FileDataStore and S3DataStore load all data into self.data
    on init and share identical retrieval logic. This mixin avoids
    duplicating that logic.
    """

    data: pd.DataFrame

    def get_history(
        self,
        before: pd.Timestamp,
        hours: int = MINIMUM_HISTORY_HOURS,
    ) -> pd.DataFrame:
        """Retrieve historical data from the in-memory dataset.

        Args:
            before: Fetch data before this timestamp.
            hours: Number of hours of history to retrieve.

        Returns:
            DataFrame with `hours` rows of data ending just
            before the given timestamp, sorted by time ascending.

        Raises:
            InsufficientHistoryError: If not enough data available.
        """
        mask = self.data["time"] < before
        available = self.data[mask]

        if len(available) < hours:
            raise InsufficientHistoryError(
                required=hours,
                available=len(available),
            )

        result = available.tail(hours).copy()
        logger.debug(
            "History retrieved: %d rows before %s",
            len(result),
            before,
        )
        return result


class FileDataStore(_InMemoryHistoryMixin):
    """File-based data store for development and testing.

    Loads the full dataset into memory on initialization and
    filters by time range on each request. Supports CSV and
    Parquet formats.

    Attributes:
        data: The full historical dataset in memory.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the store by loading a CSV or Parquet file.

        Args:
            path: Path to the merged energy + weather data file.
                Supports .csv and .parquet formats.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If the file format is unsupported or
                the time column is missing.
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".parquet":
            self.data = pd.read_parquet(path)
        elif path.suffix == ".csv":
            self.data = pd.read_csv(path, parse_dates=["time"])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if "time" not in self.data.columns:
            raise ValueError(f"Data file missing 'time' column: {path}")

        self.data["time"] = pd.to_datetime(self.data["time"], utc=True)
        self.data = self.data.sort_values("time").reset_index(drop=True)

        logger.info(
            "DataStore initialized: %d rows, %s to %s",
            len(self.data),
            self.data["time"].min(),
            self.data["time"].max(),
        )


class S3DataStore(_InMemoryHistoryMixin):
    """S3-based data store for cloud deployment.

    Downloads the Parquet file from S3 on initialization,
    then behaves identically to FileDataStore via _InMemoryHistoryMixin.
    The S3 client is injectable to allow mocking in tests.

    Attributes:
        data: The full historical dataset in memory.
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        s3_client: Any = None,
    ) -> None:
        """Initialize the store by downloading a Parquet file from S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key (path to the Parquet file).
            s3_client: Optional boto3 S3 client. If None, a default
                client is created via boto3.client("s3"). Pass a mock
                or fake client in tests.

        Raises:
            ImportError: If boto3 is not installed.
            botocore.exceptions.ClientError: If the S3 object does
                not exist or access is denied.
        """
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        obj = s3_client.get_object(Bucket=bucket, Key=key)
        self.data = pd.read_parquet(BytesIO(obj["Body"].read()))

        if "time" not in self.data.columns:
            raise ValueError(f"S3 data file missing 'time' column: s3://{bucket}/{key}")

        self.data["time"] = pd.to_datetime(self.data["time"], utc=True)
        self.data = self.data.sort_values("time").reset_index(drop=True)

        logger.info(
            "S3DataStore initialized: %d rows from s3://%s/%s, %s to %s",
            len(self.data),
            bucket,
            key,
            self.data["time"].min(),
            self.data["time"].max(),
        )
