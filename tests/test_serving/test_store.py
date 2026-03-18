"""Tests for the data store.

Medium tests: uses temporary files for Parquet/CSV loading.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.store import FileDataStore
from energy_forecast.models import InsufficientHistoryError


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Two weeks of synthetic data."""
    n_hours = 14 * 24
    return pd.DataFrame(
        {
            "time": pd.date_range("2018-01-01", periods=n_hours, freq="h", tz="UTC"),
            "total load actual": np.random.default_rng(42).normal(28000, 4000, n_hours),
            "temp": np.random.default_rng(42).normal(290, 5, n_hours),
        }
    )


@pytest.fixture
def parquet_store(sample_data: pd.DataFrame, tmp_path: Path) -> FileDataStore:
    """FileDataStore loaded from a Parquet file."""
    path = tmp_path / "test_data.parquet"
    sample_data.to_parquet(path, index=False)
    return FileDataStore(path=path)


@pytest.fixture
def csv_store(sample_data: pd.DataFrame, tmp_path: Path) -> FileDataStore:
    """FileDataStore loaded from a CSV file."""
    path = tmp_path / "test_data.csv"
    sample_data.to_csv(path, index=False)
    return FileDataStore(path=path)


class TestFileDataStoreInit:
    """Tests for store initialization."""

    def test_loads_parquet(self, parquet_store: FileDataStore) -> None:
        assert len(parquet_store.data) == 14 * 24

    def test_loads_csv(self, csv_store: FileDataStore) -> None:
        assert len(csv_store.data) == 14 * 24

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FileDataStore(path=tmp_path / "nonexistent.parquet")

    def test_raises_on_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xlsx"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            FileDataStore(path=path)

    def test_raises_on_missing_time_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_data.parquet"
        pd.DataFrame({"value": [1, 2, 3]}).to_parquet(path)
        with pytest.raises(ValueError, match="missing 'time' column"):
            FileDataStore(path=path)

    def test_data_is_sorted_by_time(self, parquet_store: FileDataStore) -> None:
        times = parquet_store.data["time"]
        assert times.is_monotonic_increasing


class TestGetHistory:
    """Tests for get_history()."""

    def test_returns_correct_number_of_rows(self, parquet_store: FileDataStore) -> None:
        before = pd.Timestamp("2018-01-10", tz="UTC")
        result = parquet_store.get_history(before=before, hours=48)
        assert len(result) == 48

    def test_all_rows_are_before_timestamp(self, parquet_store: FileDataStore) -> None:
        before = pd.Timestamp("2018-01-10", tz="UTC")
        result = parquet_store.get_history(before=before, hours=48)
        assert (result["time"] < before).all()

    def test_returns_most_recent_rows(self, parquet_store: FileDataStore) -> None:
        before = pd.Timestamp("2018-01-10", tz="UTC")
        result = parquet_store.get_history(before=before, hours=48)
        # Last row should be the hour just before the timestamp
        last_time = result["time"].iloc[-1]
        assert last_time == pd.Timestamp("2018-01-09 23:00:00", tz="UTC")

    def test_raises_on_insufficient_history(self, parquet_store: FileDataStore) -> None:
        before = pd.Timestamp("2018-01-01 12:00:00", tz="UTC")
        with pytest.raises(InsufficientHistoryError):
            parquet_store.get_history(before=before, hours=168)

    def test_returns_copy_not_view(self, parquet_store: FileDataStore) -> None:
        before = pd.Timestamp("2018-01-10", tz="UTC")
        result = parquet_store.get_history(before=before, hours=48)
        result["time"] = None
        # Original data should be unaffected
        assert parquet_store.data["time"].notna().all()
