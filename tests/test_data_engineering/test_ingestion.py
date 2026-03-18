"""Tests for the DuckDB ingestion pipeline.

Small tests verify individual pipeline steps with minimal synthetic data.
One medium test runs the full pipeline end-to-end with small CSV fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engineering.ingestion import IngestionPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_energy_csv(path: Path, n_hours: int = 48) -> None:
    """Write a minimal energy CSV matching the raw dataset schema."""
    timestamps = pd.date_range("2018-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)

    data = {
        "time": timestamps.strftime("%Y-%m-%d %H:%M:%S+01:00"),
        "total load actual": rng.normal(28000, 4000, n_hours),
        "total load forecast": rng.normal(28000, 4000, n_hours),
        "generation fossil gas": rng.normal(5000, 1000, n_hours),
        "generation hydro pumped storage aggregated": [np.nan] * n_hours,
        "forecast wind offshore eday ahead": [np.nan] * n_hours,
    }

    pd.DataFrame(data).to_csv(path, index=False)


def _create_weather_csv(path: Path, n_hours: int = 48) -> None:
    """Write a minimal weather CSV with duplicates and city name issues."""
    cities = ["Barcelona", " Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]
    rng = np.random.default_rng(42)
    rows = []

    timestamps = pd.date_range("2018-01-01", periods=n_hours, freq="h", tz="UTC")

    for ts in timestamps:
        for city in cities:
            rows.append({
                "dt_iso": ts.strftime("%Y-%m-%d %H:%M:%S+01:00"),
                "city_name": city,
                "temp": rng.normal(290, 5),
                "temp_min": rng.normal(288, 5),
                "temp_max": rng.normal(292, 5),
                "pressure": rng.normal(1015, 10),
                "humidity": rng.normal(65, 15),
                "wind_speed": rng.normal(3, 1.5),
                "wind_deg": rng.normal(180, 90),
                "rain_1h": max(0, rng.normal(0, 0.5)),
                "rain_3h": 0.0,
                "snow_3h": 0.0,
                "clouds_all": rng.integers(0, 100),
                "weather_id": rng.integers(200, 800),
                "weather_main": "Clear",
                "weather_description": "clear sky",
                "weather_icon": "01d",
            })

    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Create temporary directory with raw CSV fixtures."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _create_energy_csv(raw / "energy_dataset.csv")
    _create_weather_csv(raw / "weather_features.csv")
    return raw


@pytest.fixture
def pipeline() -> IngestionPipeline:
    """Fresh in-memory DuckDB pipeline with relaxed validation for test data."""
    return IngestionPipeline(
        db_path=":memory:",
        min_expected_rows=10,
        min_expected_columns=5,
    )


# ---------------------------------------------------------------------------
# Small Tests: Individual Steps
# ---------------------------------------------------------------------------

class TestLoadRawData:
    """Tests for _load_raw_data step."""

    def test_loads_both_tables(self, pipeline: IngestionPipeline, raw_dir: Path) -> None:
        pipeline._load_raw_data(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
        )

        energy_count = pipeline.con.execute(
            "SELECT COUNT(*) FROM raw_energy"
        ).fetchone()[0]
        weather_count = pipeline.con.execute(
            "SELECT COUNT(*) FROM raw_weather"
        ).fetchone()[0]

        assert energy_count == 48
        assert weather_count > 48  # multiple cities + duplicates

    def test_raises_on_missing_file(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                energy_path=tmp_path / "nonexistent.csv",
                weather_path=tmp_path / "also_missing.csv",
                output_path=tmp_path / "out.parquet",
            )


class TestDeduplicateWeather:
    """Tests for _deduplicate_weather step."""

    def test_removes_duplicates_from_leading_space(
        self, pipeline: IngestionPipeline, raw_dir: Path
    ) -> None:
        pipeline._load_raw_data(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
        )
        pipeline._deduplicate_weather()

        cities = pipeline.con.execute(
            "SELECT DISTINCT city_name FROM weather_deduped"
        ).fetchall()
        city_names = [row[0] for row in cities]

        # " Barcelona" should be trimmed to "Barcelona"
        assert "Barcelona" in city_names
        assert " Barcelona" not in city_names

    def test_exactly_five_cities_after_dedup(
        self, pipeline: IngestionPipeline, raw_dir: Path
    ) -> None:
        pipeline._load_raw_data(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
        )
        pipeline._deduplicate_weather()

        city_count = pipeline.con.execute(
            "SELECT COUNT(DISTINCT city_name) FROM weather_deduped"
        ).fetchone()[0]

        assert city_count == 5


class TestNullRemoval:
    """Tests for _remove_null_rows step."""

    def test_drops_100_percent_null_columns(
        self, pipeline: IngestionPipeline, raw_dir: Path
    ) -> None:
        pipeline._load_raw_data(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
        )
        pipeline._deduplicate_weather()
        pipeline._aggregate_temperature_across_cities()
        pipeline._pivot_weather_by_city()
        pipeline._combine_weather_features()
        pipeline._join_energy_and_weather()
        pipeline._remove_null_rows()

        columns = [
            row[0]
            for row in pipeline.con.execute("DESCRIBE training_data").fetchall()
        ]

        assert "generation hydro pumped storage aggregated" not in columns
        assert "forecast wind offshore eday ahead" not in columns

    def test_no_null_targets_after_cleanup(
        self, pipeline: IngestionPipeline, raw_dir: Path
    ) -> None:
        pipeline._load_raw_data(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
        )
        pipeline._deduplicate_weather()
        pipeline._aggregate_temperature_across_cities()
        pipeline._pivot_weather_by_city()
        pipeline._combine_weather_features()
        pipeline._join_energy_and_weather()
        pipeline._remove_null_rows()

        null_count = pipeline.con.execute(
            "SELECT COUNT(*) FROM training_data WHERE \"total load actual\" IS NULL"
        ).fetchone()[0]

        assert null_count == 0


# ---------------------------------------------------------------------------
# Medium Test: Full Pipeline End-to-End
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end test of the complete ingestion pipeline."""

    def test_produces_valid_parquet(
        self, pipeline: IngestionPipeline, raw_dir: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "output.parquet"

        pipeline.run(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
            output_path=output_path,
        )

        assert output_path.exists()
        result = pd.read_parquet(output_path)
        assert len(result) > 0
        assert "total load actual" in result.columns

    def test_output_has_no_nulls(
        self, pipeline: IngestionPipeline, raw_dir: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "output.parquet"

        pipeline.run(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
            output_path=output_path,
        )

        result = pd.read_parquet(output_path)
        assert result.isnull().sum().sum() == 0

    def test_output_has_pivoted_weather_columns(
        self, pipeline: IngestionPipeline, raw_dir: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "output.parquet"

        pipeline.run(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
            output_path=output_path,
        )

        result = pd.read_parquet(output_path)
        expected_cols = ["pressure_Madrid", "humidity_Barcelona", "wind_speed_Bilbao"]
        for col in expected_cols:
            assert col in result.columns, f"Missing pivoted column: {col}"

    def test_output_has_aggregated_temperature(
        self, pipeline: IngestionPipeline, raw_dir: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "output.parquet"

        pipeline.run(
            energy_path=raw_dir / "energy_dataset.csv",
            weather_path=raw_dir / "weather_features.csv",
            output_path=output_path,
        )

        result = pd.read_parquet(output_path)
        assert "temp" in result.columns
        assert "temp_min" in result.columns
        assert "temp_max" in result.columns
