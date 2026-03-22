"""DuckDB-based data ingestion pipeline for energy and weather data.

Automates the data cleaning and merging steps discovered during EDA
in the Kaggle notebook. This pipeline handles the Data Engineering
part of the system. Its single responsibility is to transform raw
CSVs into a clean, merged Parquet file ready for the serving layer.

Feature engineering is NOT part of this pipeline. It stays in Python
(energy_forecast.features) to prevent training-serving skew.
Re-using the same feature code in training and serving is critical
(Google Rules of ML #32, #37).

Pipeline steps:
    1. Load raw CSVs into DuckDB
    2. Deduplicate weather (multiple conditions per timestamp)
    3. Aggregate temperature across cities (low inter-city variance)
    4. Pivot high-variance weather features by city
    5. Combine weather features into single table
    6. Join energy and weather on timestamp
    7. Remove null rows (documented strategy from EDA)
    8. Validate output schema (Shifting Left principle)
    9. Export to Parquet
"""

import logging
from pathlib import Path

import duckdb

from energy_forecast.config import TARGET

logger = logging.getLogger(__name__)

# Weather features with low inter-city variance (CV < 2%, verified in EDA)
_LOW_VARIANCE_FEATURES = ["temp", "temp_min", "temp_max"]

# Weather features with high inter-city variance (CV > 10%, verified in EDA)
# See notebook section "Weather Feature Selection: Aggregate vs. Pivot"
_HIGH_VARIANCE_FEATURES = ["pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]

# Cities in the weather dataset
_CITIES = ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]

# Columns that are 100% null in the raw dataset (discovered in EDA)
_DROP_COLUMNS = [
    "generation hydro pumped storage aggregated",
    "forecast wind offshore eday ahead",
]

# Expected minimums for output validation (Shifting Left)
_MIN_EXPECTED_ROWS = 30000
_MIN_EXPECTED_COLUMNS = 50


class IngestionPipeline:
    """DuckDB-based ingestion pipeline for energy and weather data.

    Orchestrates the full data cleaning and merging process using SQL
    transformations. Each step corresponds to a discovery made during
    EDA and is documented with its rationale.

    This class follows Google Rules of ML #4 (get infrastructure right)
    and #5 (test infrastructure independently from ML). The pipeline
    is fully testable without any ML dependencies.

    Attributes:
        con: DuckDB connection (in-memory or file-based).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        min_expected_rows: int = _MIN_EXPECTED_ROWS,
        min_expected_columns: int = _MIN_EXPECTED_COLUMNS,
    ) -> None:
        """Initialize the pipeline with a DuckDB connection.

        Args:
            db_path: Path to DuckDB file, or ":memory:" for in-memory.
            min_expected_rows: Minimum row count for output validation.
                Override in tests with small fixture data.
            min_expected_columns: Minimum column count for output validation.
        """
        self.con = duckdb.connect(db_path)
        # NOTE: "self.con" is a library standard of DuckDB, but can seem un-obvious
        # to readers unfamiliar with DuckDB. For the future, clearer naming would be: "self.connection"
        self.min_expected_rows = min_expected_rows
        self.min_expected_columns = min_expected_columns
        logger.info("DuckDB connection initialized: %s", db_path)

    def run(
        self,
        energy_path: Path,
        weather_path: Path,
        output_path: Path,
    ) -> None:
        """Execute the full ingestion pipeline.

        Runs all steps sequentially: load, clean, transform, merge,
        validate, export. If validation fails, no output is written.

        Args:
            energy_path: Path to the raw energy CSV file.
            weather_path: Path to the raw weather CSV file.
            output_path: Path for the output Parquet file.

        Raises:
            FileNotFoundError: If an input file does not exist.
            AssertionError: If output validation fails.
        """
        if not energy_path.exists():
            raise FileNotFoundError(f"Energy data not found: {energy_path}")
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather data not found: {weather_path}")

        logger.info("Starting ingestion pipeline")

        self._load_raw_data(energy_path, weather_path)
        self._deduplicate_weather()
        self._aggregate_temperature_across_cities()
        self._pivot_weather_by_city()
        self._combine_weather_features()
        self._join_energy_and_weather()
        self._remove_null_rows()
        self._validate_output()
        self._export_to_parquet(output_path)

        row_count = self.con.execute(
            "SELECT COUNT(*) FROM training_data"
        ).fetchone()[0]
        logger.info(
            "Pipeline complete: %d rows written to %s",
            row_count,
            output_path,
        )

    def _load_raw_data(self, energy_path: Path, weather_path: Path) -> None:
        """Load raw CSV files into DuckDB tables.

        DuckDB's read_csv() does not support parameterized file paths.
        Paths are inserted via f-string. This is safe because paths
        come from config.py, not from user input. Documented per
        Google Style Guide requirement to explain non-obvious decisions.

        Args:
            energy_path: Path to the raw energy CSV.
            weather_path: Path to the raw weather CSV.
        """
        logger.info("Loading raw data")

        self.con.execute(f"""
            CREATE TABLE raw_energy AS
            SELECT * FROM read_csv('{energy_path}', auto_detect=true)
        """)
        # NOTE: auto_detect=true is acceptable for static Kaggle CSVs.
        # For production pipelines with daily data delivery,
        # replace with explicit column types to prevent
        # silent type inference failures.

        self.con.execute(f"""
            CREATE TABLE raw_weather AS
            SELECT * FROM read_csv('{weather_path}', auto_detect=true)
        """)

        energy_count = self.con.execute(
            "SELECT COUNT(*) FROM raw_energy"
        ).fetchone()[0]
        weather_count = self.con.execute(
            "SELECT COUNT(*) FROM raw_weather"
        ).fetchone()[0]
        logger.info(
            "Loaded: %d energy rows, %d weather rows",
            energy_count,
            weather_count,
        )

    def _deduplicate_weather(self) -> None:
        """Remove duplicate weather entries per timestamp and city.

        The raw weather data logs multiple weather conditions for the
        same hour and city (e.g. "rain" and "thunderstorm with rain").
        Verified in EDA: all numerical values are identical across
        duplicates (checked via nunique() per group). Only the
        categorical weather description differs.

        Also fixes a leading whitespace in the "Barcelona" city name,
        a known data quality issue discovered during EDA.
        """
        logger.info("Deduplicating weather data")

        self.con.execute("""
            CREATE TABLE weather_deduped AS
            SELECT * FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY dt_iso, TRIM(city_name)
                        ORDER BY weather_id
                    ) AS row_num
                FROM raw_weather
            )
            WHERE row_num = 1
        """)

        self.con.execute("ALTER TABLE weather_deduped DROP COLUMN row_num")
        self.con.execute("UPDATE weather_deduped SET city_name = TRIM(city_name)")

        count = self.con.execute(
            "SELECT COUNT(*) FROM weather_deduped"
        ).fetchone()[0]
        logger.info("Weather after dedup: %d rows", count)

    def _aggregate_temperature_across_cities(self) -> None:
        """Average temperature features across all five cities.

        Temperature has CV < 2% between cities (verified in EDA).
        The national mean preserves nearly all information.
        This is safe to aggregate because Spain's five major cities
        show near-identical temperature patterns.

        Features aggregated: temp, temp_min, temp_max.
        """
        logger.info("Aggregating temperature across cities")

        agg_columns = ", ".join(
            f"AVG({col}) AS {col}" for col in _LOW_VARIANCE_FEATURES
        )

        self.con.execute(f"""
            CREATE TABLE weather_temperature AS
            SELECT dt_iso, {agg_columns}
            FROM weather_deduped
            GROUP BY dt_iso
        """)

    def _pivot_weather_by_city(self) -> None:
        """Pivot high-variance weather features into per-city columns.

        Features like wind_speed (CV 62%), rain_1h (CV 181%), and
        clouds_all (CV 104%) differ substantially between cities.
        Inter-city correlation analysis confirmed most pairs have
        r < 0.3. Averaging would destroy signal.

        Creates columns like pressure_Madrid, humidity_Barcelona, etc.
        LightGBM handles the resulting correlated features well
        (splits only where useful, ignores the rest).

        Excluded from pivot:
            wind_deg: circular variable, numeric mean is meaningless
            rain_3h, snow_3h: near-zero values, no usable signal
        """
        logger.info("Pivoting weather features by city")

        pivot_expressions = []
        for feature in _HIGH_VARIANCE_FEATURES:
            for city in _CITIES:
                pivot_expressions.append(
                    f"MAX(CASE WHEN city_name = '{city}' "
                    f"THEN {feature} END) AS {feature}_{city}"
                )

        select_clause = ",\n            ".join(pivot_expressions)

        self.con.execute(f"""
            CREATE TABLE weather_pivoted AS
            SELECT dt_iso,
                {select_clause}
            FROM weather_deduped
            GROUP BY dt_iso
        """)

    def _combine_weather_features(self) -> None:
        """Join aggregated temperature and pivoted city-level features.

        Produces a single weather table with one row per timestamp:
        3 aggregated temperature columns + 25 city-specific columns
        (5 features x 5 cities).
        """
        logger.info("Combining weather features")

        self.con.execute("""
            CREATE TABLE weather_combined AS
            SELECT t.*, p.* EXCLUDE (dt_iso)
            FROM weather_temperature t
            JOIN weather_pivoted p USING (dt_iso)
        """)

    def _join_energy_and_weather(self) -> None:
        """Merge energy and weather data on timestamp.

        Uses LEFT JOIN to preserve all energy rows even if weather
        data has gaps. The merge key is energy.time = weather.dt_iso.
        Both timestamps cover 2015-01-01 to 2018-12-31 at hourly
        granularity (verified during EDA: exact 1:1 match after
        weather deduplication).
        """
        logger.info("Joining energy and weather data")

        self.con.execute("""
            CREATE TABLE energy_weather AS
            SELECT e.*, w.* EXCLUDE (dt_iso)
            FROM raw_energy e
            LEFT JOIN weather_combined w ON e.time = w.dt_iso
        """)

    def _remove_null_rows(self) -> None:
        """Apply the null handling strategy documented in EDA.

        Three-step strategy:
            1. Drop columns that are 100% null (no information).
            2. Drop rows where the target is null (cannot train/predict).
            3. Drop rows with any remaining nulls (0.03% of data).

        This strategy was validated in the notebook: only 47 of 35,064
        rows are affected. The data loss is negligible and the result
        is a guaranteed null-free dataset for downstream consumers.
        """
        logger.info("Removing null rows")

        for col in _DROP_COLUMNS:
            self.con.execute(f"""
                ALTER TABLE energy_weather DROP COLUMN IF EXISTS "{col}"
            """)

        self.con.execute(f"""
            CREATE TABLE training_data AS
            SELECT * FROM energy_weather
            WHERE "{TARGET}" IS NOT NULL
        """)

        before_count = self.con.execute(
            "SELECT COUNT(*) FROM training_data"
        ).fetchone()[0]

        columns = [
            row[0]
            for row in self.con.execute("DESCRIBE training_data").fetchall()
        ]
        not_null_conditions = " AND ".join(
            f'"{col}" IS NOT NULL' for col in columns
        )

        self.con.execute(f"""
            CREATE OR REPLACE TABLE training_data AS
            SELECT * FROM training_data
            WHERE {not_null_conditions}
        """)

        after_count = self.con.execute(
            "SELECT COUNT(*) FROM training_data"
        ).fetchone()[0]
        logger.info(
            "Null removal: %d -> %d rows (dropped %d)",
            before_count,
            after_count,
            before_count - after_count,
        )

    def _validate_output(self) -> None:
        """Verify the output table matches expected schema and quality.

        Catches data quality issues before the serving layer encounters
        them. This is the Shifting Left principle from SE at Google
        Ch. 1: validate early, fail loudly, never pass bad data downstream.

        Also addresses Google Rules of ML #10 (watch for silent failures):
        if upstream data changes cause unexpected row counts or null
        targets, this validation will catch it.

        Raises:
            AssertionError: If any validation check fails.
        """
        row_count = self.con.execute(
            "SELECT COUNT(*) FROM training_data"
        ).fetchone()[0]

        target_nulls = self.con.execute(
            f'SELECT COUNT(*) FROM training_data WHERE "{TARGET}" IS NULL'
        ).fetchone()[0]

        column_count = len(
            self.con.execute("DESCRIBE training_data").fetchall()
        )

        # NOTE: One note to the following assert-statements:
        # Those should be used with caution: It is sufficient to use them here to validate internal invariants,
        # but should never be used to validate external / user facing data or in security critical environments
        # as the python -O flag deactivates them.
        
        assert row_count > self.min_expected_rows, (
            f"Row count validation failed: expected > {self.min_expected_rows}, "
            f"got {row_count}. Possible upstream data issue."
        )
        assert target_nulls == 0, (
            f"Target null validation failed: {target_nulls} null values "
            f"in '{TARGET}' after cleanup."
        )
        assert column_count > self.min_expected_columns, (
            f"Column count validation failed: expected > {self.min_expected_columns}, "
            f"got {column_count}. Possible schema change in raw data."
        )

        logger.info(
            "Output validation passed: %d rows, %d columns, 0 target nulls",
            row_count,
            column_count,
        )

    def _export_to_parquet(self, output_path: Path) -> None:
        """Write the clean training dataset to a Parquet file.

        Parquet is chosen over CSV for three reasons: smaller file size,
        preserved column types (no timestamp parsing issues on reload),
        and faster read performance in the serving layer.

        Args:
            output_path: Destination path for the Parquet file.
        """
        logger.info("Exporting to %s", output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.con.execute(f"""
            COPY training_data TO '{output_path}' (FORMAT PARQUET)
        """)

    def close(self) -> None:
        """Close the DuckDB connection and release resources."""
        self.con.close()
        logger.info("DuckDB connection closed")
