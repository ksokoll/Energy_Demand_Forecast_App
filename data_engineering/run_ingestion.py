"""Run the DuckDB ingestion pipeline.

Entrypoint script for the Data Engineering part of the system.
Reads raw CSVs from data/raw/, runs the full cleaning and merging
pipeline, and writes a clean Parquet file to data/.

This script is intended to run once (initial setup) or periodically
(when new data arrives). In production, this would be triggered by
a scheduler (e.g. AWS Step Functions, Airflow). For the portfolio
project, it runs manually from the command line.

Usage:
    python scripts/run_ingestion.py
"""

import logging
import sys

from data_engineering.ingestion import IngestionPipeline
from energy_forecast.config import DATA_PATH, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

ENERGY_FILENAME = "energy_dataset.csv"
WEATHER_FILENAME = "weather_features.csv"


def main() -> None:
    """Execute the ingestion pipeline with error handling.

    Validates that input files exist before starting the pipeline.
    Logs the outcome and exits with a non-zero code on failure,
    which is required for CI and scheduler integration (a silent
    failure in a data pipeline is worse than a loud crash).
    """
    energy_path = RAW_DIR / ENERGY_FILENAME
    weather_path = RAW_DIR / WEATHER_FILENAME

    logger.info("Input:  %s, %s", energy_path, weather_path)
    logger.info("Output: %s", DATA_PATH)

    pipeline = IngestionPipeline()

    try:
        pipeline.run(
            energy_path=energy_path,
            weather_path=weather_path,
            output_path=DATA_PATH,
        )
    except FileNotFoundError as e:
        logger.error("Input file missing: %s", e)
        sys.exit(1)
    except AssertionError as e:
        logger.error("Output validation failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed unexpectedly: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
