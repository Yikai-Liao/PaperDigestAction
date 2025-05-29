import os
from pathlib import Path
import polars as pl
from loguru import logger
from pathlib import Path
import polars as pl
import sys

# Add project root to sys.path for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.archiver import archive_summaries

def initialize_summarized_data() -> None:
    """
    Initializes summarized data by downloading from Hugging Face and archiving it.
    """
    logger.info("Starting initialization of summarized data...")

    # Load configuration
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    try:
        config = Config.from_toml(str(config_path))
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)

    repo_id = config.recommend_pipeline.data.content_repo_id
    output_dir = Path(__file__).resolve().parent.parent / "summarized"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = f"hf://datasets/{repo_id}/main.parquet"
    logger.info(f"Reading parquet file directly from Hugging Face dataset: {dataset_path}")
    try:
        # Read the parquet file directly using Polars with hf:// protocol
        df = pl.read_parquet(dataset_path)
        logger.success(f"Successfully read {df.height} records from {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to read parquet file from {dataset_path}: {e}")
        logger.info("Please ensure you have authenticated with Hugging Face if it's a private dataset, e.g., by running `huggingface-cli login`.")
        sys.exit(1)

    logger.info(f"Archiving summarized data to {output_dir}")
    try:
        # Archive the data
        archive_summaries(df, output_dir)
        logger.success("Summarized data archived successfully.")
    except Exception as e:
        logger.error(f"Failed to archive summarized data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_summarized_data()