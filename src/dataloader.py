import sys
import time
from pathlib import Path
from typing import Optional, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from loguru import logger
import polars as pl

def download_dataset(config: Config, start_year_override: Optional[int] = None): # Added start_year_override
    data_config = config.recommend_pipeline.data
    repo_id = data_config.embed_repo_id
    cache_dir = data_config.cache_dir
    
    # Use override if provided, else use config, ensuring it's within reasonable bounds
    base_start_year = data_config.background_start_year
    if start_year_override is not None:
        base_start_year = start_year_override
        logger.info(f"Overriding dataset download start year to: {base_start_year}")

    # Ensure preference_start_year is also considered for the min year
    start_year = max(2017, min(base_start_year, data_config.preference_start_year, data_config.background_start_year))
    
    cur_year = time.localtime().tm_year
    logger.info(f"Effective start year for download: {start_year}, Current year: {cur_year}")
    logger.info(f"Data Config (for download): {data_config}") # Log the config being used
    cache_paths = []
    if start_year > cur_year:
        logger.warning(f"Start year {start_year} is after current year {cur_year}. No data will be downloaded.")
        return [] # Return empty list if start_year is in the future

    for year in range(start_year, cur_year + 1):
        filename = f"{year}.parquet"
        cache_paths.append(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=False,
                repo_type="dataset",
            )
        )
    return cache_paths

def load_local_summarized() -> pl.DataFrame:
    """
    Loads all summarized data from the 'summarized/' folder.
    Uses polars.scan_ndjson to efficiently read all JSONL files.
    Returns an empty DataFrame if no data is found.
    """
    summarized_dir = REPO_ROOT / "summarized"
    if not summarized_dir.exists():
        logger.info(f"Summarized directory not found: {summarized_dir}. Returning empty DataFrame.")
        return pl.DataFrame(schema={"id": pl.String, "summary": pl.String}) # Define schema for empty DataFrame

    try:
        # Use scan_ndjson to efficiently read all JSONL files
        # The glob pattern "summarized/*.jsonl" is relative to the current working directory
        # For REPO_ROOT, it should be REPO_ROOT / "summarized" / "*.jsonl"
        # However, polars.scan_ndjson expects a string path or list of paths.
        # Let's use the absolute path for robustness.
        jsonl_files = list(summarized_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.info(f"No JSONL files found in {summarized_dir}. Returning empty DataFrame.")
            return pl.DataFrame(schema={"id": pl.String, "summary": pl.String})

        # Convert Path objects to strings for scan_ndjson
        file_paths_str = [str(p) for p in jsonl_files]
        
        # Scan all JSONL files and collect into a single DataFrame
        df = pl.scan_ndjson(file_paths_str).collect()
        logger.info(f"Loaded {len(df)} summarized items from {summarized_dir}")
        return df
    except Exception as e:
        logger.error(f"Error loading local summarized data: {e}. Returning empty DataFrame.")
        return pl.DataFrame(schema={"id": pl.String, "summary": pl.String})


def load_preference(config: Config) -> pl.DataFrame:
    preference_dir = REPO_ROOT / config.recommend_pipeline.data.preference_dir
    if not preference_dir.exists():
        logger.error(f"Preference directory not found: {preference_dir}")
        sys.exit(1)

    preferences = [ pl.read_csv(p, columns=['id', 'preference'], schema={'id': pl.String, 'preference': pl.String}) for p in preference_dir.glob("**/*.csv") ]
    if not preferences:
        logger.error(f"No preference files found in {preference_dir}")
        sys.exit(1)
    # Combine all DataFrames into one
    preferences = pl.concat(preferences, how="vertical").unique(subset="id")
    logger.info(f"{len(preferences)} preference items loaded from {preference_dir}")
    return preferences


def load_dataset(config: Config) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    data_config = config.recommend_pipeline.data
    preferences = load_preference(config)
    
    parquet_paths = download_dataset(config) # Use temp_config with adjusted year

    df = pl.scan_parquet(parquet_paths, allow_missing_columns=True)
    
    categories = config.recommend_pipeline.data.categories
    filter_condition = pl.col("categories").list.contains(pl.lit(categories[0]))
    for category in categories[1:]:
        filter_condition = filter_condition | pl.col("categories").list.contains(pl.lit(category))
    
    lazy_df = df.filter(filter_condition)

    embedding_columns = data_config.embedding_columns
    lazy_df = lazy_df.filter(
        pl.all_horizontal([pl.col(col).is_not_null() for col in embedding_columns])
    )
    
    preference_ids = preferences.select("id").to_series()
    indicator_col_name = "__is_preferred__" # Using a distinct name
    # 使用implode()修复is_in弃用警告
    database = lazy_df.with_columns(
        pl.col("id").is_in(preference_ids.implode()).alias(indicator_col_name)
    )

    logger.info(f"Splitting collected data based on '{indicator_col_name}' column...")

    prefered_df = database.filter(pl.col(indicator_col_name))
    remaining_df = database.filter(~pl.col(indicator_col_name))

    preferences = preferences.lazy()

    prefered_df = prefered_df.join(
        preferences, on="id", how="inner"
    ).drop(indicator_col_name)
        
    remaining_df = remaining_df.drop(indicator_col_name)

    # So, load_dataset should return the full remaining_df. The filtering for specific IDs will happen in script/summarize.py.

    start_year = max(2017, min(data_config.background_start_year, data_config.preference_start_year))
    background_start_year = data_config.background_start_year
    if start_year < background_start_year:
        # year is stored in updated column, like 2023-10-01 in string format
        logger.info(f"Filtering out rows in remaining_df whose year is less than {background_start_year}...")
        remaining_df = remaining_df\
            .with_columns(pl.col("updated").str.slice(0, 4).cast(pl.Int32).alias("year"))\
            .filter(pl.col("year") >= background_start_year)
    remaining_df = remove_recommended(remaining_df) # Call the updated remove_recommended
    return prefered_df, remaining_df

def show_df_size(df: pl.DataFrame, name: str):
    # Row count and size in MB
    row_count = df.height
    size_mb = df.estimated_size() / (1024 ** 2)
    logger.debug(f"{name} - Rows: {row_count}, Size: {size_mb:.2f} MB")

def remove_recommended(remaining_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Removes recommended items from remaining_df by loading local summarized data.
    """
    logger.info("Removing recommended items from remaining_df...")
    recommended_df = load_local_summarized()
    if recommended_df.is_empty():
        logger.info("No local summarized data found, no items to remove.")
        return remaining_df
    
    recommended_ids = recommended_df.select("id").to_series()
    # 使用implode()修复is_in弃用警告
    remaining_df = remaining_df.filter(~pl.col("id").is_in(recommended_ids.implode()))
    return remaining_df

if __name__ == "__main__":
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()

    show_df_size(prefered_df, "Prefered DataFrame")
    show_df_size(remaining_df, "Remaining DataFrame")