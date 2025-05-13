import sys
import time
from pathlib import Path
from typing import Optional, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from huggingface_hub import hf_hub_download
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
    start_year = max(2017, min(base_start_year, data_config.preference_start_year))
    
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

def load_recommended(config: Config) -> pl.DataFrame:
    """
    Raw data dir contains lots of json files like 2407.0288.json
    We need to extract the Arxiv ID from the filename
    and collect all the IDs into a pl.DataFrame
    """
    # raw_dir = REPO_ROOT / config.get("raw_data_dir", "raw")
    # if not raw_dir.exists():
        # return pl.DataFrame(columns=["id"], schema={"id": pl.String})
    content_repo = config.recommend_pipeline.data.content_repo_id
    recommended = pl.scan_parquet(f"hf://datasets/{content_repo}/main.parquet").collect()
    return recommended

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

def remove_recommended(remaining_df: pl.LazyFrame, recommended_df: pl.DataFrame):
    # Remove recommended items from remaining_df
    logger.info("Removing recommended items from remaining_df...")
    recommended_ids = recommended_df.select("id").to_series()
    remaining_df = remaining_df.filter(~pl.col("id").is_in(recommended_ids))
    return remaining_df

def load_dataset(config: Config, manual_start_year: Optional[int] = None, arxiv_ids_list: Optional[List[str]] = None) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    data_config = config.recommend_pipeline.data
    preferences = load_preference(config)

    # Determine the start year for downloading data
    effective_start_year = data_config.background_start_year
    if manual_start_year is not None:
        effective_start_year = min(effective_start_year, manual_start_year)
        logger.info(f"Manual start year provided: {manual_start_year}. Effective start year for data loading: {effective_start_year}")
    
    # Adjust download_dataset to use this effective_start_year
    # This requires download_dataset to be modified to accept a start_year parameter
    # For now, assuming download_dataset is modified or its internal logic respects a more dynamic start year if set in config
    # Or, we adjust the call here if download_dataset is refactored.
    # Let's assume for now that download_dataset needs to be passed the effective_start_year.
    # We'll need to modify download_dataset signature and logic.

    # Placeholder for modified download_dataset call
    # parquet_paths = download_dataset(config, start_year_override=effective_start_year) 
    # Assuming download_dataset is modified to accept start_year_override
    # For now, we'll proceed with the existing download_dataset and filter later if necessary,
    # or rely on the fact that it downloads a range and we filter from that.
    # The ideal way is to make download_dataset more flexible.
    # Let's simulate this by adjusting the config if possible, or filtering after load.
    
    # Create a temporary config for data loading if needed to adjust start year
    temp_data_config = data_config.model_copy()
    temp_data_config.background_start_year = effective_start_year
    temp_data_config.preference_start_year = min(data_config.preference_start_year, effective_start_year) # also adjust preference start year if needed

    temp_config = config.model_copy()
    temp_config.recommend_pipeline.data = temp_data_config
    
    parquet_paths = download_dataset(temp_config) # Use temp_config with adjusted year

    logger.info(f"Loading dataset from {parquet_paths} with effective start year {effective_start_year}")
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
    indicator_col_name = "__is_preferred__"
    
    database = lazy_df.with_columns(
        pl.col("id").is_in(preference_ids.implode()).alias(indicator_col_name)
    )

    prefered_df = database.filter(pl.col(indicator_col_name)).drop(indicator_col_name)
    remaining_df = database.filter(pl.col(indicator_col_name).not_()).drop(indicator_col_name)

    # If arxiv_ids_list is provided (for manual summarize), filter remaining_df to only these IDs.
    # The 'prefered_df' should remain as is for model training.
    # The 'remaining_df' for manual summarization context should be *only* the target IDs.
    # However, for training the model, 'remaining_df' should be the broader background set.
    # So, the function should return:
    # 1. prefered_df (for training)
    # 2. EITHER the full remaining_df (for recommend.py or for training background in summarize.py)
    #    OR a filtered remaining_df (if arxiv_ids_list is given, this becomes the target for summarization *after* training)
    # The script/summarize.py will need to handle this: load full for training, then filter for prediction.
    # So, load_dataset should return the full remaining_df. The filtering for specific IDs will happen in script/summarize.py.

    return prefered_df, remaining_df

def show_df_size(df: pl.DataFrame, name: str):
    # Row count and size in MB
    row_count = df.height
    size_mb = df.estimated_size() / (1024 ** 2)
    logger.debug(f"{name} - Rows: {row_count}, Size: {size_mb:.2f} MB")

def remove_recommended(remaining_df: pl.LazyFrame, recommended_df: pl.DataFrame):
    # Remove recommended items from remaining_df
    logger.info("Removing recommended items from remaining_df...")
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