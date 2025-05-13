import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from huggingface_hub import hf_hub_download
from src.config import Config
from loguru import logger
import polars as pl

def download_dataset(config: Config):
    data_config = config.recommend_pipeline.data
    repo_id = data_config.embed_repo_id
    cache_dir = data_config.cache_dir
    start_year = max(2017, min(data_config.background_start_year, data_config.preference_start_year))
    cur_year = time.localtime().tm_year
    logger.info("Downloading dataset from Hugging Face Hub")
    logger.info(f"Data Config: {data_config}")
    cache_paths = []
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

def load_dataset(config: Config) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    data_config = config.recommend_pipeline.data
    preferences = load_preference(config)
    parquet_paths = download_dataset(config)

    # lazy load the dataset from huggingface
    logger.info(f"Loading dataset from {parquet_paths}")
    df = pl.scan_parquet(parquet_paths, allow_missing_columns=True)
    

    categories  = config.recommend_pipeline.data.categories
    # construct filter condition
    filter_condition = pl.col("categories").list.contains(pl.lit(categories[0]))
    for category in categories[1:]:
        filter_condition = filter_condition | pl.col("categories").list.contains(pl.lit(category))
    logger.debug(f"Filter condition: {filter_condition}")

    # filter the dataset
    lazy_df = df.filter(filter_condition)

    embedding_columns = data_config.embedding_columns
    lazy_df = lazy_df.filter(
        pl.all_horizontal([pl.col(col).is_not_null() for col in embedding_columns])
    )
    logger.info(f"Removing rows with null values in columns: {embedding_columns}")
    
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

    start_year = max(2017, min(data_config.background_start_year, data_config.preference_start_year))
    background_start_year = data_config.background_start_year
    if start_year < background_start_year:
        # year is stored in updated column, like 2023-10-01 in string format
        logger.info(f"Filtering out rows in remaining_df whose year is less than {background_start_year}...")
        remaining_df = remaining_df\
            .with_columns(pl.col("updated").str.slice(0, 4).cast(pl.Int32).alias("year"))\
            .filter(pl.col("year") >= background_start_year)
    remaining_df = remove_recommended(remaining_df, load_recommended(config))
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