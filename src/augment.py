import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from loguru import logger
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression


def sample_pesudo_negtive(prefered_df: pl.DataFrame, backgroud_df: pl.DataFrame, config: Config) -> tuple:
    """
    Sample pseudo negative examples from background data for training
    
    Args:
        prefered_df: DataFrame containing user preference data
        backgroud_df: DataFrame containing background data for sampling negatives
        config: Configuration object with training parameters
        
    Returns:
        tuple of feature matrix x and target vector y
    """
    logger.info("Sampling pesudo negative data")
    data_config = config.recommend_pipeline.data
    embedding_columns = data_config.embedding_columns
    bg_sample_rate = config.recommend_pipeline.trainer.bg_sample_rate
    prefered_df = prefered_df.with_columns(
        pl.when(pl.col("preference") == "like").then(1).otherwise(0).alias("label")
    ).select("label", *embedding_columns)

    backgroud_df = backgroud_df.select(*embedding_columns)

    # Calculate positive sample number (like)
    positive_sample_num = prefered_df.filter(pl.col("label") == 1).height
    logger.debug(f"Positive sample num: {positive_sample_num}")
    
    neg_sample_ratio = bg_sample_rate
    neg_sample_num = int(neg_sample_ratio * positive_sample_num)
    logger.debug(f"Negative sample num: {neg_sample_num}")
    
    # Sample negative data
    pesudo_neg_df = backgroud_df.sample(n=neg_sample_num, seed=42)
    pesudo_neg_df = pesudo_neg_df.with_columns(
        pl.lit(0).alias("label")
    ).select("label", *embedding_columns)

    combined_df = pl.concat([prefered_df, pesudo_neg_df], how="vertical")
    x = np.hstack([np.vstack(combined_df[col].to_numpy()) for col in embedding_columns])
    y = combined_df.select("label").to_numpy().ravel()
    return x, y