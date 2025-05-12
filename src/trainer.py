import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.augment import sample_pesudo_negtive
from src.dataloader import load_dataset
from loguru import logger
import polars as pl
import numpy as np
import sklearn

def train_model(prefered_df: pl.DataFrame, remaining_df: pl.DataFrame, config: Config) -> None:
    # Sample pseudo negative examples
    x, y = sample_pesudo_negtive(prefered_df, remaining_df, config)
    logger.info(f"Original data - x: {x.dtype} | {x.shape}, y shape: {y.dtype} | {y.shape}")
    
    trainer_cfg = config.recommend_pipeline.trainer
    lg_cfg = trainer_cfg.logci_regression
    
    # Initialize the base classifier with the same parameters as the main training
    model = sklearn.linear_model.LogisticRegression(
        C=lg_cfg.C,
        max_iter=lg_cfg.max_iter,
        random_state=trainer_cfg.seed,
        class_weight="balanced",
    )
    
    model.fit(x, y)
    logger.info("Model training completed")
    return model

if __name__ == "__main__":
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()
    final_model = train_model(prefered_df, remaining_df, config)
    logger.info("Final model training successfully completed")