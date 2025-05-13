import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.dataloader import load_dataset
from src.trainer import train_model
from src.sampler import predict
from src.summarize import summarize, merge_keywords

from loguru import logger

if __name__ == "__main__":
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    logger.info("Started to collect the dataset")
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()
    logger.info("Dataset collection completed")
    
    final_model = train_model(prefered_df, remaining_df, config)
    recommended_df = predict(final_model, remaining_df, config)
    summarized_df = summarize(recommended_df, config)
    summarized_df = merge_keywords(summarized_df, config)
    summarized_df.write_parquet(REPO_ROOT / "summarized.parquet")
    logger.info("Final model training successfully completed")