import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.augment import sample_pesudo_negtive
from src.dataloader import load_dataset
from src.trainer import train_model
from src.sampler import predict_and_save

from loguru import logger

if __name__ == "__main__":
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()
    final_model = train_model(prefered_df, remaining_df, config)
    predict_and_save(final_model, remaining_df, config)
    logger.info("Final model training successfully completed")