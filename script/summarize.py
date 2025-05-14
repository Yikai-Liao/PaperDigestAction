import sys
import argparse
from pathlib import Path
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.dataloader import load_dataset, download_dataset
from src.trainer import train_model
from src.sampler import predict
from src.summarize import summarize, merge_keywords

from loguru import logger

def model_training_pipeline():
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    logger.info("Started to collect the dataset")
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()
    return train_model(prefered_df, remaining_df, config)

def main():
    parser = argparse.ArgumentParser(description="Manually summarize arXiv papers.")
    parser.add_argument("arxiv_ids", type=str,
                        help="Comma-separated list of arXiv IDs to summarize.")
    args = parser.parse_args()
    arxiv_ids_list = [id.strip() for id in args.arxiv_ids.split(',') if id.strip()]
    if not arxiv_ids_list:
        logger.error("No arXiv IDs provided.")
        return

    logger.info(f"Starting manual summarization for arXiv IDs: {arxiv_ids_list}")

    config = Config.default() # Assuming Config.default() loads configuration

    start_year = int(f"20{min(int(id[:2]) for id in arxiv_ids_list)}")
    logger.debug(f"Setting start year for dataset loading: {start_year}")
    logger.info("Loading dataset...")
    parquet_paths = download_dataset(config, start_year_override=start_year)
    df = pl.scan_parquet(parquet_paths, allow_missing_columns=True)
    target_df = df.filter(pl.col("id").is_in(pl.Series(arxiv_ids_list).implode())).collect()
    logger.info(f"Loaded dataset with {target_df.height} rows for the specified arXiv IDs.")

    model = model_training_pipeline()
    logger.info("Model training completed.")

    recommended_df = predict(model, target_df, config, force_include_all=True)
    
    if recommended_df.is_empty() or 'show' not in recommended_df.columns:
        logger.error("Prediction step did not return a valid DataFrame with 'show' column. Cannot proceed.")
        empty_df = pl.DataFrame()
        empty_df.write_parquet(REPO_ROOT / "manual_summarized.parquet")
        logger.info("Empty manual_summarized.parquet created due to prediction error.")
        return

    papers_to_summarize = recommended_df.filter(pl.col('show') == 1)
    logger.info(f"Prediction completed. Papers to summarize: {papers_to_summarize.height}")

    logger.info("Summarizing papers...")
    summarized_df = summarize(papers_to_summarize, config)
    logger.info("Summarization completed.")

    logger.info("Merging keywords...")
    summarized_df = merge_keywords(summarized_df, config)
    logger.info("Keyword merging completed.")

    output_path = REPO_ROOT / "summarized.parquet"  # Changed output filename
    summarized_df.write_parquet(output_path)
    logger.info(f"Manual summarization completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
