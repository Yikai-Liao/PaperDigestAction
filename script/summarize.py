import sys
import argparse
from pathlib import Path
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.dataloader import load_dataset
from src.trainer import train_model
from src.sampler import predict as predict_fn # Renamed to avoid conflict
from src.summarize import summarize, merge_keywords

from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Manually summarize arXiv papers.")
    parser.add_argument("--arxiv_ids", type=str, required=True,
                        help="Comma-separated list of arXiv IDs to summarize.")
    args = parser.parse_args()

    arxiv_ids_list = [id.strip() for id in args.arxiv_ids.split(',') if id.strip()]
    if not arxiv_ids_list:
        logger.error("No arXiv IDs provided.")
        return

    logger.info(f"Starting manual summarization for arXiv IDs: {arxiv_ids_list}")

    config = Config.default() # Assuming Config.default() loads configuration

    min_year_from_ids = None
    if arxiv_ids_list:
        parsed_years = []
        for arxiv_id in arxiv_ids_list:
            try:
                year_part = ""
                if '.' in arxiv_id: # Format like YYMM.XXXXX or YYYY.XXXXX
                    year_part = arxiv_id.split('.')[0]
                elif '/' in arxiv_id: # Format like archive/YYMMDDD
                    year_part = arxiv_id.split('/')[1][:2] # Takes YY from YYMMDDD

                if year_part.isdigit():
                    if len(year_part) == 2: # YY
                        year_val = int(year_part)
                        # Heuristic: if YY is < 70, assume 20YY, else 19YY (though arXiv started in '91)
                        # More accurately, arXiv IDs like 0706.0001 are 2007.
                        # IDs like hep-th/9108001 are 1991.
                        # For simplicity, assuming YY < 90 is 20YY, otherwise 19YY for 2-digit years.
                        # This might need refinement based on actual ID formats encountered.
                        year = 2000 + year_val if year_val < 90 else 1900 + year_val
                    elif len(year_part) == 4: # YYYY
                        year = int(year_part)
                    else:
                        continue # Not a recognized year format part
                    parsed_years.append(year)
            except Exception as e:
                logger.warning(f"Could not parse year from arXiv ID: {arxiv_id}. Error: {e}. Using broader dataset range.")
        if parsed_years:
            min_year_from_ids = min(parsed_years)
            logger.info(f"Minimum year parsed from provided arXiv IDs: {min_year_from_ids}")

    logger.info("Loading dataset...")
    prefered_df_lazy, full_remaining_df_lazy = load_dataset(config, manual_start_year=min_year_from_ids)
    
    prefered_df = prefered_df_lazy.collect()
    full_remaining_df_collected = full_remaining_df_lazy.collect()
    logger.info(f"Full dataset loaded. Preferred: {prefered_df.height}, Full Remaining (background for training): {full_remaining_df_collected.height}")

    target_papers_df = full_remaining_df_collected.filter(pl.col("id").is_in(arxiv_ids_list))

    if target_papers_df.is_empty():
        logger.warning(f"No papers found in the loaded dataset for the provided arXiv IDs: {arxiv_ids_list}. "
                       "This could be due to incorrect IDs, or the papers not being in the configured categories/date range. "
                       "An empty artifact will be created.")
        empty_df = pl.DataFrame()
        empty_df.write_parquet(REPO_ROOT / "manual_summarized.parquet")
        logger.info("Empty manual_summarized.parquet created.")
        return
        
    logger.info(f"Target papers for summarization: {target_papers_df.height} rows.")

    logger.info("Training model using preferred data and full remaining data as background...")
    final_model = train_model(prefered_df, full_remaining_df_collected, config)
    logger.info("Model training completed.")

    logger.info("Predicting scores for specified papers (forcing inclusion)...")
    recommended_df = predict_fn(final_model, target_papers_df, config, force_include_all=True)
    
    if recommended_df.is_empty() or 'show' not in recommended_df.columns:
        logger.error("Prediction step did not return a valid DataFrame with 'show' column. Cannot proceed.")
        empty_df = pl.DataFrame()
        empty_df.write_parquet(REPO_ROOT / "manual_summarized.parquet")
        logger.info("Empty manual_summarized.parquet created due to prediction error.")
        return

    papers_to_summarize = recommended_df.filter(pl.col('show') == 1)
    logger.info(f"Prediction completed. Papers to summarize: {papers_to_summarize.height}")

    if papers_to_summarize.is_empty():
        logger.warning("No papers marked for summarization after prediction. This shouldn't happen with force_include_all=True if target_papers_df was not empty.")
        empty_df = pl.DataFrame()
        empty_df.write_parquet(REPO_ROOT / "manual_summarized.parquet")
        logger.info("Empty manual_summarized.parquet created as no papers were marked for summarization.")
        return

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
