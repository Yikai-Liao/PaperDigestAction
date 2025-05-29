import sys
import pytest
import polars as pl
from pathlib import Path
from loguru import logger

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.summarize import load_cached_summaries # Added load_cached_summaries
from src.config import Config # Removed unused specific config imports

def test_hf_correct_cache():
    """
    Test the load_cached_summaries function by interacting with the real Hugging Face dataset.
    This test assumes that specific data exists in the hf://datasets/lyk/ArxivSummarize/ repository.
    If the assumed data is not present, this test may fail or not validate all scenarios.
    """
    # 1. Load real configuration
    # Assuming config.toml is in the root of the project, relative to this test file's parent's parent
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    if not config_path.exists():
        pytest.fail(f"Config file not found at {config_path}. This test requires the main config.toml.")
    
    try:
        config = Config.from_toml(str(config_path))
    except Exception as e:
        pytest.fail(f"Failed to load config from {config_path}: {e}")

    id_fully_cached_1 = "2310.03903"
    id_fully_cached_2 = "2410.12735"

    # --- Test Scenario 1: Fully Cached Hits ---
    logger.info("--- Test Scenario 1: Fully Cached Hits ---")
    paper_ids_fully_cached = [id_fully_cached_1, id_fully_cached_2]
    result_df_fully_cached = load_cached_summaries(paper_ids_fully_cached, config)
    assert not result_df_fully_cached.is_empty(), "Expected non-empty DataFrame for fully cached hits."
    assert set(result_df_fully_cached["id"].to_list()) == set(paper_ids_fully_cached), \
        "Expected DataFrame to contain only the requested IDs for fully cached hits."
    
def test_hf_invalid_cache():
    config = Config.from_toml("config.toml")
    ids = ["9999.00000", "8888.00000"]
    result_df_invalid = load_cached_summaries(ids, config)
    assert result_df_invalid.is_empty(), "Expected empty DataFrame for invalid cache hits."