import pytest
import polars as pl
from pathlib import Path
import json
import sys

# Add the src directory to the Python path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataloader import load_local_summarized, remove_recommended, REPO_ROOT

@pytest.fixture
def setup_summarized_dir(tmp_path):
    """
    Fixture to set up a temporary 'summarized' directory for testing.
    It temporarily changes REPO_ROOT to tmp_path for the duration of the test.
    """
    original_repo_root = REPO_ROOT
    # Temporarily change REPO_ROOT to the tmp_path for testing
    # This is a bit hacky, but necessary for the current REPO_ROOT usage in dataloader.py
    # A better long-term solution would be to pass the base path to dataloader functions.
    dataloader_module = sys.modules['src.dataloader']
    dataloader_module.REPO_ROOT = tmp_path
    
    summarized_path = tmp_path / "summarized"
    summarized_path.mkdir()
    
    yield summarized_path
    
    # Restore original REPO_ROOT after test
    dataloader_module.REPO_ROOT = original_repo_root

@pytest.fixture
def setup_summarized_data(setup_summarized_dir):
    """
    Fixture to create dummy JSONL files in the temporary 'summarized' directory.
    """
    summarized_path = setup_summarized_dir
    
    data1 = [
        {"id": "1", "summary": "summary 1"},
        {"id": "2", "summary": "summary 2"}
    ]
    data2 = [
        {"id": "3", "summary": "summary 3"},
        {"id": "4", "summary": "summary 4"}
    ]

    with open(summarized_path / "file1.jsonl", "w") as f:
        for item in data1:
            f.write(json.dumps(item) + "\n")

    with open(summarized_path / "file2.jsonl", "w") as f:
        for item in data2:
            f.write(json.dumps(item) + "\n")
            
    return summarized_path

def test_load_local_summarized_no_dir(tmp_path):
    """
    Test that load_local_summarized returns an empty DataFrame when 'summarized' directory does not exist.
    """
    original_repo_root = REPO_ROOT
    dataloader_module = sys.modules['src.dataloader']
    dataloader_module.REPO_ROOT = tmp_path # Ensure summarized dir won't be found
    
    df = load_local_summarized()
    assert df.is_empty()
    assert df.columns == ["id", "summary"] # Check schema
    
    dataloader_module.REPO_ROOT = original_repo_root # Restore

def test_load_local_summarized_empty_dir(setup_summarized_dir):
    """
    Test that load_local_summarized returns an empty DataFrame when 'summarized' directory is empty.
    """
    df = load_local_summarized()
    assert df.is_empty()
    assert df.columns == ["id", "summary"] # Check schema

def test_load_local_summarized_with_data(setup_summarized_data):
    """
    Test that load_local_summarized correctly loads and merges data from JSONL files.
    """
    df = load_local_summarized()
    assert not df.is_empty()
    assert df.height == 4
    assert "id" in df.columns
    assert "summary" in df.columns
    assert set(df["id"].to_list()) == {"1", "2", "3", "4"}

def test_remove_recommended_with_data(setup_summarized_data):
    """
    Test remove_recommended correctly filters out recommended items.
    """
    # Create a dummy remaining_df
    remaining_data = [
        {"id": "1", "title": "Paper A"},
        {"id": "2", "title": "Paper B"},
        {"id": "5", "title": "Paper E"},
        {"id": "6", "title": "Paper F"}
    ]
    remaining_df = pl.LazyFrame(remaining_data, schema={"id": pl.String, "title": pl.String})

    # '1' and '2' are in summarized data, so they should be removed
    filtered_df = remove_recommended(remaining_df).collect()
    
    assert filtered_df.height == 2
    assert set(filtered_df["id"].to_list()) == {"5", "6"}

def test_remove_recommended_no_summarized_data(tmp_path):
    """
    Test remove_recommended when no summarized data exists.
    It should return the original remaining_df unchanged.
    """
    original_repo_root = REPO_ROOT
    dataloader_module = sys.modules['src.dataloader']
    dataloader_module.REPO_ROOT = tmp_path # Ensure summarized dir won't be found
    
    remaining_data = [
        {"id": "1", "title": "Paper A"},
        {"id": "2", "title": "Paper B"}
    ]
    remaining_df = pl.LazyFrame(remaining_data, schema={"id": pl.String, "title": pl.String})

    filtered_df = remove_recommended(remaining_df).collect()
    
    assert filtered_df.height == 2
    assert set(filtered_df["id"].to_list()) == {"1", "2"}
    
    dataloader_module.REPO_ROOT = original_repo_root # Restore

def test_remove_recommended_empty_remaining_df(setup_summarized_data):
    """
    Test remove_recommended with an empty remaining_df.
    """
    remaining_df = pl.LazyFrame(schema={"id": pl.String, "title": pl.String})
    filtered_df = remove_recommended(remaining_df).collect()
    assert filtered_df.is_empty()
    assert filtered_df.columns == ["id", "title"]