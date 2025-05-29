import sys
import pytest
from pathlib import Path
import polars as pl
from loguru import logger
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.archiver import Archiver

@pytest.fixture
def setup_archiver(tmp_path):
    """设置Archiver测试环境"""
    output_dir = tmp_path / "summarized"
    return Archiver(output_dir), output_dir
def test_archive_basic(setup_archiver):
    """测试基本归档功能"""
    archiver, output_dir = setup_archiver
    data = {"id": ["2505.12345", "2505.67890"]}
    df = pl.DataFrame(data)
    
    archiver.archive(df, {})
    
    output_file = output_dir / "2505.jsonl"
    assert output_file.exists()
    
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert "2505.12345" in lines[0]
        assert "2505.67890" in lines[1]

def test_archive_empty_df(setup_archiver):
    """测试空DataFrame处理"""
    archiver, _ = setup_archiver
    df = pl.DataFrame()

    # 只需验证函数正常执行而不抛出异常
    archiver.archive(df, {})

def test_archive_missing_column(setup_archiver):
    """测试缺少id列的情况"""
    archiver, _ = setup_archiver
    df = pl.DataFrame({"title": ["Test Paper"]})

    # 只需验证抛出正确异常
    with pytest.raises(ValueError, match="DataFrame missing 'id' column"):
        archiver.archive(df, {})

def test_archive_multiple_months(setup_archiver):
    """测试多个月份归档"""
    archiver, output_dir = setup_archiver
    data = {
        "id": [
            "2505.12345",
            "2505.67890",
            "2401.54321",
            "2401.98765"
        ]
    }
    df = pl.DataFrame(data)
    
    archiver.archive(df, {})
    
    assert (output_dir / "2505.jsonl").exists()
    assert (output_dir / "2401.jsonl").exists()
    
    with open(output_dir / "2505.jsonl") as f:
        assert len(f.readlines()) == 2
    
    with open(output_dir / "2401.jsonl") as f:
        assert len(f.readlines()) == 2