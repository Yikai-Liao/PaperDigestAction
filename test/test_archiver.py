import sys
import pytest
from pathlib import Path
import polars as pl
from loguru import logger
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.archiver import Archiver, archive_summaries

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

def test_archive_summaries_function(tmp_path):
    """测试archive_summaries函数"""
    output_dir = tmp_path / "summarized"

    data = {"id": ["2505.12345", "2505.67890", "2401.54321"]}
    df = pl.DataFrame(data)

    archive_summaries(df, output_dir=output_dir)
        
    assert output_dir.exists()
    assert (output_dir / "2505.jsonl").exists()
    assert (output_dir / "2401.jsonl").exists()
    
    with open(output_dir / "2505.jsonl") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert "2505.12345" in lines[0]
        assert "2505.67890" in lines[1]
    
    with open(output_dir / "2401.jsonl") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert "2401.54321" in lines[0]

def test_archive_filter_empty_content(setup_archiver):
    """测试归档时过滤掉只有id而没有其他有效内容的记录"""
    archiver, output_dir = setup_archiver
    data = {
        "id": ["2505.00001", "2505.00002", "2505.00003", "2505.00004"],
        "title": ["Title 1", None, "", "   "],
        "summary": ["Summary 1", None, " ", ""]
    }
    df = pl.DataFrame(data)
    
    archiver.archive(df, {})
    
    output_file = output_dir / "2505.jsonl"
    assert output_file.exists()
    
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 1  # 只有第一条记录应该被保存
        record = json.loads(lines[0])
        assert record["id"] == "2505.00001"
        assert record["title"] == "Title 1"
        assert record["summary"] == "Summary 1"

def test_archive_utf8_characters(setup_archiver):
    """测试归档包含UTF-8字符的记录"""
    archiver, output_dir = setup_archiver
    data = {
        "id": ["2505.00005"],
        "title": ["中文标题"],
        "summary": ["这是一段包含中文的摘要内容。"]
    }
    df = pl.DataFrame(data)
    
    archiver.archive(df, {})
    
    output_file = output_dir / "2505.jsonl"
    assert output_file.exists()
    
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "2505.00005"
        assert record["title"] == "中文标题"
        assert record["summary"] == "这是一段包含中文的摘要内容。"

def test_lock_file_cleanup(setup_archiver):
    """测试.lock文件在归档完成后是否被清理"""
    archiver, output_dir = setup_archiver
    data = {"id": ["2505.99999"], "title": ["Lock Test"], "summary": ["This is a test for lock file cleanup."]}
    df = pl.DataFrame(data)
    
    month_str = "2505"
    lock_file = output_dir / f"{month_str}.jsonl.lock"
    
    # 确保在归档前锁文件不存在
    assert not lock_file.exists()
    
    archiver.archive(df, {})
    
    # 归档完成后，锁文件应该被清理
    assert not lock_file.exists()
    
    # 验证数据是否正确写入
    output_file = output_dir / f"{month_str}.jsonl"
    assert output_file.exists()
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "2505.99999"