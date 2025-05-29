import json
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
from loguru import logger
from filelock import FileLock  # 用于文件锁

class Archiver:
    """归档处理器，按arxiv ID分组保存JSONL文件"""
    
    def __init__(self, output_dir: Path) -> None:
        """
        初始化归档器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Archiver initialized with output directory: {output_dir}")

    def archive(self, df: pl.DataFrame, config: Dict[str, Any]) -> None:
        """
        归档DataFrame数据到JSONL文件
        
        Args:
            df: 输入DataFrame
            config: 配置字典
            
        Raises:
            ValueError: 如果输入数据无效
            IOError: 如果文件写入失败
        """
        try:
            # 验证输入
            if df.is_empty():
                logger.warning("Received empty DataFrame, skipping archiving")
                return
            
            if "id" not in df.columns:
                raise ValueError("DataFrame missing 'id' column")
                
            logger.info(f"Archiving {df.height} records")
            
            # 添加月份列 (arxiv ID前四位)
            df = df.with_columns(
                month=pl.col("id").str.slice(0, 4)
            )
            
            # 按月份分组处理
            # Extract month string from group key tuple
            for group_key, group in df.group_by("month", maintain_order=True):
                month_str = group_key[0]  # Group key is a tuple (month_value,)
                month_file = self.output_dir / f"{month_str}.jsonl"
                logger.info(f"Processing {len(group)} records for month {month_str}")
                
                # 使用文件锁确保线程安全
                with FileLock(f"{month_file}.lock"):
                    self._append_to_jsonl(month_file, group.drop("month"))
                    
        except Exception as e:
            logger.error(f"Archiving failed: {e}")
            raise

    def _append_to_jsonl(self, file_path: Path, df: pl.DataFrame) -> None:
        """
        追加数据到JSONL文件（线程安全）
        
        Args:
            file_path: JSONL文件路径
            df: 要追加的DataFrame
        """
        # 转换为排序后的字典列表
        sorted_data = df.sort("id").to_dicts()
        
        # 追加写入文件
        try:
            with file_path.open("a") as f:
                for record in sorted_data:
                    f.write(json.dumps(record) + "\n")
            logger.success(f"Appended {len(sorted_data)} records to {file_path}")
        except IOError as e:
            logger.error(f"Failed to write to {file_path}: {e}")
            raise