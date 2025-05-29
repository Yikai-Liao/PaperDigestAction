import json
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
from loguru import logger
from filelock import FileLock  # 用于文件锁

REPO_ROOT = Path(__file__).resolve().parent.parent

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
                logger.warning("Received empty DataFrame, skipping archiving.")
                return
            
            if "id" not in df.columns:
                raise ValueError("DataFrame missing 'id' column.")
            
            original_height = df.height
            
            # Filter out records with only 'id' and no other meaningful content
            # A record is considered to have meaningful content if any non-'id' column is not null
            # and for Utf8 columns, not an empty string after stripping whitespace.
            
            # Only apply filtering if there are other columns besides 'id'
            other_columns = [col for col in df.columns if col != "id"]
            if other_columns:
                has_meaningful_content_expr = pl.lit(False)
                for col in other_columns:
                    if df[col].dtype == pl.Utf8:
                        has_meaningful_content_expr = has_meaningful_content_expr | (pl.col(col).is_not_null() & (pl.col(col).str.strip_chars() != ""))
                    else:
                        has_meaningful_content_expr = has_meaningful_content_expr | pl.col(col).is_not_null()
                
                df = df.filter(has_meaningful_content_expr)
                
                filtered_height = df.height
                if filtered_height < original_height:
                    logger.info(f"Filtered out {original_height - filtered_height} records with only 'id' and no other meaningful content.")
                
                if df.is_empty():
                    logger.warning("After filtering, DataFrame is empty, skipping archiving.")
                    return
                
            logger.info(f"Archiving {df.height} records.")
            
            # 添加月份列 (arxiv ID前四位)
            df = df.with_columns(
                month=pl.col("id").str.slice(0, 4)
            )
            
            # 按月份分组处理
            # Extract month string from group key tuple
            for group_key, group in df.group_by("month", maintain_order=True):
                month_str = group_key[0]  # Group key is a tuple (month_value,)
                month_file = self.output_dir / f"{month_str}.jsonl"
                lock_file = Path(f"{month_file}.lock") # Define lock_file path
                logger.info(f"Processing {len(group)} records for month {month_str}.")
                
                # 使用文件锁确保线程安全
                with FileLock(lock_file):
                    self._append_to_jsonl(month_file, group.drop("month"))
                
                # 显式删除锁文件，确保清理
                if lock_file.exists():
                    lock_file.unlink()
                    logger.info(f"Cleaned up lock file: {lock_file}")
                    
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
        # Convert new DataFrame to a list of dictionaries
        new_records = df.to_dicts()

        # Read existing data and build a dictionary for efficient lookup and update
        existing_records_map: Dict[str, Dict[str, Any]] = {}
        if file_path.exists():
            try:
                with file_path.open("r", encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        if "id" in record:
                            existing_records_map[record["id"]] = record
                logger.info(f"Read {len(existing_records_map)} existing records from {file_path}.")
            except Exception as e:
                logger.warning(f"Could not read existing JSONL file {file_path}: {e}. Treating as empty.")
                existing_records_map = {} # Reset to empty if reading fails

        # Process new records: add them or overwrite existing ones by 'id'.
        # The dictionary assignment `existing_records_map[record["id"]] = record`
        # naturally handles both adding new unique IDs and overwriting records
        # with matching IDs from the new data. This is the desired "覆盖" (overwrite) behavior.
        for record in new_records:
            if "id" in record:
                logger.warning(f"ID {record['id']} already exists, overwriting with new data.")
            existing_records_map[record["id"]] = record
        
        # Convert map values back to a list and sort by 'id' to ensure overall order.
        final_records = sorted(existing_records_map.values(), key=lambda x: x.get("id"))
        
        # Write the entire deduplicated and sorted list back to the file, overwriting its previous content.
        try:
            with file_path.open("w", encoding='utf-8') as f: # Use "w" to overwrite the file
                for record in final_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.success(f"Archived {len(final_records)} unique records to {file_path}.")
        except IOError as e:
            logger.error(f"Failed to write to {file_path}: {e}")
            raise

def archive_summaries(summarized_df: pl.DataFrame, output_dir: Path | None = None) -> None:
    """
    将摘要数据归档到 'summarized/' 文件夹中的JSONL文件。
    
    Args:
        summarized_df: 包含摘要数据的DataFrame。
        output_dir: 可选的输出目录路径。如果未提供，则默认为 REPO_ROOT / "summarized"。
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "summarized"
    archiver = Archiver(output_dir)
    archiver.archive(summarized_df, {}) # Pass an empty config dict as it's not used by archive method currently
    logger.info(f"Summaries archived to {output_dir}")