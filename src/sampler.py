import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.dataloader import load_dataset
from loguru import logger
import polars as pl
import numpy as np
import datetime
from typing import Optional

def predict_and_recommend(model, target_df: pl.DataFrame, config: Config, force_include_all: bool = False) -> pl.DataFrame:
    logger.info("开始预测和推荐阶段...")
    
    if force_include_all:
        logger.info("force_include_all is True. Skipping date filtering and adaptive sampling.")
    else:
        # 2. 根据配置选择目标时间段的数据
        predict_config = config.recommend_pipeline.predict
        
        # 解析日期相关配置
        last_n_days = predict_config.last_n_days
        start_date = predict_config.start_date
        end_date = predict_config.end_date
        
        # 如果指定了起止日期，使用日期范围筛选
        if start_date and end_date:
            logger.info(f"使用指定的日期范围: {start_date} 到 {end_date}")
            target_df = target_df.filter(
                (pl.col("updated") >= start_date) & (pl.col("updated") <= end_date)
            )
        # 否则使用最近N天
        else:
            # 计算最近N天的日期
            today = datetime.datetime.now()
            n_days_ago = (today - datetime.timedelta(days=last_n_days)).strftime("%Y-%m-%d")
            logger.info(f"使用最近{last_n_days}天的数据: {n_days_ago} 到 {today.strftime('%Y-%m-%d')}")
            
            target_df = target_df.filter(pl.col("updated") >= n_days_ago)
        
    # 3. 收集目标数据
    logger.info("收集目标数据...")

    data_config = config.recommend_pipeline.data
    embedding_columns = data_config.embedding_columns
    
    # 如果没有目标数据，提前返回
    if target_df.is_empty():
        logger.warning("没有满足条件的目标数据可供推荐")
        return target_df
    
    logger.info(f"目标数据收集完成，共 {target_df.height} 条记录")
    
    # 4. 提取特征和预测
    logger.info("提取特征并进行预测...")
    # X_target = target_df.select(*embedding_columns).to_numpy()
    X_target = np.hstack([np.vstack(target_df[col].to_numpy()) for col in embedding_columns])
    
    # 使用模型预测"喜欢"的概率
    try:
        target_scores = model.predict_proba(X_target)[:, 1]
        logger.info(f"预测完成，分数范围: {np.min(target_scores):.4f} - {np.max(target_scores):.4f}, 平均: {np.mean(target_scores):.4f}")
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        raise
    
    # 5. 添加预测分数到DataFrame
    target_df = target_df.with_columns(pl.lit(target_scores).alias("score"))
    
    if force_include_all:
        logger.info("force_include_all is True. Marking all papers for inclusion.")
        target_df = target_df.with_columns(pl.lit(1).cast(pl.Int8).alias("show"))
    else:
        # 6. 执行自适应采样决定推荐
        logger.info("执行自适应采样确定推荐...")
        
        high_threshold = predict_config.high_threshold
        boundary_threshold = predict_config.boundary_threshold
        sample_rate = predict_config.sample_rate
        
        show_flags = adaptive_sample(
            target_scores, 
            target_sample_rate=sample_rate,
            high_threshold=high_threshold,
            boundary_threshold=boundary_threshold,
            random_state=config.recommend_pipeline.trainer.seed
        )
        
        # 7. 添加推荐标记到DataFrame
        target_df = target_df.with_columns(pl.lit(show_flags.astype(np.int8)).alias("show"))
    
    # 8. 统计和记录结果
    if "show" in target_df.columns:
        recommended_count = target_df.select(pl.col("show").sum()).item()
        logger.info(f"推荐完成: 总计 {target_df.height} 篇论文中推荐 {recommended_count} 篇 ({(recommended_count/target_df.height*100 if target_df.height > 0 else 0.0):.2f}%)")

        if not force_include_all:
            score_intervals = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
            for low, high in score_intervals:
                interval_mask = (target_scores >= low) & (target_scores < high)
                if np.sum(interval_mask) > 0:
                    interval_show = np.sum(show_flags & interval_mask) 
                    interval_total = np.sum(interval_mask)
                    logger.info(f"分数 {low:.1f}-{high:.1f}: {interval_show}/{interval_total} ({(interval_show/interval_total*100 if interval_total > 0 else 0.0):.2f}%)")
    else:
        logger.warning("'show' column not found in target_df after processing. Cannot log recommendation stats.")

    # 9. 返回结果DataFrame
    return target_df


def adaptive_sample(scores, target_sample_rate=0.15, high_threshold=0.95, boundary_threshold=0.5, random_state=42):
    """自适应采样算法，根据分数决定哪些样本应该被推荐。
    
    策略:
    1. 所有高于high_threshold的样本被优先推荐
    2. 如果高分样本数量超过目标数量，随机抽取一部分
    3. 如果高分样本不足，从boundary_threshold到high_threshold之间的分数中按权重采样
    
    Args:
        scores: 每个样本的预测分数
        target_sample_rate: 目标推荐比例
        high_threshold: 高置信度阈值
        boundary_threshold: 边界阈值
        random_state: 随机种子
    
    Returns:
        布尔数组，标记哪些样本被推荐
    """
    np.random.seed(random_state)
    n_samples = len(scores)
    target_count = int(n_samples * target_sample_rate)
    
    if target_count <= 0:
        target_count = 1
    
    show_flags = np.zeros(n_samples, dtype=bool)
    
    high_score_mask = scores >= high_threshold
    high_score_indices = np.where(high_score_mask)[0]
    high_score_count = len(high_score_indices)
    
    logger.info(f"目标推荐数量: {target_count} / {n_samples} ({target_sample_rate*100:.2f}%)")
    logger.info(f"高分样本(>={high_threshold:.4f})数量: {high_score_count} ({high_score_count/n_samples*100:.2f}%)")
    
    if high_score_count >= target_count:
        if high_score_count > target_count:
            selected_indices = np.random.choice(high_score_indices, target_count, replace=False)
            show_flags[selected_indices] = True
            logger.info(f"高分样本超过目标数量，随机选择了{target_count}个")
        else:
            show_flags[high_score_indices] = True
    else:
        # 先标记所有高分样本
        show_flags[high_score_indices] = True
        remaining_count = target_count - high_score_count
        
        # 找出边界区域的样本
        boundary_mask = (scores >= boundary_threshold) & (scores < high_threshold)
        boundary_indices = np.where(boundary_mask)[0]
        boundary_count = len(boundary_indices)
        
        logger.info(f"边界样本({boundary_threshold:.4f}-{high_threshold:.4f})数量: {boundary_count} ({boundary_count/n_samples*100:.2f}%)")
        
        if boundary_count == 0:
            # 如果没有边界样本，从所有剩余样本中随机选择
            remaining_indices = np.where(~high_score_mask)[0]
            if len(remaining_indices) > 0:
                if len(remaining_indices) > remaining_count:
                    # 随机选择一部分
                    selected_indices = np.random.choice(remaining_indices, remaining_count, replace=False)
                else:
                    # 全部选择
                    selected_indices = remaining_indices
                
                show_flags[selected_indices] = True
                logger.info(f"无边界样本，从所有剩余样本中随机选择了{len(selected_indices)}个")
        else:
            # 从边界区域按权重采样
            # 计算边界区域样本的权重（分数越高权重越大）
            boundary_scores = scores[boundary_indices]
            # 归一化到[0,1]区间，提高对比度
            min_score = boundary_threshold
            max_score = high_threshold
            normalized_scores = (boundary_scores - min_score) / (max_score - min_score)
            # 使用指数函数增强差异 (可选)
            weights = np.exp(normalized_scores * 2)  # 乘以2是为了增加对比度
            weights = weights / np.sum(weights)
            
            # 加权采样
            sample_size = min(remaining_count, boundary_count)
            selected_indices = np.random.choice(
                boundary_indices, sample_size, replace=False, p=weights
            )
            show_flags[selected_indices] = True
            
            logger.info(f"从边界区域加权采样了{len(selected_indices)}个样本")
            
            # 如果边界区域样本数量仍不足，从低分区域随机采样补足
            if sample_size < remaining_count:
                still_remaining = remaining_count - sample_size
                low_score_mask = scores < boundary_threshold
                low_score_indices = np.where(low_score_mask)[0]
                
                if len(low_score_indices) > 0:
                    sample_size_low = min(still_remaining, len(low_score_indices))
                    selected_indices_low = np.random.choice(low_score_indices, sample_size_low, replace=False)
                    show_flags[selected_indices_low] = True
                    logger.info(f"从低分区域({boundary_threshold:.4f}以下)随机采样了{len(selected_indices_low)}个样本")
    
    return show_flags


def predict(model, remaining_df: pl.DataFrame, config: Config, force_include_all: bool = False) -> pl.DataFrame:
    """预测、推荐并保存结果。
    
    Args:
        model: 训练好的模型
        remaining_df: 剩余数据
        config: 配置参数
        force_include_all: If True, all items in remaining_df will be marked as 'show'=1, bypassing adaptive sampling.
    
    Returns:
        推荐的DataFrame
    """
    # 执行预测和推荐
    results_df = predict_and_recommend(model, remaining_df, config, force_include_all=force_include_all)
    
    # 如果结果为空，提前返回
    if results_df.is_empty():
        logger.warning("没有推荐结果可保存")
        return results_df
    
    # 移除embedding列以减小文件大小
    data_config = config.recommend_pipeline.data
    embedding_columns = data_config.embedding_columns
    if embedding_columns:
        results_df = results_df.drop(*embedding_columns)
    # 保存结果到CSV
    output_file = REPO_ROOT / config.recommend_pipeline.predict.output_path
    
    # 保存
    # results_df.write_parquet(output_file)
    
    # 提取并返回推荐的论文
    if "show" not in results_df.columns:
        logger.error("'show' column is missing from results_df. Cannot filter recommended results.")
        return pl.DataFrame() 

    recommended_results = results_df.filter(pl.col("show") == 1)
    logger.info(f"推荐{recommended_results.height}篇论文")
    show_df = recommended_results.select("id", "title", "abstract", "score")
    logger.debug(f"{show_df}")
    
    return recommended_results


if __name__ == "__main__":
    from src.trainer import train_model
    config = Config.default()
    prefered_df, remaining_df = load_dataset(config)
    prefered_df = prefered_df.collect()
    remaining_df = remaining_df.collect()
    final_model = train_model(prefered_df, remaining_df, config)
    predict(final_model, remaining_df, config)
    logger.info("Final model training successfully completed")