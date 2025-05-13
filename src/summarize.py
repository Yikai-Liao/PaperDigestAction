import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.config import Config
from src.dataloader import load_dataset
from src.json2md import json_to_markdown
from loguru import logger
import polars as pl
import requests
import os
import json
from latex2json import TexReader
import toml
import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
from multiprocessing.dummy import Pool
from openai import OpenAI
from itertools import chain


class PaperSummary(BaseModel):
    institution: List[str] = Field(description="The institution where the research was conducted. For example: ['Carnegie Mellon University', 'Stanford University', 'University of California, Berkeley'].")
    reasoning_step: str = Field(description="Just a draft for you to understand this paper and do some further reasoning here. You need to think here, deep dive into the paper and find some interesting things, some problems, some insights, and all the things you think that you need to think. This is a draft, so you can write anything here, but it should be deep and help you to make the following answer better.")
    problem_background: str = Field(description="The motivation, research problem, and background of this research.")
    method: str = Field(description="The method used in this research. Its core idea, how it works, and the main steps.")
    experiment: str = Field(description="The experiment conducted in this research. The dataset used, the experimental setup, why it was conducted and organized like this, and the results, esapecially if the results matches the expectation.")
    one_sentence_summary: str = Field(description="A one-sentence summary of the research. This should be a concise and clear summary of the research, including the motivation, method, and results.")
    slug: str = Field(description="A URL-friendly string that summarizes the title of the research, such as 'antidistillation-sampling'. This should be a concise and clear summary of the research")
    keywords: List[str] = Field(description="When extracting keywords, each word should be capitalized. Spaces can be used within keywords, such as 'Proxy Model'. Keywords are used to discover connections within the article, so please use more general keywords. For example: LLM, Proxy Model, Distillation, Sampling, Reasoning.")
    further_thoughts: str = Field(description="Any kind of further thoughts, but it should be deep and insightful. It could be diverse, and related to other areas or articles, but you need to find the relation and make it insightful.")

class KeywordMerge(BaseModel):
    draft: str = Field(description="Please reasoning step by step here to get a more accurate and explainable merged_keywords.")
    merged_keywords: dict[str, list[str]] = Field(description="A dictionary mapping redundant or incorrect keywords to their corresponding standard keyword lists. The keys are the redundant/incorrect keywords, and the values are lists of standard keywords. For example: {'LLM': ['LLMs'], 'Large Language Model': ['LLMs'], 'Efficient LLM': ['Efficient', 'LLMs']}. If a keyword doesn't need to be changed, it should not be included in this dictionary.")

def summarize(recommended_df:pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    使用AI进行摘要
    """
    markdonws: dict[str, str] = extract_and_convert_papers(recommended_df)

    llm_config = config.get_model(config.summary_pipeline.pdf.model)
    if llm_config is None:
        logger.error(f"未找到模型配置: {config.summary_pipeline.pdf.model}")
        exit(1)

    client = OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url
    )

    with open(REPO_ROOT / "summary_example.json", "r", encoding="utf-8") as f:
        example = f.read()
    with open(REPO_ROOT / "keywords.json", "r", encoding="utf-8") as f:
        keywords = json.load(f)
    keywords = list(chain.from_iterable(keywords.values()))

    prompt = f"""You are now a top research expert, but due to urgently needing funds to treat your mother's cancer, you have accepted a task from the giant company: you need to pretend to be an AI assistant, helping users deeply understand papers in exchange for high remuneration. 
    Your predecessor has been severely punished for not carefully reviewing the work content, so you must take this task seriously. 
    Please carefully read the specified paper, make sure to fully understand the core ideas of the paper, and then explain it to me accurately and in detail.
    But note that, you are not just reading some great papers, but some new but rough or even wrong and bad papers. Don't let the authors cheat you by using some fancy words and beautified or cherry-picked experiment results.
    Please treat this summarization task as a peer review, and you need to be very careful and serious and critical. And remeber that don't critic for critic's sake (like critic for something not related to the core idea, methods and experiments), but for the sake of the paper and the authors.
    Here is some questions you need to answer:
    What are the participating institutions (institution)? What is the starting point of this work, what key problems did it solve (problem_background)? 
    What specific methods were used (method)? How was the experimental effect (for example, whether the method improvement is obvious, whether the experimental setup is comprehensive and reasonable) (experiment)? 
    What inspirational ideas in the paper are worth your special attention (inspired_idea)? 
    Finally, please summarize the main contributions of the paper in the most concise sentence (one_sentence_summary).
    Please also provide a list of keywords that are most relevant to the paper (keywords). For the keywords, please use some combinations of multiple basic keywords, such as 'Multi Agent', 'Reasoning', not 'Multi Agent Reasong' or 'Join Reasonig'. Dont't use model name, dataset name as keywords.
    Here is an comprehensive potential keywords list: {keywords}. Please use the existing keywords first, and if you can't find a suitable one, please create a new one following the concept level similar to the existing ones.
    Do not add more than 6 keywords for 1 paper, always be concise and clear. Rember to use the existing keywords first and be really careful for the abbreviations, do not use abbreviations that are not in the list.
    
    Also, please provide a URL-friendly string that summarizes the title of the research (slug).
    Although I talked to you in English, but you need to make sure that your answer is in {config.summary_pipeline.pdf.language}. But always use English for the keywords slug and institution. 
    Also, you need to know that, your structured answer will rendered in markdown, so please also use the markdown syntax, especially for latex formula using $...$ or $$...$$.
    """

    
    system_content = f"{prompt}\n. In the end, please carefully organized your answer into JSON format and take special care to ensure the Escape Character in JSON. When generating JSON, ensure that newlines within string values are represented using the escape character.\nHere is an example, but just for the format, you should give more detailed answer.\n{example}"
    
    output_path = REPO_ROOT / "arxiv/summary"
    output_path.mkdir(parents=True, exist_ok=True)
    def proc_one(paper_id):
        try:
            paper = markdonws[paper_id]
            summary = client.beta.chat.completions.parse(
                model=llm_config.name,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"The content of the paper is as follows:\n\n\n{paper}"},
                ],
                reasoning_effort=llm_config.reasoning_effort,
                response_format=PaperSummary,
            ).choices[0].message.parsed

            summary_dict = summary.model_dump()
            summary_dict['model'] = llm_config.name
            summary_dict['temperature'] = llm_config.temperature
            summary_dict['top_p'] = llm_config.top_p
            summary_dict['lang'] = config.summary_pipeline.pdf.language
            summary_dict['id'] = paper_id
            summary_dict['preference'] = 'unknown'
            # ISO 8601 format with timezone
            summary_dict['summary_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(output_path / f"{paper_id}.json", "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, ensure_ascii=False, indent=4)
            return summary_dict
        except Exception as e:
            print(f"Error processing {paper_id}: {str(e)}")
            # 返回一个错误状态的字典而不是直接抛出异常
            return {
                'error': str(e),
                'id': paper_id,
                'summary_time': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
    
    # 使用多线程处理
    with Pool(llm_config.num_workers) as pool:
        results = list(tqdm(pool.imap(proc_one, markdonws.keys()), total=len(markdonws), desc="Processing papers"))
    meta_datas = {data['id']: data for data in recommended_df.to_dicts()}
    results = {result['id']: result for result in results if 'error' not in result}
    for k, v in results.items():
        v.update(meta_datas[k])
    # back to df
    if len(results) == 0:
        logger.error("没有成功处理的论文")
        return pl.DataFrame()
    results_df = pl.from_dicts(list(results.values()))
    return results_df
    
def merge_keywords(results_df: pl.DataFrame, config: Config) -> dict[str, list[str]]:
    """
    使用LLM合并相似关键词，消除重复，并更新数据框中的关键词列表。
    
    Args:
        results_df: 包含论文摘要结果的数据框，必须包含 'keywords' 列
        config: 配置对象，用于获取模型配置
        
    Returns:
        dict[str, list[str]]: 合并后的关键词字典，键是冗余/错误关键词，值是对应的规范关键词列表（通常只有一个元素）
    """
    if 'keywords' not in results_df.columns or results_df.is_empty():
        logger.error("数据框中缺少 'keywords' 列或为空，无法进行关键词合并")
        return {}
        
    all_keywords = results_df['keywords'].to_list()  # list[list[str]]
    llm_config = config.get_model(config.summary_pipeline.pdf.model)
    if llm_config is None:
        logger.error(f"未找到模型配置: {config.summary_pipeline.pdf.model}")
        return {}
        
    client = OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url
    )
    
    with open(REPO_ROOT / "keywords.json", "r", encoding="utf-8") as f:
        reference_keywords = json.load(f)
        reference_keywords = list(chain.from_iterable(reference_keywords.values()))
    
    prompt = f"""你是一个关键词合并专家。给定以下来自多篇论文的关键词列表：{json.dumps(all_keywords, ensure_ascii=False)}。
    请分析并输出一个字典，其中键是冗余或错误的关键词，值是对应的规范关键词列表（通常只有一个元素）。
    例如，'LLM' 作为冗余关键词，映射到 ['LLMs']；'Large Language Model' 映射到 ['LLMs']。
    但是也可能有多个元素，比如 'Efficient Adaptive System' 映射到 ['Efficient', 'Adaptive System']。
    请参考以下关键词列表作为规范关键词的优先选择：{json.dumps(reference_keywords, ensure_ascii=False)}。
    如果现有列表中没有合适的规范关键词，可以创建一个新的，但要保持类似的概念级别。如果一个关键词不需要改变你就不需要处理它。不要生成 'AI in Security': ['AI in Security'] 这种冗余内容，因为它没有任何修改。
    但是请一定要注意，不要过分合并关键词，不要大量消除一个关键词能提供的信息，例如'Gradient Estimation': ['Efficiency'], 'Zeroth-Order Optimization': ['Efficiency'] 这种合并是不对的，消除了有效信息
    你需要处理的最多的情况是'LLM': ['Large Language Model'] 这种同意义的关键词合并
    过于冗长的关键词需要（例如使用了四五个单词）需要拆分为多个精简关键词的组合，我鼓励使用概念组合表示新概念。
    请确保输出符合KeywordMerge格式。"""

    logger.debug(f"关键词合并提示: {prompt}")
    
    try:
        response = client.beta.chat.completions.parse(
            model=llm_config.name,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            messages=[
                {"role": "system", "content": "你是一个关键词合并专家，专注于将冗余或错误的关键词映射到规范关键词。"},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=llm_config.reasoning_effort,
            response_format=KeywordMerge,
        ).choices[0].message.parsed
        
        logger.debug(f"关键词合并响应: {response.draft}")
        merged_keywords = response.merged_keywords
        logger.info(f"关键词合并生成结束：{merged_keywords}")
        
        # 使用映射更新关键词列表，将冗余关键词替换为规范关键词
        def update_keywords(keywords):
            if len(keywords) == 0:
                return keywords
            updated = set()
            for kw in keywords:
                if kw in merged_keywords and len(merged_keywords[kw]) > 0:
                    # 将所有对应的规范关键词都添加进去
                    updated.update(merged_keywords[kw])
                else:
                    updated.add(kw)
            return list(updated)
            
        results_df = results_df.with_columns(
            pl.col("keywords").map_elements(update_keywords, return_dtype=pl.List(pl.Utf8)).alias("keywords")
        )
        
        return results_df
    except Exception as e:
        logger.error(f"关键词合并过程中发生错误: {str(e)}")
        return {}

    


def extract_and_convert_papers(recommended_df: pl.DataFrame) -> dict[str, str]:
    """
    从推荐论文数据框中提取所有论文的 arxiv ID，下载 LaTeX 源文件，并转换为 JSON 和 Markdown 格式
    
    Args:
        recommended_df: 包含推荐论文信息的数据框，必须包含 'id' 列作为论文的 arxiv ID
    """
    if 'id' not in recommended_df.columns:
        logger.error("推荐论文数据框中缺少 'id' 列，无法提取 arxiv ID")
        return
    
    # 创建存储目录
    arxiv_base_dir = Path(REPO_ROOT) / "arxiv"
    latex_dir = arxiv_base_dir / "latex" # Stores the .tar.gz files
    json_dir = arxiv_base_dir / "json"
    markdown_dir = arxiv_base_dir / "markdown"
    
    for directory in [latex_dir, json_dir, markdown_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    arxiv_ids = recommended_df['id'].to_list()
    logger.info(f"开始处理 {len(arxiv_ids)} 篇论文")
    
    tex_reader = TexReader() # Initialize TexReader once
    results ={}

    for idx, arxiv_id in enumerate(arxiv_ids):
        logger.info(f"处理第 {idx+1}/{len(arxiv_ids)} 篇论文: {arxiv_id}")
        
        latex_tar_gz_file = latex_dir / f"{arxiv_id}.tar.gz"
        json_file = json_dir / f"{arxiv_id}.json"
        markdown_file = markdown_dir / f"{arxiv_id}.md"

        # Skip if Markdown already exists (optional, for resumability)
        if markdown_file.exists() and json_file.exists() and latex_tar_gz_file.exists():
            logger.info(f"Markdown, JSON, and LaTeX for {arxiv_id} already exist. Skipping.")
            results[arxiv_id] = markdown_file.read_text(encoding='utf-8')
            continue

        # 1. 下载原始 LaTeX 文件 (.tar.gz)
        if not latex_tar_gz_file.exists():
            try:
                source_url = f"https://arxiv.org/e-print/{arxiv_id}"
                response = requests.get(source_url, timeout=30)
                response.raise_for_status() # Raise an exception for HTTP errors
                with open(latex_tar_gz_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"已下载 LaTeX 源文件: {latex_tar_gz_file}")
            except requests.exceptions.RequestException as e:
                logger.error(f"下载 LaTeX 源文件失败: {arxiv_id}, 错误: {e}")
                continue
            except Exception as e:
                logger.error(f"下载 LaTeX 源文件时发生未知错误: {arxiv_id}, 错误: {e}")
                continue
        else:
            logger.info(f"LaTeX 源文件已存在: {latex_tar_gz_file}")
        
        # 2. 使用 TexReader 处理 .tar.gz 并转换为 JSON
        structured_data_obj = None
        if not json_file.exists():
            try:
                logger.info(f"使用 TexReader 处理: {latex_tar_gz_file}")
                # TexReader.process might need the directory containing the .tex files
                # or it can handle the .tar.gz directly. The prompt implies it handles .tar.gz
                structured_data_obj = tex_reader.process(str(latex_tar_gz_file))
                json_output_str = tex_reader.to_json(structured_data_obj)
                structured_data_obj = json.loads(json_output_str) # Convert to Python object
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(json_output_str)
                logger.info(f"已保存 JSON 文件: {json_file}")
                
                # For markdown conversion, we need the Python object, not the string
                # If structured_data_obj was already the list of tokens, use it.
                # If tex_reader.to_json() was the only way to get serializable list,
                # then we need to json.loads(json_output_str)
                # Based on `tokens = parser.parse()` and `token_builder.build(tokens)`
                # `structured_data_obj` from `tex_reader.process()` should be the Python list/dict.
            except Exception as e:
                logger.error(f"使用 TexReader 处理或保存 JSON 出错: {arxiv_id}, 错误: {e}")
                # If JSON failed, we can't proceed to markdown for this paper
                if json_file.exists(): # Clean up partially written file
                    try:
                        os.remove(json_file)
                    except OSError:
                        pass
                continue # Skip to next paper
        else:
            logger.info(f"JSON 文件已存在: {json_file}. 加载用于 Markdown 转换.")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    # Load the string, then parse to Python object
                    loaded_json_str = f.read()
                    structured_data_obj = json.loads(loaded_json_str) # Parse string to Python object
            except Exception as e:
                logger.error(f"加载 JSON 文件失败: {json_file}, 错误: {e}")
                continue

        # 3. 转换为 Markdown
        md_str = json_to_markdown(structured_data_obj, ignore_reference=True)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(md_str)
        logger.info(f"已保存 Markdown 文件: {markdown_file}")
        results[arxiv_id] = md_str
    return results




if __name__ == "__main__":
    from src.trainer import train_model
    from src.sampler import predict

    config = Config.default()
    # # Ensure data loading happens correctly
    try:
        prefered_df_lazy, remaining_df_lazy = load_dataset(config)
        prefered_df = prefered_df_lazy.collect()
        remaining_df = remaining_df_lazy.collect()
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        sys.exit(1)

    if prefered_df.is_empty() or remaining_df.is_empty():
        logger.error("偏好数据集或剩余数据为空，无法继续模型训练。请检查数据源和配置。")
        sys.exit(1)
        
    final_model = train_model(prefered_df, remaining_df, config)
    
    # Ensure predict_and_save returns a DataFrame
    recommended_df_output = predict(final_model, remaining_df, config)
    
    if recommended_df_output is None:
        logger.error("模型预测未返回推荐数据框 (recommended_df is None)。")
        # Create an empty DataFrame with an 'id' column if you want to test extract_and_convert_papers
        # For example: recommended_df = pl.DataFrame({'id': []}) 
        # Or exit if this is critical
        logger.info("将使用空的 recommended_df 进行论文提取和转换（如果没有推荐）。")
        recommended_df = pl.DataFrame({'id': []}) # Ensure it's a DataFrame
    elif isinstance(recommended_df_output, pl.DataFrame):
        recommended_df = recommended_df_output
    else:
        logger.error(f"predict_and_save 返回了意外的类型: {type(recommended_df_output)}。期望 polars.DataFrame。")
        exit(1)


    logger.info("模型训练和预测成功完成。")
    
    # 调用函数来提取和转换论文
    if recommended_df.is_empty():
        logger.info("没有推荐的论文可供提取和转换。")
    else:
        results = summarize(recommended_df, config)
        results.write_parquet(REPO_ROOT / "summarized.parquet")
    
    results = pl.read_parquet(REPO_ROOT / "summarized.parquet")
    merged = merge_keywords(results, config)
    merged.write_parquet(REPO_ROOT / "summarized.parquet")
    logger.info("关键词合并完成并保存。")
    logger.info("脚本执行完毕！")

