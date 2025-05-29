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
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from multiprocessing.dummy import Pool
from openai import OpenAI
from itertools import chain
from timeout_decorator import timeout_decorator
import logging
import inspect
import re # Import re for wildcard matching

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Configure logging to use loguru
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Set loguru to only show INFO and above
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add new handler with INFO level

# Set root logger to ERROR level to reduce noise
logging.getLogger().setLevel(logging.ERROR)

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

class KeywordMap(BaseModel):
    """Model representing a single keyword mapping"""
    key: str = Field(description="The redundant or incorrect keyword")
    values: List[str] = Field(description="List of standard replacement keywords")

class KeywordMerge(BaseModel):
    draft: str = Field(description="Please reasoning step by step here to get a more accurate and explainable merged_keywords.")
    merged_keywords: List[KeywordMap] = Field(
        description="A list of keyword mappings where each item maps a redundant keyword to its standard replacements",
    )

def load_cached_summaries(paper_ids: List[str], config: Config) -> pl.DataFrame:
    """
    Loads cached paper summaries from Hugging Face Datasets based on paper IDs and configuration.

    Args:
        paper_ids: A list of paper IDs (arxiv_ids) to look for in the cache.
        config: The configuration object, used to determine acceptable cache models and target language.

    Returns:
        A Polars DataFrame containing the cached summaries that match the criteria.
        Returns an empty DataFrame if no cached summaries are found or an error occurs.
    """
    logger.info(f"Loading cached summaries for {len(paper_ids)} paper IDs.")
    if not paper_ids:
        logger.info("No paper IDs provided to load_cached_summaries.")
        return pl.DataFrame()

    acceptable_cache_models = config.summary_pipeline.pdf.acceptable_cache_model
    target_language = config.summary_pipeline.pdf.language
    hf_dataset_repo_base = "hf://datasets/lyk/ArxivSummarize"

    # Determine unique prefixes from paper_ids
    unique_prefixes = set()
    for pid in paper_ids:
        match = re.match(r"^(\d{4})\.", pid)
        if match:
            unique_prefixes.add(match.group(1))
    
    if not unique_prefixes:
        logger.warning("No valid prefixes could be extracted from the provided paper_ids.")
        return pl.DataFrame()

    links = [f"{hf_dataset_repo_base}/{prefix}.jsonl" for prefix in unique_prefixes]
    dfs = []
    for link in links:
        try:
            dfs.append(pl.read_ndjson(link))
        except Exception as e:
            logger.warning(f"Failed to read dataset from {link}: {e}")
    if not dfs:
        logger.error("No valid datasets were loaded from the provided links.")
        return pl.DataFrame()
    df = pl.concat(dfs)
    logger.info(f"Loaded {df.height} cached summaries from {len(links)} links.")

    df = df.lazy().filter(
        (pl.col("id").is_in(pl.Series(paper_ids).implode())) &
        (pl.col("lang") == target_language) &
        (pl.col("model").str.contains(r"^(" + '|'.join(acceptable_cache_models) + r")"))
    )
    # sort by time, and unique by id to keep the latest version of each paper
    df = df.sort("summary_time", descending=True).unique(subset="id", keep="first").collect()
    logger.info(f"Loaded {df.height} cached summaries for {len(paper_ids)} requested paper IDs.")
    # add column 'preference' with default value 'unknown'
    if 'preference' not in df.columns:
        df = df.with_columns(pl.lit('unknown').alias('preference'))
    return df
    
def summarize(recommended_df:pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Summarize using AI, with a caching layer.
    """
    # Step 1: Load cached summaries
    cached_df = load_cached_summaries(recommended_df["id"].to_list(), config)
    uncached_df = recommended_df.filter(~pl.col("id").is_in(cached_df["id"]))

    markdowns: dict[str, str] = extract_and_convert_papers(uncached_df)

    llm_config = config.get_model(config.summary_pipeline.pdf.model)
    
    if llm_config is None:
        logger.error(f"Model configuration not found: {config.summary_pipeline.pdf.model}")
        exit(1) # Or handle more gracefully, e.g., return pl.DataFrame()

    native_json_schema: bool = llm_config.native_json_schema

    with open(REPO_ROOT / "summary_example.json", "r", encoding="utf-8") as f:
        example = f.read()
    with open(REPO_ROOT / "keywords.json", "r", encoding="utf-8") as f:
        keywords_data = json.load(f) # Renamed to avoid conflict
    keywords_list = list(chain.from_iterable(keywords_data.values())) # Renamed to avoid conflict

    base_prompt = f"""You are now a top research expert, but due to urgently needing funds to treat your mother's cancer, you have accepted a task from the giant company: you need to pretend to be an AI assistant, helping users deeply understand papers in exchange for high remuneration. 
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
    Here is an comprehensive potential keywords list: {keywords_list}. Please use the existing keywords first, and if you can't find a suitable one, please create a new one following the concept level similar to the existing ones.
    Do not add more than 6 keywords for 1 paper, always be concise and clear. Rember to use the existing keywords first and be really careful for the abbreviations, do not use abbreviations that are not in the list.
    
    Also, please provide a URL-friendly string that summarizes the title of the research (slug).
    Although I talked to you in English, but you need to make sure that your answer is in {config.summary_pipeline.pdf.language}. But always use English for the keywords slug and institution. 
    """
    logger.debug(f"enable-latex: {config.summary_pipeline.pdf.enable_latex}")
    if config.summary_pipeline.pdf.enable_latex:
        base_prompt += "    Also, you need to know that, your structured answer will rendered in markdown, so please also use the markdown syntax, especially for latex formula using $...$ or $$...$$.\n"
    else:
        base_prompt += "    Also, you need to know that, your answer will be rendered in a platform without support for latex equations, so please do not use any latex syntax in your answer, use more UTF-8 equivalent characters instead, like  ∫ ∑ ∏ ∈ ∉ ∪ ∩ ⊂ ⊆ ⊄ ⊈ ⊇ ⊃ ⊄ ⊈ ⊇ ⊃ ∀ ∃ ∄ ∴ ∵ ∷ ≡ ≠ ≈ ≅ ≪ ≫ ≤ ≥ < > ≤ ≥ < > ≪ ≫ ≤ ≥ < > ≪ ≫ ≤ ≥ < >\n"

    
    system_content_for_api = f"{base_prompt}\n. In the end, please carefully organized your answer into JSON format and take special care to ensure the Escape Character in JSON. When generating JSON, ensure that newlines within string values are represented using the escape character.\nHere is an example, but just for the format, you should give more detailed answer.\n{example}"
    
    if not native_json_schema:
        logger.warning(f"Model {llm_config.name} does not support native JSON Schema, falling back to prompt constraints for paper summarization.")
        schema_instruction = f"\nPlease ensure your output strictly adheres to the following Pydantic model's JSON Schema definition:\n{PaperSummary.model_json_schema()}"
        system_content_for_api += schema_instruction
        system_content_for_api += "Only output the JSON content, do not add any other content."

    output_path = REPO_ROOT / "arxiv/summary"
    output_path.mkdir(parents=True, exist_ok=True)
    
    def proc_one(paper_id):
        try:
            paper_content = markdowns[paper_id] # Renamed to avoid conflict
            logger.info(f"Paper content length for {paper_id}: {len(paper_content)} characters")
            logger.info(f"System content length for API: {len(system_content_for_api)} characters")
            client = OpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url
            )
            if not native_json_schema:
                raw_response = client.chat.completions.create(
                    model=llm_config.name,
                    temperature=llm_config.temperature,
                    top_p=llm_config.top_p,
                    messages=[
                        {"role": "system", "content": system_content_for_api},
                        {"role": "user", "content": f"The content of the paper is as follows:\n\n\n{paper_content}"},
                    ],
                )
                json_content = raw_response.choices[0].message.content
                logger.info(f"Raw JSON response content length for {paper_id}: {len(json_content)} characters")
                summary = PaperSummary.model_validate_json(json_content)
            else:
                api_response = client.beta.chat.completions.parse(
                    model=llm_config.name,
                    temperature=llm_config.temperature,
                    top_p=llm_config.top_p,
                    messages=[
                        {"role": "system", "content": system_content_for_api},
                        {"role": "user", "content": f"The content of the paper is as follows:\n\n\n{paper_content}"},
                    ],
                    reasoning_effort=llm_config.reasoning_effort, # Assuming this is a valid param for your OpenAI client setup (e.g. instructor)
                    response_format=PaperSummary,
                )
                # For native_json_schema, the parsing happens internally, so we can't easily log raw json_content here.
                # The error "Could not parse response content as the length limit was reached" is likely from the internal parsing.
                summary = api_response.choices[0].message.parsed

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
            logger.error(f"Error processing paper {paper_id}: {str(e)}")
            # Return a dictionary with error status instead of raising an exception directly
            return {
                'error': str(e),
                'id': paper_id,
                'summary_time': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
    
    # Using multi-threading
    with Pool(llm_config.num_workers) as pool:
        results = list(tqdm(pool.imap(proc_one, markdowns.keys()), total=len(markdowns), desc="Processing papers"))
    
    meta_datas = {data['id']: data for data in recommended_df.to_dicts()}
    results = {result['id']: result for result in results if 'error' not in result}
    for cached_data in cached_df.to_dicts():
        results[cached_data['id']] = cached_data # Add cached data to results
    
    for k, v in results.items():
        if k in meta_datas: # Ensure key exists before updating
            v.update(meta_datas[k])
        else:
            logger.warning(f"Metadata for paper ID {k} not found in recommended_df. Skipping metadata update for this paper.")

    if len(results) == 0:
        logger.error("No papers were processed successfully.")
        return pl.DataFrame()
        
    results_df = pl.from_dicts(list(results.values()))
    return results_df

def merge_keywords(results_df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Merge similar keywords using LLM, eliminate duplicates, and update the keyword list in the DataFrame.
    
    Args:
        results_df: DataFrame containing paper summary results, must include a 'keywords' column.
        config: Configuration object, used to get model configuration.
        
    Returns:
        pl.DataFrame: DataFrame with updated 'keywords' column.
    """
    if 'keywords' not in results_df.columns or results_df.is_empty():
        logger.error("The 'keywords' column is missing or the DataFrame is empty, cannot perform keyword merging.")
        return results_df
        
    all_keywords = results_df['keywords'].to_list()
    llm_config = config.get_model(config.summary_pipeline.pdf.model)
    if llm_config is None:
        logger.error(f"Model configuration not found: {config.summary_pipeline.pdf.model}")
        return results_df # Or raise an error

    native_json_schema: bool = llm_config.native_json_schema
    client = OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url
    )
    
    with open(REPO_ROOT / "keywords.json", "r", encoding="utf-8") as f:
        reference_keywords_data = json.load(f) # Renamed
        reference_keywords_list = list(chain.from_iterable(reference_keywords_data.values())) # Renamed
    
    base_user_prompt = f"""You are a keyword merging expert. Given the following list of keywords from multiple papers: {json.dumps(all_keywords, ensure_ascii=False)}.
    Please analyze and output a dictionary where keys are redundant or incorrect keywords, and values are lists of corresponding standard keywords (usually with one element).
    For example, 'LLM' as a redundant keyword maps to ['LLMs']; 'Large Language Model' maps to ['LLMs'].
    However, there might be multiple elements, for instance, 'Efficient Adaptive System' maps to ['Efficient', 'Adaptive System'].
    Please refer to the following keyword list as the preferred choice for standard keywords: {json.dumps(reference_keywords_list, ensure_ascii=False)}.
    If no suitable standard keyword exists in the current list, you can create a new one, but maintain a similar conceptual level. If a keyword does not need to be changed, you do not need to process it. Do not generate redundant content like 'AI in Security': ['AI in Security'] as it has no modifications.
    However, please be very careful not to over-merge keywords or lose significant information a keyword can provide. For example, merging 'Gradient Estimation': ['Efficiency'] or 'Zeroth-Order Optimization': ['Efficiency'] is incorrect as it eliminates valid information.
    The most common case you need to handle is merging synonymous keywords like 'LLM': ['Large Language Model'].
    Overly verbose keywords (e.g., using four or five words) should be split into a combination of concise keywords. I encourage using conceptual combinations to represent new concepts.
    Please ensure the output conforms to the KeywordMerge format."""
    user_prompt_for_api = base_user_prompt
    if not native_json_schema:
        logger.warning(f"Model {llm_config.name} does not support native JSON Schema, falling back to prompt constraints for keyword merging.")
        schema_instruction = f"\nPlease ensure your output strictly adheres to the following Pydantic model's JSON Schema definition:\n{KeywordMerge.model_json_schema()}"
        user_prompt_for_api += schema_instruction
        user_prompt_for_api += "Only output the JSON content, do not add any other content."
            
    logger.debug(f"Keyword merge prompt: {user_prompt_for_api}")
    
    system_message = {"role": "system", "content": "You are a keyword merging expert, specializing in mapping redundant or incorrect keywords to standard keywords."}
    user_message = {"role": "user", "content": user_prompt_for_api}
    messages_to_llm = [system_message, user_message]
    
    logger.debug(f"Messages sent to LLM for keyword merging: {messages_to_llm}")

    try:
        if not native_json_schema:
            raw_response = client.chat.completions.create(
                model=llm_config.name,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                messages=messages_to_llm,
            )
            json_content = raw_response.choices[0].message.content
            logger.info(f"Raw JSON response for keyword merging: {json_content}") # Added log
            
            # Remove markdown code block delimiters if present
            if json_content.startswith("```json") and json_content.endswith("```"):
                json_content = json_content[len("```json"): -len("```")].strip()
                logger.debug("Removed markdown code block delimiters from JSON content.")

            # Manually ensure merged_keywords is not empty before Pydantic validation
            try:
                temp_parsed_json = json.loads(json_content)
                if "merged_keywords" in temp_parsed_json and not temp_parsed_json["merged_keywords"]:
                    logger.warning("LLM returned empty 'merged_keywords'. Inserting dummy entry.")
                    temp_parsed_json["merged_keywords"] = {"NoChange": []}
                json_content = json.dumps(temp_parsed_json, ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON content before validation: {e}")
                # Re-raise or handle as appropriate, for now, proceed with original json_content
                pass
            
            parsed_result = KeywordMerge.model_validate_json(json_content)
        else:
            api_response = client.beta.chat.completions.parse(
                model=llm_config.name,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                messages=messages_to_llm,
                reasoning_effort=llm_config.reasoning_effort, # Assuming valid param
                response_format=KeywordMerge,
            )
            # For native_json_schema, the parsing happens internally, so we can't easily log raw json_content here.
            # The error "should be non-empty for OBJECT type" is likely from the internal parsing.
            parsed_result = api_response.choices[0].message.parsed
        
        logger.debug(f"Keyword merge response draft: {parsed_result.draft}")
        merged_keywords = parsed_result.merged_keywords
        merged_keywords = {kw.key: kw.values for kw in merged_keywords}  # Convert to dict for easier access
        logger.info(f"Keyword merging finished: {merged_keywords}")
        
        # Update keyword list using the mapping, replace redundant keywords with standard ones
        def update_keywords(keywords):
            if len(keywords) == 0:
                return keywords
            updated = set()
            for kw in keywords:
                # Skip the dummy entry if it exists
                if kw == 'NoChange' and merged_keywords.get(kw) == []:
                    continue
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
        logger.error(f"Error during keyword merging: {str(e)}")
        return results_df

@timeout_decorator.timeout(5)
def latex_file_to_json_str(latex_tar_gz_file: str | Path) -> str:
    # Set TexReader and related logging to ERROR level only
    loggers_to_silence = [
        'latex2json',
        'latex2json.tex_reader', 
        'latex2json.parser',
        'tex_reader',
        'chardet',
        'chardet.charsetgroupprober',
        'chardet.universaldetector',
        'charset_normalizer'
    ]
    
    for logger_name in loggers_to_silence:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Also silence the root logger temporarily
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    
    try:
        tex_reader = TexReader() # Initialize TexReader once
        structured_data_obj = tex_reader.process(str(latex_tar_gz_file))
        return tex_reader.to_json(structured_data_obj)
    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)

def extract_and_convert_papers(recommended_df: pl.DataFrame) -> dict[str, str]:
    """
    Extract all paper arXiv IDs from the recommended papers DataFrame, 
    download LaTeX source files, and convert them to JSON and Markdown formats.
    
    Args:
        recommended_df: DataFrame containing recommended paper information, 
                        must include an 'id' column for the paper's arXiv ID.
    """
    if 'id' not in recommended_df.columns:
        logger.error("The 'id' column is missing in the recommended papers DataFrame, cannot extract arXiv IDs.")
        return {} # Return empty dict on error
    
    # Create storage directories
    arxiv_base_dir = Path(REPO_ROOT) / "arxiv"
    latex_dir = arxiv_base_dir / "latex" # Stores the .tar.gz files
    json_dir = arxiv_base_dir / "json"
    markdown_dir = arxiv_base_dir / "markdown"
    
    for directory in [latex_dir, json_dir, markdown_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    arxiv_ids = recommended_df['id'].to_list()
    logger.info(f"Starting to process {len(arxiv_ids)} papers for extraction and conversion.")
    
    conversion_results ={} # Renamed

    for idx, arxiv_id in enumerate(arxiv_ids):
        logger.info(f"Processing paper {idx+1}/{len(arxiv_ids)} for extraction/conversion: {arxiv_id}")
        
        latex_tar_gz_file = latex_dir / f"{arxiv_id}.tar.gz"
        json_file = json_dir / f"{arxiv_id}.json"
        markdown_file = markdown_dir / f"{arxiv_id}.md"

        # Skip if Markdown already exists (optional, for resumability)
        if markdown_file.exists() and json_file.exists() and latex_tar_gz_file.exists():
            logger.info(f"Markdown, JSON, and LaTeX for {arxiv_id} already exist. Skipping download/conversion.")
            try:
                conversion_results[arxiv_id] = markdown_file.read_text(encoding='utf-8')
            except Exception as e:
                 logger.error(f"Failed to read existing markdown for {arxiv_id}: {e}")
            continue

        # 1. Download original LaTeX files (.tar.gz)
        if not latex_tar_gz_file.exists():
            try:
                source_url = f"https://arxiv.org/e-print/{arxiv_id}"
                response = requests.get(source_url, timeout=30)
                response.raise_for_status() # Raise an exception for HTTP errors
                with open(latex_tar_gz_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded LaTeX source file: {latex_tar_gz_file}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download LaTeX source file for {arxiv_id}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unknown error while downloading LaTeX source file for {arxiv_id}: {e}")
                continue
        else:
            logger.info(f"LaTeX source file already exists: {latex_tar_gz_file}")
        
        # 2. Process .tar.gz using TexReader and convert to JSON
        structured_data_obj = None
        if not json_file.exists():
            try:
                logger.info(f"Processing with TexReader: {latex_tar_gz_file}")
                json_output_str = latex_file_to_json_str(latex_tar_gz_file)
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(json_output_str)
                logger.info(f"Saved JSON file: {json_file}")
                # For markdown conversion, load the JSON string back to an object
                structured_data_obj = json.loads(json_output_str) 
            except Exception as e:
                logger.error(f"Error processing with TexReader or saving JSON for {arxiv_id}: {e}")
                if json_file.exists(): # Clean up partially written file
                    try:
                        os.remove(json_file)
                    except OSError as oe:
                        logger.error(f"Could not remove partially written JSON file {json_file}: {oe}")
                continue # Skip to next paper
        else:
            logger.info(f"JSON file already exists: {json_file}. Loading for Markdown conversion.")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    loaded_json_str = f.read()
                    structured_data_obj = json.loads(loaded_json_str) # Parse string to Python object
            except Exception as e:
                logger.error(f"Failed to load JSON file {json_file}: {e}")
                continue
 
        # 3. Convert to Markdown
        if structured_data_obj is not None:
            try:
                md_str = json_to_markdown(structured_data_obj, ignore_reference=True)
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(md_str)
                logger.info(f"Saved Markdown file: {markdown_file}")
                conversion_results[arxiv_id] = md_str
            except Exception as e:
                logger.error(f"Failed to convert JSON to Markdown for {arxiv_id}: {e}")
        else:
            logger.warning(f"Skipping Markdown conversion for {arxiv_id} due to missing structured data.")
            
    return conversion_results


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