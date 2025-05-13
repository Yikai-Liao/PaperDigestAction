from pydantic import BaseModel, Field
from pathlib import Path
import toml
from typing import List, Dict, Any, Optional, Union
import os
import copy


class LogisticRegressionConfig(BaseModel):
    """Configuration for logistic regression model parameters"""
    C: float = Field(1, description="Inverse of regularization strength")
    max_iter: int = Field(1000, description="Maximum number of iterations")


class TrainerConfig(BaseModel):
    """Configuration for model training"""
    seed: int = Field(42, description="Random seed")
    bg_sample_rate: float = Field(5.0, description="Background sampling rate")
    logci_regression: LogisticRegressionConfig = Field(
        default_factory=LogisticRegressionConfig,
        description="Logistic regression model configuration"
    )


class DataConfig(BaseModel):
    """Configuration for data parameters"""
    categories: List[str] = Field(
        ["cs.CL", "cs.CV", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.CY"],
        description="Paper categories"
    )
    embedding_columns: List[str] = Field(..., description="Columns for embeddings")
    preference_dir: str = Field("./preference", description="Preference data directory")
    background_start_year: int = Field(2024, description="Start year for background data")
    preference_start_year: int = Field(2023, description="Start year for preference data")
    embed_repo_id: str = Field("lyk/ArxivEmbedding", description="Hugging Face repository ID for embeddings")
    content_repo_id: str = Field("lyk/ArxivContent", description="Hugging Face repository ID for generated contents")
    cache_dir: str = Field(..., description="Cache directory for datasets")

class PredictConfig(BaseModel):
    """Configuration for prediction parameters"""
    last_n_days: int = Field(7, description="Number of recent days to consider")
    start_date: str = Field(..., description="The start date to recommend, empty to use last_n_days")
    end_date: str = Field(..., description="End date to recommend, empty to use last_n_days")
    high_threshold: float = Field(0.85, description="High threshold")
    boundary_threshold: float = Field(0.6, description="Boundary threshold")
    sample_rate: float = Field(0.004, description="Sampling rate")
    output_path: str = Field(..., description="Output path for predictions")

class RecommendPipelineConfig(BaseModel):
    """Configuration for recommendation system"""
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data-related configuration"
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Trainer configuration"
    )
    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Prediction configuration"
    )


class PdfConfig(BaseModel):
    """Configuration for PDF processing"""
    output_dir: str = Field("./pdfs", description="PDF output directory")
    delay: int = Field(3, description="Delay time in seconds")
    max_retry: int = Field(3, description="Maximum retry attempts")
    model: str = Field("grok-3", description="Language model alias to use")
    language: str = Field("en", description="Language for processing")
    enable_latex: bool = Field(False, description="Enable LaTeX in AI summary")


class SummaryPipelineConfig(BaseModel):
    """Configuration for summary generation pipeline"""
    pdf: PdfConfig = Field(
        default_factory=PdfConfig,
        description="PDF processing configuration"
    )


class LLMConfig(BaseModel):
    """Configuration for language models"""
    alias: str = Field(..., description="Model alias")
    name: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key, can be an environment variable name")
    temperature: float = Field(0.1, description="Temperature parameter")
    top_p: float = Field(0.8, description="Top-p parameter")
    num_workers: int = Field(1, description="Number of worker threads")
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort level")


class Config(BaseModel):
    """
    Main configuration class containing all configuration items
    """
    recommend_pipeline: RecommendPipelineConfig = Field(
        default_factory=RecommendPipelineConfig,
        description="Recommendation system configuration"
    )
    summary_pipeline: SummaryPipelineConfig = Field(
        default_factory=SummaryPipelineConfig,
        description="Summary generation pipeline configuration"
    )
    llms: List[LLMConfig] = Field(
        default_factory=list,
        description="List of language model configurations"
    )

    @classmethod
    def from_toml(cls, file_path: str) -> "Config":
        """
        Load configuration from a TOML file
        
        Args:
            file_path: Path to the TOML configuration file
            
        Returns:
            Config instance with loaded configuration
        """
        with open(file_path, 'r') as f:
            config_data = toml.load(f)
        return cls(**config_data)
    
    @classmethod
    def default(cls) -> "Config":
        """
        Load the default configuration
        
        Returns:
            Config instance with default configuration
        """
        return cls.from_toml("config.toml")
    
    def get_model(self, alias: str) -> Optional[LLMConfig]:
        """
        Get language model configuration by alias and process environment variables
        
        Args:
            alias: Model alias to look for
            
        Returns:
            Processed model configuration if found, None otherwise
        """
        for llm in self.llms:
            if llm.alias == alias:
                # Create a deep copy of the configuration to avoid modifying the original
                model_config = copy.deepcopy(llm)
                
                # Process API keys in environment variable format
                if model_config.api_key and model_config.api_key.startswith("env:"):
                    env_var = model_config.api_key[4:]  # Extract environment variable name
                    model_config.api_key = os.environ.get(env_var, "")
                    if not model_config.api_key:
                        raise ValueError(f"Environment variable {env_var} not set or empty")
                
                return model_config
        return None
    
__all__ = [
    "Config",
    "LLMConfig",
    "RecommendPipelineConfig",
    "TrainerConfig",
    "DataConfig",
    "PredictConfig",
    "PdfConfig",
    "SummaryPipelineConfig",
]

if __name__ == "__main__":
    # Test configuration loading
    config = Config.default()
    print(config)
    print(config.get_model("grok-3"))