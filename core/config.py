"""
Configuration file for Agentic ReID system
Contains all hyperparameters and settings for the framework
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for vision-language models"""
    # Visual backbone settings
    clip_model_name: str = "ViT-B/16"
    clip_pretrained: str = "openai"
    embedding_dim: int = 512
    
    # LLM settings
    llm_model: str = "gemini-2.0-flash-exp"  # or "gpt-4", "claude-3-sonnet"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    
    # Device
    device: str = "cuda"  # or "cpu"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval and reranking"""
    # Identity-level retrieval settings
    use_identity_aggregation: bool = True
    aggregation_method: str = "top_m_mean"  # "max", "mean", "top_m_mean"
    top_m_views: int = 3  # For top-m mean pooling
    
    # Candidate pool settings
    num_candidates: int = 100
    num_candidate_groups: int = 5
    
    # Negation-explicit reranking
    negation_penalty_alpha: float = 0.6
    positive_boost_beta: float = 0.3
    
    # Confidence weighting
    use_confidence_weighting: bool = True
    confidence_temperature: float = 1.0


@dataclass
class InteractionConfig:
    """Configuration for interactive dialogue"""
    max_turns: int = 10
    question_pool_size: int = 5
    
    # Question selection method
    selection_method: str = "counterfactual_elimination"  # or "kl_divergence"
    
    # Redundancy filtering
    filter_redundant_questions: bool = True
    
    # User simulation (for experiments)
    simulate_user: bool = True
    user_uncertainty_prob: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for experiments and evaluation"""
    dataset_name: str = "CUHK-PEDES"
    dataset_path: str = "./data/CUHK-PEDES"
    
    # Evaluation settings
    rank_k_values: List[int] = None
    compute_map: bool = True
    compute_ats: bool = True
    
    # Logging
    log_dir: str = "./logs"
    save_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.rank_k_values is None:
            self.rank_k_values = [1, 5, 10]


@dataclass
class AgenticReIDConfig:
    """Main configuration combining all sub-configs"""
    model: ModelConfig = None
    retrieval: RetrievalConfig = None
    interaction: InteractionConfig = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.interaction is None:
            self.interaction = InteractionConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()


def get_default_config() -> AgenticReIDConfig:
    """Get default configuration"""
    return AgenticReIDConfig()


def load_config_from_dict(config_dict: dict) -> AgenticReIDConfig:
    """Load configuration from dictionary"""
    model_config = ModelConfig(**config_dict.get("model", {}))
    retrieval_config = RetrievalConfig(**config_dict.get("retrieval", {}))
    interaction_config = InteractionConfig(**config_dict.get("interaction", {}))
    experiment_config = ExperimentConfig(**config_dict.get("experiment", {}))
    
    return AgenticReIDConfig(
        model=model_config,
        retrieval=retrieval_config,
        interaction=interaction_config,
        experiment=experiment_config
    )