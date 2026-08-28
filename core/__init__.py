"""ALTTRNET core infrastructure — shared by all ingestion entry points."""

from core.checkpoint import CheckpointMeta
from core.config import CHROMA, CHUNKING, ENV, MODELS, PATHS, PROJECT, RETRIEVAL
from core.config_loader import load_yaml, merge_configs
from core.datasets import DatasetManifest, DatasetRegistry, DatasetValidator
from core.env import get_env, load_env, require_env
from core.eval_format import EvalResult, EvalSuite
from core.exceptions import (
    AlttrnetError,
    CheckpointError,
    ConfigError,
    DatasetError,
    PipelineError,
    RetrievalError,
    TrainingError,
    ValidationError,
)
from core.experiment import ExperimentMeta
from core.logging import get_logger
from core.model_schema import ModelMetadata
from core.pipeline import DataSource, Pipeline, Tokenizer
from core.runs import RunManager
from core.seeds import derive_seed, get_global_seed, set_global_seed
from core.tokenizer import PlaceholderTokenizer, create_tokenizer
from core.training_config import TrainingConfig

__all__ = [
    # Config
    "PROJECT",
    "PATHS",
    "MODELS",
    "CHUNKING",
    "RETRIEVAL",
    "CHROMA",
    "ENV",
    # Seeds & logging
    "set_global_seed",
    "get_global_seed",
    "derive_seed",
    "get_logger",
    # Experiment & eval
    "ExperimentMeta",
    "EvalResult",
    "EvalSuite",
    # Exceptions
    "AlttrnetError",
    "ConfigError",
    "DatasetError",
    "CheckpointError",
    "TrainingError",
    "PipelineError",
    "RetrievalError",
    "ValidationError",
    # Datasets
    "DatasetManifest",
    "DatasetRegistry",
    "DatasetValidator",
    # Checkpoints
    "CheckpointMeta",
    # Runs
    "RunManager",
    # Training
    "TrainingConfig",
    # Pipeline
    "Pipeline",
    "DataSource",
    "Tokenizer",
    # Tokenizer
    "PlaceholderTokenizer",
    "create_tokenizer",
    # Model schema
    "ModelMetadata",
    # Environment
    "load_env",
    "require_env",
    "get_env",
    # Config loader
    "load_yaml",
    "merge_configs",
]
