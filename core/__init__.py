"""ALTTRNET core infrastructure — shared by all ingestion entry points."""

from core.config import PROJECT, PATHS, MODELS, CHUNKING, RETRIEVAL, CHROMA, ENV
from core.seeds import set_global_seed, get_global_seed, derive_seed
from core.logging import get_logger
from core.experiment import ExperimentMeta
from core.eval_format import EvalResult, EvalSuite

__all__ = [
    "PROJECT",
    "PATHS",
    "MODELS",
    "CHUNKING",
    "RETRIEVAL",
    "CHROMA",
    "ENV",
    "set_global_seed",
    "get_global_seed",
    "derive_seed",
    "get_logger",
    "ExperimentMeta",
    "EvalResult",
    "EvalSuite",
]
