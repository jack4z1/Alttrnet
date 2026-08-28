"""
core/config.py — ALTTRNET central configuration
=================================================
Single source of truth for project-level constants, paths, and model
settings. All other modules should import from here instead of
hardcoding values.

This is a FOUNDATION module — it defines the configuration structure
without choosing specific architectures, training strategies, or
data mixtures. Those decisions belong to the research layer.

Usage:
    from core.config import PROJECT, PATHS, MODELS, CHUNKING

    print(PROJECT.name)          # "alttrnet"
    print(PATHS.chroma_db)       # Path to ChromaDB directory
    print(MODELS.embed)          # "nomic-embed-text"
    print(CHUNKING.size)         # 400
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectConfig:
    """Project identity and versioning."""
    name: str = "alttrnet"
    version: str = "0.1.0"
    max_parameters: int = 17_000_000_000  # 17B hard cap
    description: str = (
        "Efficient coding-focused AI with system/model co-design: "
        "reasoning, tools, web research, external memory, verification, "
        "inference-time scaling, distillation."
    )


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Canonical paths relative to the project root."""
    # Computed from this file's location (core/config.py -> project root)
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def chroma_db(self) -> Path:
        return self.root / "chroma_db"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def tests(self) -> Path:
        return self.root / "tests"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Model names and settings used by the current RAG prototype."""
    # Embedding model (frozen)
    embed: str = "nomic-embed-text"

    # LLM for answering (frozen prototype)
    llm: str = "qwen3:14b"

    # Cross-encoder reranker (frozen, validated)
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    """Frozen chunking parameters (validated by Step 1B experiments)."""
    size: int = 400       # words per chunk
    overlap: int = 50     # words of overlap
    step: int = 350       # computed: size - overlap

    def __post_init__(self):
        # frozen=True dataclass — validate via __init_subclass__ alternative
        pass

    @property
    def effective_step(self) -> int:
        return self.size - self.overlap


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    """Frozen retrieval parameters (validated by Step 1B.6b)."""
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60  # kept for backward-compat with eval scripts


# ---------------------------------------------------------------------------
# ChromaDB configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB collection settings."""
    collection_name: str = "knowledge_base"
    space: str = "cosine"


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """Expected environment requirements."""
    python_min: str = "3.10"
    required_ollama_models: tuple = (  # type: ignore[assignment]
        "nomic-embed-text",
        "qwen3:14b",
    )


# ---------------------------------------------------------------------------
# Singleton instances — import these
# ---------------------------------------------------------------------------

PROJECT = ProjectConfig()
PATHS = PathConfig()
MODELS = ModelConfig()
CHUNKING = ChunkingConfig()
RETRIEVAL = RetrievalConfig()
CHROMA = ChromaConfig()
ENV = EnvConfig()
