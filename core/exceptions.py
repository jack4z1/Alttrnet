"""
core/exceptions.py — ALTTRNET custom exception hierarchy
=========================================================
Standardized error types so that callers can catch specific failure
modes and produce useful diagnostics.

Usage:
    from core.exceptions import (
        AlttrnetError,
        ConfigError,
        DatasetError,
        CheckpointError,
        TrainingError,
        PipelineError,
        TokenizerError,
        RetrievalError,
        ValidationError,
    )
"""


class AlttrnetError(Exception):
    """Base exception for all Alttrnet errors."""

    def __init__(self, message: str = "", *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{base} [{detail_str}]"
        return base


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigError(AlttrnetError):
    """Raised when configuration is invalid or missing."""


class EnvError(AlttrnetError):
    """Raised when environment variables or dependencies are missing."""


# ---------------------------------------------------------------------------
# Data / Dataset errors
# ---------------------------------------------------------------------------

class DatasetError(AlttrnetError):
    """Raised when a dataset is invalid or cannot be loaded."""


class DatasetManifestError(DatasetError):
    """Raised when a dataset manifest is malformed or incomplete."""


class DatasetValidationError(DatasetError):
    """Raised when dataset content fails validation checks."""


# ---------------------------------------------------------------------------
# Checkpoint errors
# ---------------------------------------------------------------------------

class CheckpointError(AlttrnetError):
    """Raised when a checkpoint cannot be saved, loaded, or validated."""


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------

class TrainingError(AlttrnetError):
    """Raised when a training run encounters a fatal error."""


class TrainingConfigError(TrainingError, ConfigError):
    """Raised when training configuration is invalid."""


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------

class PipelineError(AlttrnetError):
    """Raised when a data pipeline stage fails."""


class TokenizerError(PipelineError):
    """Raised when tokenization fails."""


# ---------------------------------------------------------------------------
# Retrieval / RAG errors
# ---------------------------------------------------------------------------

class RetrievalError(AlttrnetError):
    """Raised when the retrieval pipeline fails."""


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ValidationError(AlttrnetError):
    """Raised when an input fails schema or content validation."""

    def __init__(self, message: str = "", *, field: str = "", value: object = None):
        super().__init__(message)
        self.field = field
        self.value = value

    def __str__(self) -> str:
        base = super().__str__()
        if self.field:
            return f"{base} (field={self.field!r}, value={self.value!r})"
        return base


# ---------------------------------------------------------------------------
# Experiment / run errors
# ---------------------------------------------------------------------------

class ExperimentError(AlttrnetError):
    """Raised when experiment tracking or run management fails."""
