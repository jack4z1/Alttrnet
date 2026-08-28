"""
core/tokenizer.py — ALTTRNET tokenizer interface scaffold
===========================================================
Provides a wrapper interface for tokenizers. The specific tokenizer
algorithm (BPE, SentencePiece, etc.) and vocabulary will be chosen
by the research layer. This module provides:

1. An abstract Tokenizer class (inheriting from core.pipeline.Tokenizer)
2. A placeholder implementation for testing
3. Factory function for creating tokenizers from config

Usage:
    from core.tokenizer import create_tokenizer, PlaceholderTokenizer

    # Placeholder for testing infrastructure
    tok = PlaceholderTokenizer(vocab_size=1000)

    # Factory (will load real tokenizers when available)
    tok = create_tokenizer({"type": "placeholder", "vocab_size": 1000})
"""

from typing import Optional

from core.exceptions import TokenizerError
from core.pipeline import Tokenizer

# ---------------------------------------------------------------------------
# Placeholder tokenizer (for infrastructure testing)
# ---------------------------------------------------------------------------

class PlaceholderTokenizer(Tokenizer):
    """
    Simple character-level tokenizer for testing the pipeline.

    NOT suitable for actual training — this exists only so the pipeline
    infrastructure can be tested without a real tokenizer.
    """

    def __init__(self, vocab_size: int = 256):
        self._vocab_size = vocab_size

    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode by mapping each character to its ordinal (mod vocab_size)."""
        return [ord(c) % self._vocab_size for c in text]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode by mapping each token back to a character."""
        return "".join(chr(t) for t in token_ids)

    def vocab_size(self) -> int:
        return self._vocab_size


# ---------------------------------------------------------------------------
# Tokenizer configuration schema
# ---------------------------------------------------------------------------

class TokenizerConfig:
    """
    Configuration for tokenizer creation.

    This is a scaffold — the actual config fields will be determined
    when the tokenizer is chosen.
    """

    SUPPORTED_TYPES = ["placeholder", "huggingface", "sentencepiece", "tiktoken"]

    def __init__(self, config: dict):
        self.type = config.get("type", "placeholder")
        self.name_or_path = config.get("name_or_path", "")
        self.vocab_size = config.get("vocab_size", 32000)
        self.extra = {k: v for k, v in config.items() if k not in (
            "type", "name_or_path", "vocab_size"
        )}

        if self.type not in self.SUPPORTED_TYPES:
            raise TokenizerError(
                f"Unknown tokenizer type: {self.type!r}",
                details={"supported": self.SUPPORTED_TYPES},
            )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name_or_path": self.name_or_path,
            "vocab_size": self.vocab_size,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tokenizer(config: Optional[dict | TokenizerConfig] = None) -> Tokenizer:
    """
    Create a tokenizer from configuration.

    Args:
        config: A dict or TokenizerConfig with tokenizer settings.

    Returns:
        A Tokenizer instance.

    Raises:
        TokenizerError: If the tokenizer type is unknown or the
            required library is not available.
    """
    if config is None:
        config = TokenizerConfig({"type": "placeholder"})
    elif isinstance(config, dict):
        config = TokenizerConfig(config)

    if config.type == "placeholder":
        return PlaceholderTokenizer(vocab_size=config.vocab_size)

    elif config.type == "huggingface":
        try:
            from transformers import AutoTokenizer
            if not config.name_or_path:
                raise TokenizerError(
                    "HuggingFace tokenizer requires 'name_or_path'"
                )
            return _HuggingFaceTokenizerWrapper(
                AutoTokenizer.from_pretrained(config.name_or_path)
            )
        except ImportError:
            raise TokenizerError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )

    elif config.type == "sentencepiece":
        raise TokenizerError(
            "SentencePiece tokenizer not yet implemented. "
            "This will be added when the model tokenizer is chosen."
        )

    elif config.type == "tiktoken":
        raise TokenizerError(
            "Tiktoken tokenizer not yet implemented."
        )

    else:
        raise TokenizerError(f"Unknown tokenizer type: {config.type!r}")


# ---------------------------------------------------------------------------
# HuggingFace wrapper
# ---------------------------------------------------------------------------

class _HuggingFaceTokenizerWrapper(Tokenizer):
    """Wraps a HuggingFace tokenizer to conform to our interface."""

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._hf.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self._hf.decode(token_ids, **kwargs)

    def vocab_size(self) -> int:
        return len(self._hf)

    def tokenize(self, text: str) -> list[str]:
        return self._hf.tokenize(text)
