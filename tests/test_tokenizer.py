"""
tests/test_tokenizer.py — Tests for core.tokenizer
====================================================
"""

import pytest

from core.exceptions import TokenizerError
from core.tokenizer import (
    PlaceholderTokenizer,
    TokenizerConfig,
    create_tokenizer,
)


class TestPlaceholderTokenizer:
    def test_encode_returns_list(self):
        tok = PlaceholderTokenizer()
        result = tok.encode("hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_decode_returns_string(self):
        tok = PlaceholderTokenizer()
        encoded = tok.encode("test")
        decoded = tok.decode(encoded)
        assert decoded == "test"

    def test_roundtrip(self):
        tok = PlaceholderTokenizer()
        text = "Hello, World!"
        assert tok.decode(tok.encode(text)) == text

    def test_vocab_size(self):
        tok = PlaceholderTokenizer(vocab_size=50)
        assert tok.vocab_size() == 50

    def test_empty_string(self):
        tok = PlaceholderTokenizer()
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_callable(self):
        tok = PlaceholderTokenizer()
        result = tok("test")
        assert isinstance(result, list)

    def test_modular_arithmetic(self):
        """Characters map to ord(c) % vocab_size."""
        tok = PlaceholderTokenizer(vocab_size=10)
        encoded = tok.encode("a")  # ord('a') = 97, 97 % 10 = 7
        assert encoded == [7]


class TestTokenizerConfig:
    def test_defaults(self):
        tc = TokenizerConfig({})
        assert tc.type == "placeholder"
        assert tc.vocab_size == 32000

    def test_huggingface_type(self):
        tc = TokenizerConfig({"type": "huggingface", "name_or_path": "gpt2"})
        assert tc.type == "huggingface"
        assert tc.name_or_path == "gpt2"

    def test_invalid_type(self):
        with pytest.raises(TokenizerError, match="Unknown tokenizer type"):
            TokenizerConfig({"type": "nonexistent"})

    def test_to_dict(self):
        tc = TokenizerConfig({"type": "placeholder", "vocab_size": 100})
        d = tc.to_dict()
        assert d["type"] == "placeholder"
        assert d["vocab_size"] == 100


class TestCreateTokenizer:
    def test_default_creates_placeholder(self):
        tok = create_tokenizer()
        assert isinstance(tok, PlaceholderTokenizer)

    def test_placeholder_explicit(self):
        tok = create_tokenizer({"type": "placeholder", "vocab_size": 50})
        assert isinstance(tok, PlaceholderTokenizer)
        assert tok.vocab_size() == 50

    def test_unknown_type_raises(self):
        with pytest.raises(TokenizerError, match="Unknown tokenizer type"):
            create_tokenizer({"type": "nonexistent"})

    def test_huggingface_missing_library(self):
        """If transformers is not installed, should raise TokenizerError."""
        import importlib
        if importlib.util.find_spec("transformers") is not None:
            pytest.skip("transformers is installed, cannot test missing case")
        with pytest.raises(TokenizerError, match="not installed"):
            create_tokenizer({"type": "huggingface", "name_or_path": "gpt2"})

    def test_huggingface_no_path(self):
        """HuggingFace type without name_or_path should raise."""
        import importlib
        if importlib.util.find_spec("transformers") is None:
            pytest.skip("transformers not installed")

        with pytest.raises(TokenizerError, match="name_or_path"):
            create_tokenizer({"type": "huggingface"})

    def test_sentencepiece_not_implemented(self):
        with pytest.raises(TokenizerError, match="not yet implemented"):
            create_tokenizer({"type": "sentencepiece"})

    def test_tiktoken_not_implemented(self):
        with pytest.raises(TokenizerError, match="not yet implemented"):
            create_tokenizer({"type": "tiktoken"})
