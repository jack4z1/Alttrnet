"""
tests/test_chunker.py — Tests for core.chunker (FROZEN module)
==============================================================
These tests verify the chunking algorithm is stable and correct.
Any failure here indicates a breaking change to a frozen component.
"""

import pytest

from core.chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STEP


class TestChunkConstants:
    """Verify frozen constants are correct."""

    def test_chunk_size(self):
        assert CHUNK_SIZE == 400

    def test_chunk_overlap(self):
        assert CHUNK_OVERLAP == 50

    def test_chunk_step(self):
        assert CHUNK_STEP == 350

    def test_step_computed_correctly(self):
        assert CHUNK_STEP == CHUNK_SIZE - CHUNK_OVERLAP


class TestChunkText:
    """Test the chunk_text function."""

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_none_input(self):
        assert chunk_text(None) == []

    def test_whitespace_only(self):
        assert chunk_text("   \t\n  ") == []

    def test_single_word(self):
        result = chunk_text("hello")
        assert len(result) == 1
        assert result[0] == "hello"

    def test_exact_chunk_size(self):
        """Exactly CHUNK_SIZE words should produce one chunk."""
        text = " ".join(f"w{i}" for i in range(CHUNK_SIZE))
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_two_chunks(self):
        """More than CHUNK_SIZE words should produce 2 chunks."""
        text = " ".join(f"w{i}" for i in range(CHUNK_SIZE + 10))
        result = chunk_text(text)
        assert len(result) == 2

    def test_overlap_preserved(self):
        """Second chunk should start with the last CHUNK_OVERLAP words of the first."""
        text = " ".join(f"w{i}" for i in range(CHUNK_SIZE + 100))
        result = chunk_text(text)
        # The overlap region: last 50 words of chunk 0 = first 50 words of chunk 1
        words_chunk0 = result[0].split()
        words_chunk1 = result[1].split()
        assert words_chunk0[-CHUNK_OVERLAP:] == words_chunk1[:CHUNK_OVERLAP]

    def test_deterministic(self):
        """Same input always produces the same output."""
        text = " ".join(f"word{i}" for i in range(200))
        result1 = chunk_text(text)
        result2 = chunk_text(text)
        assert result1 == result2

    def test_large_text(self):
        """Chunk a 5000-word text and verify structure."""
        text = " ".join(f"word{i}" for i in range(5000))
        result = chunk_text(text)
        # Should have ceil((5000 - 400) / 350) + 1 = ceil(4600/350) + 1 = 14 + 1 = 15 chunks
        assert len(result) > 10
        # Each chunk should have at most CHUNK_SIZE words
        for chunk in result:
            assert len(chunk.split()) <= CHUNK_SIZE

    def test_custom_parameters(self):
        """Custom chunk_size and overlap should work."""
        text = " ".join(f"w{i}" for i in range(100))
        result = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.split()) <= 30

    def test_words_preserved(self):
        """All original words should appear in the output (possibly repeated)."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        result = chunk_text(text)
        all_output_words = []
        for chunk in result:
            all_output_words.extend(chunk.split())
        # Every original word should appear at least once
        for w in words:
            assert w in all_output_words
