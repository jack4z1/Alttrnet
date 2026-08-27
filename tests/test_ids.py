"""
tests/test_ids.py — Tests for core.ids (deterministic ID generation)
====================================================================
Verifies that IDs are stable, deterministic, and follow the spec.
"""

import pytest

from core.ids import (
    make_chunk_id,
    make_document_id,
    normalize_doc_path,
    url_for_file,
    title_from_markdown,
    FILE_SCHEME,
)


class TestNormalizeDocPath:
    """Test path normalization."""

    def test_simple_relative_path(self):
        assert normalize_doc_path("python/control_flow.md") == "python/control_flow.md"

    def test_backslash_to_forwardslash(self):
        result = normalize_doc_path("python\\control_flow.md")
        assert "\\" not in result
        assert "python/control_flow.md" in result

    def test_with_root(self):
        result = normalize_doc_path("knowledge/python/control_flow.md", "knowledge")
        assert result == "python/control_flow.md"

    def test_strips_leading_slash(self):
        result = normalize_doc_path("./python/test.md")
        assert not result.startswith("./")
        assert not result.startswith("/")

    def test_unknown_fallback(self):
        result = normalize_doc_path("")
        assert result == "unknown"

    def test_deterministic(self):
        """Same path always normalizes to the same result."""
        p = "docs/python/intro.md"
        assert normalize_doc_path(p) == normalize_doc_path(p)


class TestMakeChunkId:
    """Test chunk ID generation."""

    def test_returns_hex_string(self):
        cid = make_chunk_id("test.md", 0)
        assert isinstance(cid, str)
        assert len(cid) == 32  # MD5 hex digest
        int(cid, 16)  # should not raise

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = make_chunk_id("test.md", 0)
        id2 = make_chunk_id("test.md", 0)
        assert id1 == id2

    def test_different_files_different_ids(self):
        id1 = make_chunk_id("file1.md", 0)
        id2 = make_chunk_id("file2.md", 0)
        assert id1 != id2

    def test_different_indices_different_ids(self):
        id1 = make_chunk_id("test.md", 0)
        id2 = make_chunk_id("test.md", 1)
        assert id1 != id2

    def test_case_insensitive_path(self):
        """Paths differ only in case should produce the same ID (lowered)."""
        id1 = make_chunk_id("Test.MD", 0)
        id2 = make_chunk_id("test.md", 0)
        assert id1 == id2

    def test_backslash_equivalence(self):
        """Backslash and forward slash paths should produce the same ID."""
        id1 = make_chunk_id("python\\test.md", 0)
        id2 = make_chunk_id("python/test.md", 0)
        assert id1 == id2


class TestMakeDocumentId:
    """Test document ID generation."""

    def test_returns_hex_string(self):
        did = make_document_id("test.md")
        assert isinstance(did, str)
        assert len(did) == 32
        int(did, 16)

    def test_deterministic(self):
        id1 = make_document_id("test.md")
        id2 = make_document_id("test.md")
        assert id1 == id2

    def test_different_files_different_ids(self):
        id1 = make_document_id("file1.md")
        id2 = make_document_id("file2.md")
        assert id1 != id2

    def test_different_from_chunk_id(self):
        """Document ID should differ from chunk IDs for the same file."""
        did = make_document_id("test.md")
        cid = make_chunk_id("test.md", 0)
        assert did != cid


class TestUrlForFile:
    """Test URL generation for file sources."""

    def test_basic(self):
        assert url_for_file("test.md") == "file://test.md"

    def test_with_path(self):
        assert url_for_file("python/intro.md") == "file://python/intro.md"

    def test_starts_with_scheme(self):
        url = url_for_file("any.md")
        assert url.startswith(FILE_SCHEME)


class TestTitleFromMarkdown:
    """Test markdown title extraction."""

    def test_extracts_h1(self):
        title = title_from_markdown("# My Title\n\nContent", "fallback")
        assert title == "My Title"

    def test_fallback_when_no_h1(self):
        title = title_from_markdown("No heading here", "fallback")
        assert title == "fallback"

    def test_first_h1_wins(self):
        text = "# First\n\n## Sub\n\n# Second"
        title = title_from_markdown(text, "fallback")
        assert title == "First"

    def test_empty_text(self):
        title = title_from_markdown("", "fallback")
        assert title == "fallback"

    def test_none_text(self):
        title = title_from_markdown(None, "fallback")
        assert title == "fallback"

    def test_h1_with_leading_whitespace(self):
        title = title_from_markdown("  # Spaced Title\n\nContent", "fallback")
        assert title == "Spaced Title"
