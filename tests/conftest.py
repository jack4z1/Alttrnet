"""
tests/conftest.py — Shared pytest fixtures and configuration
=============================================================
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path so `core.*` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_text():
    """A sample text for chunking tests (~500 words)."""
    return " ".join(f"word{i}" for i in range(500))


@pytest.fixture
def short_text():
    """A short text (~50 words) for edge case testing."""
    return " ".join(f"short{i}" for i in range(50))


@pytest.fixture
def empty_text():
    """Empty string for edge case testing."""
    return ""


@pytest.fixture
def sample_markdown():
    """Sample markdown content for ID/title tests."""
    return """# My Test Document

This is a test document with some content.

## Section 1

More content here with important information.

## Section 2

Final section with details.
"""
