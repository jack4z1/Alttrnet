"""
tests/test_imports.py — Smoke tests for all module imports
==========================================================
Verifies that all modules can be imported without error.
This catches circular imports, missing dependencies, and syntax errors.
"""

import pytest


class TestCoreImports:
    """Verify core package imports cleanly."""

    def test_import_config(self):
        import core.config
        assert hasattr(core.config, "PROJECT")

    def test_import_chunker(self):
        import core.chunker
        assert hasattr(core.chunker, "chunk_text")

    def test_import_db(self):
        import core.db
        assert hasattr(core.db, "get_collection")

    def test_import_embedder(self):
        import core.embedder
        assert hasattr(core.embedder, "get_embedding")

    def test_import_ids(self):
        import core.ids
        assert hasattr(core.ids, "make_chunk_id")

    def test_import_retriever(self):
        import core.retriever
        assert hasattr(core.retriever, "Retriever")

    def test_import_seeds(self):
        import core.seeds
        assert hasattr(core.seeds, "set_global_seed")

    def test_import_logging(self):
        import core.logging
        assert hasattr(core.logging, "get_logger")

    def test_import_experiment(self):
        import core.experiment
        assert hasattr(core.experiment, "ExperimentMeta")

    def test_import_eval_format(self):
        import core.eval_format
        assert hasattr(core.eval_format, "EvalResult")

    def test_import_core_package(self):
        """Verify the core package __init__ exposes expected names."""
        import core
        assert hasattr(core, "PROJECT")
        assert hasattr(core, "PATHS")
        assert hasattr(core, "MODELS")
        assert hasattr(core, "CHUNKING")
        assert hasattr(core, "RETRIEVAL")
        assert hasattr(core, "set_global_seed")
        assert hasattr(core, "get_logger")
        assert hasattr(core, "ExperimentMeta")
        assert hasattr(core, "EvalResult")


class TestExternalDependencies:
    """Verify critical external dependencies are available."""

    def test_chromadb(self):
        import chromadb
        assert hasattr(chromadb, "PersistentClient")

    def test_ollama(self):
        import ollama
        assert hasattr(ollama, "embeddings")

    def test_rank_bm25(self):
        from rank_bm25 import BM25Okapi
        assert BM25Okapi is not None

    def test_torch(self):
        import torch
        assert hasattr(torch, "cuda")

    def test_numpy(self):
        import numpy
        assert hasattr(numpy, "random")
