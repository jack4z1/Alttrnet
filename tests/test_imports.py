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
        try:
            import core.db
            assert hasattr(core.db, "get_collection")
        except ImportError as e:
            if "cygrpc" in str(e) or "DLL" in str(e):
                pytest.skip("chromadb DLL not available on this system")
            raise

    def test_import_embedder(self):
        import core.embedder
        assert hasattr(core.embedder, "get_embedding")

    def test_import_ids(self):
        import core.ids
        assert hasattr(core.ids, "make_chunk_id")

    def test_import_retriever(self):
        try:
            import core.retriever
            assert hasattr(core.retriever, "Retriever")
        except ImportError as e:
            if "cygrpc" in str(e) or "DLL" in str(e):
                pytest.skip("chromadb DLL not available on this system")
            raise

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

    def test_import_exceptions(self):
        import core.exceptions
        assert hasattr(core.exceptions, "AlttrnetError")

    def test_import_datasets(self):
        import core.datasets
        assert hasattr(core.datasets, "DatasetManifest")

    def test_import_checkpoint(self):
        import core.checkpoint
        assert hasattr(core.checkpoint, "CheckpointMeta")

    def test_import_runs(self):
        import core.runs
        assert hasattr(core.runs, "RunManager")

    def test_import_training_config(self):
        import core.training_config
        assert hasattr(core.training_config, "TrainingConfig")

    def test_import_pipeline(self):
        import core.pipeline
        assert hasattr(core.pipeline, "Pipeline")

    def test_import_tokenizer(self):
        import core.tokenizer
        assert hasattr(core.tokenizer, "PlaceholderTokenizer")

    def test_import_trainer(self):
        import core.trainer
        assert hasattr(core.trainer, "Trainer")

    def test_import_model_schema(self):
        import core.model_schema
        assert hasattr(core.model_schema, "ModelMetadata")

    def test_import_env(self):
        import core.env
        assert hasattr(core.env, "load_env")

    def test_import_config_loader(self):
        import core.config_loader
        assert hasattr(core.config_loader, "load_yaml")

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
        try:
            import chromadb
            assert hasattr(chromadb, "PersistentClient")
        except ImportError as e:
            if "cygrpc" in str(e) or "DLL" in str(e):
                pytest.skip("chromadb DLL not available on this system")
            raise

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
