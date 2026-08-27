"""
tests/test_config.py — Tests for core.config (central configuration)
====================================================================
"""

import pytest
from pathlib import Path

from core.config import PROJECT, PATHS, MODELS, CHUNKING, RETRIEVAL, CHROMA, ENV


class TestProjectConfig:
    def test_name(self):
        assert PROJECT.name == "alttrnet"

    def test_version(self):
        assert isinstance(PROJECT.version, str)

    def test_max_parameters(self):
        assert PROJECT.max_parameters == 17_000_000_000

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PROJECT.name = "changed"


class TestPathConfig:
    def test_root_exists(self):
        assert PATHS.root.is_dir()

    def test_root_contains_core(self):
        assert (PATHS.root / "core").is_dir()

    def test_chroma_db_path(self):
        assert PATHS.chroma_db.name == "chroma_db"

    def test_knowledge_path(self):
        assert PATHS.knowledge.name == "knowledge"

    def test_configs_path(self):
        assert PATHS.configs.name == "configs"

    def test_experiments_path(self):
        assert PATHS.experiments.name == "experiments"

    def test_scripts_path(self):
        assert PATHS.scripts.name == "scripts"

    def test_tests_path(self):
        assert PATHS.tests.name == "tests"

    def test_evaluation_path(self):
        assert PATHS.evaluation.name == "evaluation"

    def test_docs_path(self):
        assert PATHS.docs.name == "docs"

    def test_checkpoints_path(self):
        assert PATHS.checkpoints.name == "checkpoints"

    def test_artifacts_path(self):
        assert PATHS.artifacts.name == "artifacts"


class TestModelConfig:
    def test_embed_model(self):
        assert MODELS.embed == "nomic-embed-text"

    def test_llm_model(self):
        assert MODELS.llm == "qwen3:14b"

    def test_reranker_model(self):
        assert "ms-marco" in MODELS.reranker

    def test_frozen(self):
        with pytest.raises(AttributeError):
            MODELS.embed = "changed"


class TestChunkingConfig:
    def test_size(self):
        assert CHUNKING.size == 400

    def test_overlap(self):
        assert CHUNKING.overlap == 50

    def test_step(self):
        assert CHUNKING.effective_step == 350


class TestRetrievalConfig:
    def test_dense_top_k(self):
        assert RETRIEVAL.dense_top_k == 20

    def test_bm25_top_k(self):
        assert RETRIEVAL.bm25_top_k == 20

    def test_final_top_k(self):
        assert RETRIEVAL.final_top_k == 5

    def test_rrf_k(self):
        assert RETRIEVAL.rrf_k == 60


class TestChromaConfig:
    def test_collection_name(self):
        assert CHROMA.collection_name == "knowledge_base"

    def test_space(self):
        assert CHROMA.space == "cosine"


class TestEnvConfig:
    def test_python_min(self):
        assert ENV.python_min == "3.10"

    def test_required_models(self):
        assert "nomic-embed-text" in ENV.required_ollama_models
        assert "qwen3:14b" in ENV.required_ollama_models
