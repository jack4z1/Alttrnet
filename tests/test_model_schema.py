"""
tests/test_model_schema.py — Tests for core.model_schema
==========================================================
"""



from core.model_schema import ModelMetadata, list_models


class TestModelMetadata:
    def test_creation(self):
        meta = ModelMetadata(name="test_model", params=1_000_000)
        assert meta.name == "test_model"
        assert meta.params == 1_000_000

    def test_to_dict(self):
        meta = ModelMetadata(name="test", params=500, arch_type="dense")
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert "fingerprint" in d

    def test_fingerprint_deterministic(self):
        m1 = ModelMetadata(name="a", arch_type="dense", config={"x": 1})
        m2 = ModelMetadata(name="a", arch_type="dense", config={"x": 1})
        assert m1.fingerprint() == m2.fingerprint()

    def test_fingerprint_differs(self):
        m1 = ModelMetadata(name="a", config={"x": 1})
        m2 = ModelMetadata(name="a", config={"x": 2})
        assert m1.fingerprint() != m2.fingerprint()

    def test_save_and_load(self, tmp_path):
        meta = ModelMetadata(
            name="save_test",
            version="2.0",
            params=5_000_000,
            arch_type="moe",
            config={"num_experts": 8},
            benchmark_scores={"humaneval": 0.45},
        )
        path = tmp_path / "metadata.json"
        meta.save(path)
        assert path.exists()

        loaded = ModelMetadata.load(path)
        assert loaded.name == "save_test"
        assert loaded.version == "2.0"
        assert loaded.params == 5_000_000
        assert loaded.arch_type == "moe"
        assert loaded.config == {"num_experts": 8}
        assert loaded.benchmark_scores["humaneval"] == 0.45

    def test_save_creates_parent_dirs(self, tmp_path):
        meta = ModelMetadata(name="nested")
        path = tmp_path / "deep" / "metadata.json"
        meta.save(path)
        assert path.exists()

    def test_validate_valid(self):
        meta = ModelMetadata(name="valid", arch_type="dense", params=1_000_000)
        errors = meta.validate()
        assert errors == []

    def test_validate_empty_name(self):
        meta = ModelMetadata(name="", arch_type="dense")
        errors = meta.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validate_exceeds_budget(self):
        meta = ModelMetadata(name="too_big", params=20_000_000_000)
        errors = meta.validate()
        assert any("17b" in e.lower() or "exceeds" in e.lower() for e in errors)

    def test_validate_no_arch(self):
        meta = ModelMetadata(name="no_arch")
        errors = meta.validate()
        assert any("architecture" in e.lower() for e in errors)

    def test_param_budget_usage(self):
        meta = ModelMetadata(params=8_500_000_000)
        assert meta.param_budget_usage() == 0.5

    def test_summary(self):
        meta = ModelMetadata(
            name="summary_test",
            arch_type="dense",
            params=17_000_000_000,
            benchmark_scores={"humaneval": 0.5},
        )
        s = meta.summary()
        assert "summary_test" in s
        assert "100.0%" in s  # full budget
        assert "humaneval" in s

    def test_timestamp_set(self):
        meta = ModelMetadata(name="test")
        assert meta.created_at  # non-empty


class TestListModels:
    def test_empty_dir(self, tmp_path):
        assert list_models(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_models(tmp_path / "nonexistent") == []

    def test_finds_models(self, tmp_path):
        for name in ["model_a", "model_b"]:
            meta = ModelMetadata(name=name)
            meta.save(tmp_path / name / "metadata.json")
        models = list_models(tmp_path)
        assert len(models) == 2
        names = [m.name for m in models]
        assert "model_a" in names
        assert "model_b" in names
