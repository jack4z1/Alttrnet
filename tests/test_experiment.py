"""
tests/test_experiment.py — Tests for core.experiment (experiment metadata)
==========================================================================
"""



from core.experiment import ExperimentMeta


class TestExperimentMeta:
    def test_creation(self):
        exp = ExperimentMeta(name="test_exp", step="1A")
        assert exp.name == "test_exp"
        assert exp.step == "1A"
        assert exp.status == "pending"
        assert exp.results == {}
        assert exp.config == {}

    def test_add_result(self):
        exp = ExperimentMeta(name="test")
        exp.add_result("accuracy", 0.95)
        assert exp.results["accuracy"] == 0.95

    def test_set_status(self):
        exp = ExperimentMeta(name="test")
        exp.set_status("completed")
        assert exp.status == "completed"

    def test_fingerprint_deterministic(self):
        exp1 = ExperimentMeta(name="test", config={"a": 1}, results={"b": 2})
        exp2 = ExperimentMeta(name="test", config={"a": 1}, results={"b": 2})
        assert exp1.fingerprint() == exp2.fingerprint()

    def test_fingerprint_differs_on_change(self):
        exp1 = ExperimentMeta(name="test", results={"b": 2})
        exp2 = ExperimentMeta(name="test", results={"b": 3})
        assert exp1.fingerprint() != exp2.fingerprint()

    def test_to_dict(self):
        exp = ExperimentMeta(name="test", step="1A")
        d = exp.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["step"] == "1A"
        assert "fingerprint" in d

    def test_save_and_load(self, tmp_path):
        exp = ExperimentMeta(
            name="save_test",
            step="1B",
            config={"param": 42},
            results={"metric": 0.88},
        )
        exp.set_status("completed")

        path = tmp_path / "experiment.json"
        exp.save(path)
        assert path.exists()

        loaded = ExperimentMeta.load(path)
        assert loaded.name == "save_test"
        assert loaded.step == "1B"
        assert loaded.config == {"param": 42}
        assert loaded.results == {"metric": 0.88}
        assert loaded.status == "completed"

    def test_save_creates_parent_dirs(self, tmp_path):
        exp = ExperimentMeta(name="nested")
        path = tmp_path / "deep" / "nested" / "experiment.json"
        exp.save(path)
        assert path.exists()

    def test_timestamp_set(self):
        exp = ExperimentMeta(name="test")
        assert exp.created_at  # non-empty
