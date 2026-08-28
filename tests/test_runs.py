"""
tests/test_runs.py — Tests for core.runs
==========================================
"""

import json

from core.runs import RunManager


class TestRunManager:
    def test_creation(self, tmp_path):
        RunManager(tmp_path / "experiments")
        assert (tmp_path / "experiments").is_dir()

    def test_create_run(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(
            name="test_run",
            experiment_step="1A",
            config={"lr": 0.001},
        )
        assert run_dir.is_dir()
        assert (run_dir / "metadata.json").is_file()
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "logs").is_dir()
        assert (run_dir / "outputs").is_dir()
        assert (run_dir / "plots").is_dir()

    def test_run_metadata_content(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(
            name="meta_test",
            description="Test run",
            config={"batch_size": 8},
        )
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["name"] == "meta_test"
        assert meta["description"] == "Test run"

    def test_run_config_snapshot(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(
            name="config_test",
            config={"lr": 3e-4, "batch_size": 8},
        )
        config = json.loads((run_dir / "config.json").read_text())
        assert config["lr"] == 3e-4
        assert config["batch_size"] == 8

    def test_run_tags(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(
            name="tagged_run",
            tags=["baseline", "retrieval"],
        )
        tags = json.loads((run_dir / "tags.json").read_text())
        assert "baseline" in tags
        assert "retrieval" in tags

    def test_list_runs(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        rm.create_run(name="run_a")
        rm.create_run(name="run_b")
        runs = rm.list_runs()
        assert len(runs) == 2
        names = [r["name"] for r in runs]
        assert "run_a" in names
        assert "run_b" in names

    def test_list_runs_name_filter(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        rm.create_run(name="retrieval_v1")
        rm.create_run(name="retrieval_v2")
        rm.create_run(name="training_v1")
        runs = rm.list_runs(name_filter="retrieval")
        assert len(runs) == 2

    def test_list_runs_empty(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        assert rm.list_runs() == []

    def test_get_run(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        rm.create_run(name="find_me")
        found = rm.get_run("find_me")
        assert found is not None
        assert found.is_dir()

    def test_get_run_not_found(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        assert rm.get_run("nonexistent") is None

    def test_update_status(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(name="status_test")
        rm.update_status(run_dir, "running")
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["status"] == "running"

    def test_add_result(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(name="result_test")
        rm.add_result(run_dir, "accuracy", 0.95)
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["results"]["accuracy"] == 0.95

    def test_get_log_dir(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(name="log_test")
        log_dir = rm.get_log_dir(run_dir)
        assert log_dir.name == "logs"
        assert log_dir.is_dir()

    def test_get_output_dir(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        run_dir = rm.create_run(name="output_test")
        output_dir = rm.get_output_dir(run_dir)
        assert output_dir.name == "outputs"
        assert output_dir.is_dir()

    def test_summary(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        rm.create_run(name="run_1")
        rm.create_run(name="run_2")
        s = rm.summary()
        assert "2" in s
        assert "run_1" in s

    def test_summary_empty(self, tmp_path):
        rm = RunManager(tmp_path / "experiments")
        s = rm.summary()
        assert "No" in s
