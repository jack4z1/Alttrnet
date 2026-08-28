"""
tests/test_checkpoint.py — Tests for core.checkpoint
======================================================
"""


import pytest

from core.checkpoint import (
    CheckpointMeta,
    find_latest_checkpoint,
    list_checkpoints,
    verify_checkpoint,
)


class TestCheckpointMeta:
    def test_creation(self):
        ckpt = CheckpointMeta(name="test_ckpt")
        assert ckpt.name == "test_ckpt"
        assert ckpt.epoch == 0
        assert ckpt.global_step == 0
        assert not hasattr(ckpt, "status")  # checkpoint uses ExperimentMeta for status

    def test_to_dict(self):
        ckpt = CheckpointMeta(name="test", config={"lr": 0.001})
        d = ckpt.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert "fingerprint" in d

    def test_fingerprint_deterministic(self):
        c1 = CheckpointMeta(name="a", config={"x": 1}, metrics={"loss": 2.0})
        c2 = CheckpointMeta(name="a", config={"x": 1}, metrics={"loss": 2.0})
        assert c1.fingerprint() == c2.fingerprint()

    def test_fingerprint_differs_on_change(self):
        c1 = CheckpointMeta(name="a", metrics={"loss": 1.0})
        c2 = CheckpointMeta(name="a", metrics={"loss": 2.0})
        assert c1.fingerprint() != c2.fingerprint()

    def test_save_and_load(self, tmp_path):
        ckpt = CheckpointMeta(
            name="save_test",
            step="2A.1",
            epoch=5,
            global_step=1000,
            config={"lr": 3e-4},
            metrics={"train_loss": 2.34},
        )
        ckpt_dir = tmp_path / "my_checkpoint"
        ckpt.save(ckpt_dir)
        assert (ckpt_dir / "metadata.json").exists()

        loaded = CheckpointMeta.load(ckpt_dir)
        assert loaded.name == "save_test"
        assert loaded.epoch == 5
        assert loaded.global_step == 1000
        assert loaded.config == {"lr": 3e-4}
        assert loaded.metrics == {"train_loss": 2.34}

    def test_save_creates_parent_dirs(self, tmp_path):
        ckpt = CheckpointMeta(name="nested")
        ckpt_dir = tmp_path / "deep" / "nested" / "checkpoint"
        ckpt.save(ckpt_dir)
        assert (ckpt_dir / "metadata.json").exists()

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(Exception):
            CheckpointMeta.load(tmp_path / "nonexistent")

    def test_validate_valid(self):
        ckpt = CheckpointMeta(name="valid", epoch=1, global_step=100)
        errors = ckpt.validate()
        assert errors == []

    def test_validate_empty_name(self):
        ckpt = CheckpointMeta(name="")
        errors = ckpt.validate()
        assert len(errors) > 0
        assert "name" in errors[0].lower()

    def test_validate_negative_step(self):
        ckpt = CheckpointMeta(name="test", global_step=-1)
        errors = ckpt.validate()
        assert any("negative" in e.lower() for e in errors)

    def test_validate_step_exceeds_total(self):
        ckpt = CheckpointMeta(name="test", global_step=200, total_steps=100)
        errors = ckpt.validate()
        assert any("exceeds" in e.lower() or ">" in e for e in errors)

    def test_summary(self):
        ckpt = CheckpointMeta(
            name="summary_test",
            epoch=3,
            global_step=500,
            total_steps=1000,
            metrics={"loss": 1.5},
        )
        s = ckpt.summary()
        assert "summary_test" in s
        assert "50.0%" in s

    def test_timestamp_set(self):
        ckpt = CheckpointMeta(name="test")
        assert ckpt.created_at  # non-empty


class TestListCheckpoints:
    def test_empty_dir(self, tmp_path):
        assert list_checkpoints(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_checkpoints(tmp_path / "nonexistent") == []

    def test_finds_checkpoints(self, tmp_path):
        for name in ["ckpt_a", "ckpt_b"]:
            ckpt = CheckpointMeta(name=name)
            ckpt.save(tmp_path / name)
        checkpoints = list_checkpoints(tmp_path)
        assert len(checkpoints) == 2
        names = [c.name for c in checkpoints]
        assert "ckpt_a" in names
        assert "ckpt_b" in names


class TestFindLatestCheckpoint:
    def test_empty(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_finds_latest(self, tmp_path):
        import time
        ckpt1 = CheckpointMeta(name="first", epoch=1)
        ckpt1.save(tmp_path / "first")
        time.sleep(0.1)
        ckpt2 = CheckpointMeta(name="second", epoch=2)
        ckpt2.save(tmp_path / "second")
        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.name == "second"


class TestVerifyCheckpoint:
    def test_valid(self, tmp_path):
        ckpt = CheckpointMeta(name="valid")
        ckpt_dir = tmp_path / "valid"
        ckpt.save(ckpt_dir)
        result = verify_checkpoint(ckpt_dir)
        assert result["valid"]
        assert len(result["errors"]) == 0
        assert "metadata.json" in result["files_found"]

    def test_missing_dir(self, tmp_path):
        result = verify_checkpoint(tmp_path / "nonexistent")
        assert not result["valid"]
        assert "does not exist" in result["errors"][0].lower()

    def test_missing_metadata(self, tmp_path):
        ckpt_dir = tmp_path / "no_meta"
        ckpt_dir.mkdir()
        result = verify_checkpoint(ckpt_dir)
        assert not result["valid"]
        assert any("metadata" in e.lower() for e in result["errors"])
