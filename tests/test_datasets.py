"""
tests/test_datasets.py — Tests for core.datasets
==================================================
"""



from core.datasets import (
    DatasetManifest,
    DatasetRegistry,
    DatasetValidator,
    ValidationReport,
    create_dataset_dir,
)


class TestDatasetManifest:
    def test_creation(self):
        m = DatasetManifest(name="test_ds", version="1.0")
        assert m.name == "test_ds"
        assert m.version == "1.0"
        assert m.format == "jsonl"
        assert not m.validated

    def test_to_dict(self):
        m = DatasetManifest(name="test", num_samples=100)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["num_samples"] == 100

    def test_save_and_load(self, tmp_path):
        m = DatasetManifest(
            name="save_test",
            version="2.0",
            num_samples=500,
            fields={"text": "string", "label": "int"},
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        assert path.exists()

        loaded = DatasetManifest.load(path)
        assert loaded.name == "save_test"
        assert loaded.version == "2.0"
        assert loaded.num_samples == 500
        assert loaded.fields == {"text": "string", "label": "int"}

    def test_save_creates_parent_dirs(self, tmp_path):
        m = DatasetManifest(name="nested")
        path = tmp_path / "deep" / "nested" / "manifest.json"
        m.save(path)
        assert path.exists()

    def test_summary(self):
        m = DatasetManifest(name="summary_test", num_samples=100, format="csv")
        s = m.summary()
        assert "summary_test" in s
        assert "csv" in s
        assert "100" in s

    def test_compute_checksum(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        m = DatasetManifest(name="test")
        checksum = m.compute_checksum(test_file)
        assert len(checksum) == 64  # SHA-256 hex digest
        assert m.checksum_sha256 == checksum

    def test_fields_default(self):
        m = DatasetManifest(name="test")
        assert m.fields == {}
        assert m.tags == []

    def test_timestamp_set(self):
        m = DatasetManifest(name="test")
        assert m.created_at  # non-empty


class TestDatasetRegistry:
    def test_creation(self, tmp_path):
        reg = DatasetRegistry(tmp_path / "datasets")
        assert reg.list_datasets() == []

    def test_register_and_get(self, tmp_path):
        reg = DatasetRegistry(tmp_path / "datasets")
        m = DatasetManifest(name="my_ds", version="1.0")
        manifest_path = tmp_path / "my_ds" / "manifest.json"
        m.save(manifest_path)

        reg.register(m, manifest_path)
        assert "my_ds" in reg.list_datasets()

        loaded = reg.get("my_ds")
        assert loaded is not None
        assert loaded.name == "my_ds"

    def test_get_nonexistent(self, tmp_path):
        reg = DatasetRegistry(tmp_path / "datasets")
        assert reg.get("nonexistent") is None

    def test_remove(self, tmp_path):
        reg = DatasetRegistry(tmp_path / "datasets")
        m = DatasetManifest(name="to_remove")
        manifest_path = tmp_path / "manifest.json"
        m.save(manifest_path)
        reg.register(m, manifest_path)

        assert reg.remove("to_remove")
        assert "to_remove" not in reg.list_datasets()
        assert reg.get("to_remove") is None

    def test_remove_nonexistent(self, tmp_path):
        reg = DatasetRegistry(tmp_path / "datasets")
        assert not reg.remove("nonexistent")

    def test_persistence(self, tmp_path):
        """Registry persists across instances."""
        reg1 = DatasetRegistry(tmp_path / "datasets")
        m = DatasetManifest(name="persistent")
        mp = tmp_path / "manifest.json"
        m.save(mp)
        reg1.register(m, mp)

        reg2 = DatasetRegistry(tmp_path / "datasets")
        assert "persistent" in reg2.list_datasets()


class TestValidationReport:
    def test_valid_by_default(self):
        r = ValidationReport()
        assert r.is_valid

    def test_add_error(self):
        r = ValidationReport()
        r.add_error("something wrong")
        assert not r.is_valid
        assert len(r.errors) == 1

    def test_add_warning(self):
        r = ValidationReport()
        r.add_warning("just a warning")
        assert r.is_valid  # warnings don't make it invalid
        assert len(r.warnings) == 1

    def test_summary(self):
        r = ValidationReport()
        r.add_error("err1")
        r.add_warning("warn1")
        r.stats["count"] = 42
        s = r.summary()
        assert "FAIL" in s
        assert "err1" in s
        assert "warn1" in s


class TestDatasetValidator:
    def test_missing_file(self):
        m = DatasetManifest(name="test", format="jsonl")
        v = DatasetValidator(m)
        report = v.validate("/nonexistent/file.jsonl")
        assert not report.is_valid
        assert "not found" in report.errors[0].lower()

    def test_empty_file(self, tmp_path):
        m = DatasetManifest(name="test", format="jsonl")
        v = DatasetValidator(m)
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        report = v.validate(f)
        assert not report.is_valid
        assert "empty" in report.errors[0].lower()

    def test_valid_jsonl(self, tmp_path):
        m = DatasetManifest(
            name="test",
            format="jsonl",
            num_samples=3,
            fields={"text": "string", "label": "int"},
        )
        v = DatasetValidator(m)
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"text": "hello", "label": 1}\n'
            '{"text": "world", "label": 0}\n'
            '{"text": "test", "label": 1}\n'
        )
        report = v.validate(f)
        assert report.is_valid
        assert report.stats["sample_count"] == 3

    def test_jsonl_parse_errors(self, tmp_path):
        m = DatasetManifest(name="test", format="jsonl")
        v = DatasetValidator(m)
        f = tmp_path / "bad.jsonl"
        f.write_text('{"ok": true}\nnot json\n{"ok": true}\n')
        report = v.validate(f)
        assert not report.is_valid
        assert any("parse error" in e.lower() for e in report.errors)

    def test_jsonl_missing_fields(self, tmp_path):
        m = DatasetManifest(
            name="test",
            format="jsonl",
            fields={"required_field": "string"},
        )
        v = DatasetValidator(m)
        f = tmp_path / "missing.jsonl"
        f.write_text('{"other_field": "value"}\n')
        report = v.validate(f)
        assert not report.is_valid
        assert any("missing" in e.lower() for e in report.errors)

    def test_jsonl_sample_count_mismatch(self, tmp_path):
        m = DatasetManifest(
            name="test",
            format="jsonl",
            num_samples=10,
        )
        v = DatasetValidator(m)
        f = tmp_path / "small.jsonl"
        f.write_text('{"a": 1}\n')
        report = v.validate(f)
        # Should be valid structurally but warn about count
        assert report.is_valid
        assert any("mismatch" in w.lower() for w in report.warnings)

    def test_checksum_mismatch(self, tmp_path):
        m = DatasetManifest(
            name="test",
            format="jsonl",
            checksum_sha256="0" * 64,  # wrong checksum
        )
        v = DatasetValidator(m)
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n')
        report = v.validate(f)
        assert not report.is_valid
        assert any("checksum" in e.lower() for e in report.errors)

    def test_text_validation(self, tmp_path):
        m = DatasetManifest(name="test", format="text")
        v = DatasetValidator(m)
        f = tmp_path / "data.txt"
        f.write_text("line 1\nline 2\n\nline 4\n")
        report = v.validate(f)
        assert report.is_valid
        assert report.stats["line_count"] == 4
        assert report.stats["non_empty_lines"] == 3

    def test_unknown_format_warns(self, tmp_path):
        m = DatasetManifest(name="test", format="parquet")
        v = DatasetValidator(m)
        f = tmp_path / "data.parquet"
        f.write_bytes(b"\x00" * 10)
        report = v.validate(f)
        assert report.is_valid  # no error, just warning
        assert any("no validator" in w.lower() for w in report.warnings)


class TestCreateDatasetDir:
    def test_creates_directory(self, tmp_path):
        ds_dir = create_dataset_dir(tmp_path, "my_dataset", "1.0")
        assert ds_dir.is_dir()
        assert ds_dir.name == "v1.0"

    def test_creates_nested(self, tmp_path):
        ds_dir = create_dataset_dir(tmp_path, "nested/dataset", "2.0")
        assert ds_dir.is_dir()
