"""
tests/test_config_loader.py — Tests for core.config_loader
============================================================
"""

import os

import pytest

from core.config_loader import load_training_config, load_yaml, merge_configs, validate_config
from core.exceptions import ConfigError


class TestLoadYaml:
    def test_valid_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key1: value1\nkey2: 42\n")
        data = load_yaml(config_file)
        assert data["key1"] == "value1"
        assert data["key2"] == 42

    def test_nested_yaml(self, tmp_path):
        config_file = tmp_path / "nested.yaml"
        config_file.write_text(
            "section:\n"
            "  nested_key: nested_value\n"
            "  number: 10\n"
        )
        data = load_yaml(config_file)
        assert data["section"]["nested_key"] == "nested_value"
        assert data["section"]["number"] == 10

    def test_nonexistent_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_yaml("/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("key: [unterminated")
        with pytest.raises(ConfigError, match="parse error"):
            load_yaml(config_file)

    def test_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        data = load_yaml(config_file)
        assert data == {}

    def test_list_yaml(self, tmp_path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_yaml(config_file)

    def test_env_interpolation(self, tmp_path):
        os.environ["TEST_INTERP_VAR"] = "interpolated"
        config_file = tmp_path / "env.yaml"
        config_file.write_text("key: ${TEST_INTERP_VAR}\n")
        data = load_yaml(config_file, interpolate_env=True)
        assert data["key"] == "interpolated"
        os.environ.pop("TEST_INTERP_VAR", None)

    def test_env_interpolation_default(self, tmp_path):
        os.environ.pop("NONEXISTENT_ENV_VAR", None)
        config_file = tmp_path / "env_default.yaml"
        config_file.write_text("key: ${NONEXISTENT_ENV_VAR:fallback}\n")
        data = load_yaml(config_file, interpolate_env=True)
        assert data["key"] == "fallback"

    def test_no_interpolation(self, tmp_path):
        os.environ["NO_INTERP"] = "real"
        config_file = tmp_path / "nointerp.yaml"
        config_file.write_text("key: ${NO_INTERP}\n")
        data = load_yaml(config_file, interpolate_env=False)
        assert data["key"] == "${NO_INTERP}"
        os.environ.pop("NO_INTERP", None)

    def test_load_default_config(self):
        """Should be able to load the project's default.yaml."""
        from core.config import PATHS
        default = PATHS.configs / "default.yaml"
        if default.is_file():
            data = load_yaml(default)
            assert "project" in data
            assert data["project"]["name"] == "alttrnet"


class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"section": {"a": 1, "b": 2}}
        override = {"section": {"b": 3, "c": 4}}
        result = merge_configs(base, override)
        assert result["section"] == {"a": 1, "b": 3, "c": 4}

    def test_override_non_dict(self):
        base = {"key": "old"}
        override = {"key": "new"}
        result = merge_configs(base, override)
        assert result["key"] == "new"

    def test_empty_override(self):
        base = {"a": 1}
        result = merge_configs(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        override = {"a": 1}
        result = merge_configs({}, override)
        assert result == {"a": 1}


class TestValidateConfig:
    def test_no_schema(self):
        result = validate_config({"key": "value"})
        assert result == []

    def test_required_keys_present(self):
        schema = {"required": ["name", "version"]}
        data = {"name": "test", "version": "1.0"}
        result = validate_config(data, schema)
        assert result == []

    def test_required_keys_missing(self):
        schema = {"required": ["name", "version"]}
        data = {"name": "test"}
        result = validate_config(data, schema)
        assert len(result) == 1
        assert "version" in result[0]

    def test_allowed_keys(self):
        schema = {"allowed_keys": ["name", "version"]}
        data = {"name": "test", "version": "1.0", "extra": "bad"}
        result = validate_config(data, schema)
        assert len(result) == 1
        assert "extra" in result[0]

    def test_no_unexpected_keys(self):
        schema = {"allowed_keys": ["name", "version"]}
        data = {"name": "test"}
        result = validate_config(data, schema)
        assert result == []


class TestLoadTrainingConfig:
    def test_loads_training_section(self, tmp_path):
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            "training:\n"
            "  max_steps: 1000\n"
            "  seed: 123\n"
            "model:\n"
            "  name: test_model\n"
        )
        data = load_training_config(config_file)
        assert data["max_steps"] == 1000
        assert data["seed"] == 123

    def test_flat_config(self, tmp_path):
        config_file = tmp_path / "flat.yaml"
        config_file.write_text("max_steps: 500\n")
        data = load_training_config(config_file)
        assert data["max_steps"] == 500
