"""
tests/test_env.py — Tests for core.env
========================================
"""

import os
from pathlib import Path

import pytest

from core.env import get_env, get_env_bool, get_env_int, load_env, require_env
from core.exceptions import EnvError


class TestLoadEnv:
    def test_load_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_1=hello\nTEST_VAR_2=world\n")
        loaded = load_env(env_file)
        assert "TEST_VAR_1" in loaded
        assert loaded["TEST_VAR_1"] == "hello"
        assert loaded["TEST_VAR_2"] == "world"
        # Cleanup
        os.environ.pop("TEST_VAR_1", None)
        os.environ.pop("TEST_VAR_2", None)

    def test_load_sets_env_vars(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_ENV_SET=value123\n")
        load_env(env_file)
        assert os.environ.get("TEST_ENV_SET") == "value123"
        os.environ.pop("TEST_ENV_SET", None)

    def test_load_nonexistent_default(self):
        """When path is explicitly nonexistent, should raise EnvError."""
        # This tests the explicit path behavior
        with pytest.raises(EnvError, match="not found"):
            load_env(Path("/nonexistent/.env"))

    def test_load_with_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hello world"\nSINGLE=\'test\'\n')
        loaded = load_env(env_file)
        assert loaded["QUOTED"] == "hello world"
        assert loaded["SINGLE"] == "test"
        os.environ.pop("QUOTED", None)
        os.environ.pop("SINGLE", None)

    def test_load_skips_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nREAL=value\n# another comment\n")
        loaded = load_env(env_file)
        assert loaded == {"REAL": "value"}
        os.environ.pop("REAL", None)

    def test_load_skips_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nREAL=value\n\n\n")
        loaded = load_env(env_file)
        assert loaded == {"REAL": "value"}
        os.environ.pop("REAL", None)

    def test_load_no_override(self, tmp_path):
        os.environ["NO_OVERRIDE_VAR"] = "original"
        env_file = tmp_path / ".env"
        env_file.write_text("NO_OVERRIDE_VAR=new_value\n")
        load_env(env_file, override=False)
        assert os.environ["NO_OVERRIDE_VAR"] == "original"
        os.environ.pop("NO_OVERRIDE_VAR", None)

    def test_load_with_override(self, tmp_path):
        os.environ["OVERRIDE_VAR"] = "original"
        env_file = tmp_path / ".env"
        env_file.write_text("OVERRIDE_VAR=new_value\n")
        load_env(env_file, override=True)
        assert os.environ["OVERRIDE_VAR"] == "new_value"
        os.environ.pop("OVERRIDE_VAR", None)

    def test_load_nonexistent_explicit_raises(self, tmp_path):
        with pytest.raises(EnvError, match="not found"):
            load_env(tmp_path / "nonexistent.env")

    def test_load_empty_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        loaded = load_env(env_file)
        assert loaded == {}


class TestRequireEnv:
    def test_all_present(self):
        os.environ["TEST_REQ_A"] = "val_a"
        os.environ["TEST_REQ_B"] = "val_b"
        result = require_env("TEST_REQ_A", "TEST_REQ_B")
        assert result == {"TEST_REQ_A": "val_a", "TEST_REQ_B": "val_b"}
        os.environ.pop("TEST_REQ_A", None)
        os.environ.pop("TEST_REQ_B", None)

    def test_missing_raises(self):
        os.environ.pop("TEST_REQ_MISSING_X", None)
        with pytest.raises(EnvError, match="Missing"):
            require_env("TEST_REQ_MISSING_X")

    def test_with_caller(self):
        os.environ.pop("TEST_REQ_MISSING_Y", None)
        with pytest.raises(EnvError, match="test_func"):
            require_env("TEST_REQ_MISSING_Y", _caller="test_func")

    def test_empty_string_is_missing(self):
        os.environ["EMPTY_VAR"] = ""
        with pytest.raises(EnvError):
            require_env("EMPTY_VAR")
        os.environ.pop("EMPTY_VAR", None)


class TestGetEnv:
    def test_existing(self):
        os.environ["GET_ENV_TEST"] = "hello"
        assert get_env("GET_ENV_TEST") == "hello"
        os.environ.pop("GET_ENV_TEST", None)

    def test_missing_default(self):
        os.environ.pop("GET_ENV_MISSING", None)
        assert get_env("GET_ENV_MISSING") == ""

    def test_missing_custom_default(self):
        os.environ.pop("GET_ENV_MISSING2", None)
        assert get_env("GET_ENV_MISSING2", "fallback") == "fallback"


class TestGetEnvInt:
    def test_valid(self):
        os.environ["INT_VAR"] = "42"
        assert get_env_int("INT_VAR") == 42
        os.environ.pop("INT_VAR", None)

    def test_missing_default(self):
        os.environ.pop("INT_MISSING", None)
        assert get_env_int("INT_MISSING", 10) == 10

    def test_invalid_raises(self):
        os.environ["BAD_INT"] = "not_a_number"
        with pytest.raises(Exception):
            get_env_int("BAD_INT")
        os.environ.pop("BAD_INT", None)


class TestGetEnvBool:
    def test_truthy_values(self):
        for val in ["1", "true", "yes", "on", "True", "YES"]:
            os.environ["BOOL_VAR"] = val
            assert get_env_bool("BOOL_VAR") is True
        os.environ.pop("BOOL_VAR", None)

    def test_falsy_values(self):
        for val in ["0", "false", "no", "off", "random"]:
            os.environ["BOOL_VAR"] = val
            assert get_env_bool("BOOL_VAR") is False
        os.environ.pop("BOOL_VAR", None)

    def test_missing_default(self):
        os.environ.pop("BOOL_MISSING", None)
        assert get_env_bool("BOOL_MISSING") is False
        assert get_env_bool("BOOL_MISSING", True) is True
