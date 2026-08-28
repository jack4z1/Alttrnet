"""
tests/test_cli.py — Tests for the CLI entry points
=====================================================
"""

import subprocess
import sys
from pathlib import Path

CLI_PATH = Path(__file__).resolve().parent.parent / "alttrnet_cli.py"


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "alttrnet" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_info(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "info"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "alttrnet" in result.stdout.lower()

    def test_validate(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "validate"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_env(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "env"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Python" in result.stdout

    def test_runs(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "runs"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_datasets(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "datasets"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_checkpoints(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "checkpoints"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_models(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "models"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "command" in result.stdout.lower()
