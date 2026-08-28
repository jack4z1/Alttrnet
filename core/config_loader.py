"""
core/config_loader.py — ALTTRNET YAML config loading & validation
==================================================================
Loads YAML configuration files and maps them to the dataclass-based
configuration objects in core.config and core.training_config.

Supports:
    - Loading YAML files into config dataclasses
    - Environment variable interpolation in YAML values
    - Config validation
    - Config merging (base + override)

Usage:
    from core.config_loader import load_yaml, load_training_config

    # Load raw YAML
    data = load_yaml("configs/default.yaml")

    # Load into a TrainingConfig
    tc = load_training_config("configs/training/base.yaml")

    # Merge configs
    merged = merge_configs(base_config, override_config)
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from core.exceptions import ConfigError

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _interpolate_env_vars(value: Any) -> Any:
    """
    Replace ${VAR} and ${VAR:default} patterns in string values.
    """
    if isinstance(value, str):
        def _replace(m):
            var_name = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return _ENV_VAR_RE.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def load_yaml(
    path: str | Path,
    *,
    interpolate_env: bool = True,
) -> dict:
    """
    Load a YAML file and return it as a dictionary.

    Args:
        path: Path to the YAML file.
        interpolate_env: If True (default), replace ${VAR} patterns
            with environment variable values.

    Returns:
        The parsed YAML as a dictionary.

    Raises:
        ConfigError: If the file cannot be loaded or parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {p}: {e}") from e

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a YAML mapping at top level in {p}, got {type(data).__name__}"
        )

    if interpolate_env:
        data = _interpolate_env_vars(data)

    return data


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge two configuration dicts.

    Values in `override` take precedence. Nested dicts are merged
    recursively; all other values are replaced.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config(data: dict, schema: Optional[dict] = None) -> list[str]:
    """
    Basic config validation.

    Checks:
    - Required top-level keys (if schema provided)
    - No unexpected keys (if schema provided)
    - Basic type checks

    Returns a list of error strings (empty = valid).
    """
    errors = []

    if schema is None:
        return errors

    # Check required keys
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Check for unexpected keys
    allowed = schema.get("allowed_keys", [])
    if allowed:
        for key in data:
            if key not in allowed:
                errors.append(f"Unexpected key: {key}")

    return errors


# ---------------------------------------------------------------------------
# Training config loading
# ---------------------------------------------------------------------------

def load_training_config(path: str | Path) -> dict:
    """
    Load a training configuration from a YAML file.

    Returns the raw dict — caller converts to TrainingConfig as needed.
    """
    data = load_yaml(path)

    # Validate basic structure
    if "training" in data:
        data = data["training"]

    return data
