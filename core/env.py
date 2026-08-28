"""
core/env.py — ALTTRNET environment loading and validation
==========================================================
Provides a simple, portable way to load environment variables from a
.env file and validate that required variables are present.

This does NOT use python-dotenv to avoid adding another dependency.
It reads .env files in the standard KEY=VALUE format (comments and
blank lines are ignored, values may be quoted).

Usage:
    from core.env import load_env, require_env

    load_env()  # loads .env from project root (if it exists)
    api_key = require_env("MY_API_KEY")
"""

import os
from pathlib import Path
from typing import Optional

from core.exceptions import EnvError


def _parse_value(raw: str) -> str:
    """Strip surrounding quotes from a value, if present."""
    stripped = raw.strip()
    if len(stripped) >= 2:
        if (stripped[0] == stripped[-1]) and stripped[0] in ("'", '"'):
            return stripped[1:-1]
    return stripped


def load_env(
    path: Optional[str | Path] = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        path: Path to the .env file. Defaults to `<project_root>/.env`.
        override: If True, existing env vars are overwritten. If False
                  (default), only missing variables are set.

    Returns:
        A dict of variables that were loaded (set or already present).

    Raises:
        EnvError: If the specified path is given but does not exist.
    """
    if path is not None:
        env_path = Path(path)
        if not env_path.is_file():
            raise EnvError(f"Env file not found: {env_path}")
    else:
        from core.config import PATHS
        env_path = PATHS.root / ".env"

    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded

    with open(env_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, raw_value = line.partition("=")
            key = key.strip()
            value = _parse_value(raw_value)

            if not key:
                continue

            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = value

    return loaded


def require_env(*keys: str, _caller: str = "") -> dict[str, str]:
    """
    Ensure that the given environment variables are set and non-empty.

    Args:
        keys: Environment variable names that must be present.

    Returns:
        A dict mapping each key to its value.

    Raises:
        EnvError: If any required variable is missing or empty.
    """
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        ctx = f" (required by {_caller})" if _caller else ""
        raise EnvError(
            f"Missing required environment variable(s): {', '.join(missing)}{ctx}",
            details={"missing": missing},
        )
    return {k: os.environ[k] for k in keys}


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a default fallback."""
    return os.environ.get(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """Get an environment variable as an integer, with a default."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        from core.exceptions import ConfigError
        raise ConfigError(
            f"Environment variable {key!r} must be an integer, got {raw!r}"
        )


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get an environment variable as a boolean. Truthy values: 1, true, yes, on."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")
