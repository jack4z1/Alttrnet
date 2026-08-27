"""
core/logging.py — ALTTRNET lightweight structured logging
=========================================================
Provides a simple, consistent logging interface for experiments,
ingestion runs, and model evaluation. No external dependencies
beyond the Python standard library.

Designed for terminal-friendly output with optional JSON lines
for machine-readable experiment logs.

Usage:
    from core.logging import get_logger

    log = get_logger("experiment_1")
    log.info("Starting evaluation", questions=30, model="reranker")
    log.metric("p@5", 0.93)
    log.success("Evaluation complete")
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """
    Lightweight structured logger for experiment runs.

    Outputs human-readable lines to stderr and optionally appends
    JSON-lines to a log file for machine consumption.
    """

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.name = name
        self.start_time = time.perf_counter()
        self._log_file = None
        self._metrics: list[dict] = []

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._log_file = open(log_dir / f"{name}_{ts}.jsonl", "w", encoding="utf-8")

    def _elapsed(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m{seconds:.1f}s"

    def _emit(self, level: str, message: str, **kwargs: Any) -> None:
        prefix = f"[{self.name}] [{self._elapsed()}] [{level}]"
        parts = [f"{prefix} {message}"]
        for k, v in kwargs.items():
            parts.append(f"  {k}={v}")
        line = " ".join(parts)
        print(line, file=sys.stderr)

        if self._log_file is not None:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "name": self.name,
                "message": message,
                **kwargs,
            }
            self._log_file.write(json.dumps(entry) + "\n")
            self._log_file.flush()

    def info(self, message: str, **kwargs: Any) -> None:
        self._emit("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        self._emit("WARN", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit("ERROR", message, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        self._emit("OK", message, **kwargs)

    def metric(self, name: str, value: Any, **kwargs: Any) -> None:
        """Log a named metric (also appended to metrics list)."""
        self._emit("METRIC", f"{name}={value}", **kwargs)
        self._metrics.append({
            "name": name,
            "value": value,
            "elapsed": self._elapsed(),
            **kwargs,
        })

    def section(self, title: str) -> None:
        """Print a visual section separator."""
        sep = "=" * 60
        print(f"\n{sep}", file=sys.stderr)
        print(f"  {title}", file=sys.stderr)
        print(f"{sep}", file=sys.stderr)

    def summary(self) -> dict:
        """Return a summary dict of all logged metrics."""
        return {
            "name": self.name,
            "elapsed": self._elapsed(),
            "metrics": self._metrics,
        }

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_logger(name: str, log_dir: Optional[Path] = None) -> ExperimentLogger:
    """
    Create an ExperimentLogger with the given name.

    If log_dir is provided, JSON-lines logs are written there.
    """
    return ExperimentLogger(name, log_dir)
