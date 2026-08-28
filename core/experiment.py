"""
core/experiment.py — ALTTRNET experiment metadata & tracking
============================================================
Defines a standard format for experiment metadata so that every run
produces a self-describing, reproducible artifact.

This does NOT choose training strategies or architectures — it only
defines the metadata envelope that any experiment fills in.

Usage:
    from core.experiment import ExperimentMeta

    exp = ExperimentMeta(
        name="retrieval_baseline_dense",
        step="1B.3",
        description="Dense-only retrieval baseline on 15 questions",
        config={
            "chunk_size": 400,
            "chunk_overlap": 50,
            "embedding_model": "nomic-embed-text",
            "top_k": 5,
        },
    )
    exp.add_result("precision_at_5", 0.8333)
    exp.add_result("hit_at_1", 0.9333)
    exp.add_result("mrr", 0.9583)
    exp.save("artifacts/retrieval_baseline_dense.json")
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentMeta:
    """
    Standard metadata envelope for an Alttrnet experiment.

    Every experiment should create one of these and fill in the fields.
    The `save()` method writes a self-describing JSON file.
    """

    # Identity
    name: str = ""
    step: str = ""  # e.g. "1B.3", "2A.1"
    description: str = ""

    # Configuration snapshot
    config: dict = field(default_factory=dict)

    # Results
    results: dict = field(default_factory=dict)

    # Provenance
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    python_version: Optional[str] = None

    # Status
    status: str = "pending"  # pending | running | completed | failed

    def add_result(self, key: str, value: Any) -> None:
        """Add a named result metric."""
        self.results[key] = value

    def set_status(self, status: str) -> None:
        """Update the experiment status."""
        self.status = status

    def fingerprint(self) -> str:
        """
        Deterministic fingerprint of the experiment config + results.

        Useful for detecting whether an experiment has been re-run with
        the same parameters and gotten the same results.
        """
        payload = json.dumps(
            {"config": self.config, "results": self.results},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def save(self, path: str | Path) -> Path:
        """
        Save the experiment metadata to a JSON file.

        Creates parent directories if needed. Returns the path written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Try to capture git commit if available
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()
        except Exception:
            pass

        # Try to capture Python version
        import sys
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentMeta":
        """Load experiment metadata from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("fingerprint", None)  # computed, not stored
        return cls(**data)
