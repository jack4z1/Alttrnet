"""
core/model_schema.py — ALTTRNET model configuration/metadata schemas
=====================================================================
Defines the standard schema for model metadata and configuration files.
Every model variant produced by the project should have a self-describing
metadata file alongside its weights.

This is a FOUNDATION module — it defines the metadata envelope without
choosing specific architectures.

Usage:
    from core.model_schema import ModelMetadata

    meta = ModelMetadata(
        name="17b_dense_v1",
        description="17B dense transformer baseline",
        params=17_000_000_000,
        config={
            "arch_type": "dense",
            "num_layers": 40,
            "hidden_size": 6144,
            "num_heads": 48,
        },
    )
    meta.save("models/17b_dense_v1/metadata.json")
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ModelMetadata:
    """
    Self-describing metadata for an Alttrnet model variant.

    Every saved model should have one of these alongside its weights.
    """

    # Identity
    name: str = ""
    version: str = "1.0"
    description: str = ""

    # Architecture
    arch_type: str = ""  # dense | moe | ssa | hybrid
    params: int = 0  # total parameter count
    config: dict = field(default_factory=dict)
    # Full architecture config dict (flexible for different arch types)

    # Training
    trained_on: str = ""  # dataset name(s)
    training_config: dict = field(default_factory=dict)
    # Snapshot of training hyperparameters

    # Performance
    benchmark_scores: dict = field(default_factory=dict)
    # Example: {"humaneval": 0.45, "mbpp": 0.52}

    # Files
    weight_files: list = field(default_factory=list)
    # Example: ["model.safetensors", "model-00001-of-00003.safetensors"]

    # Provenance
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    trained_by: str = ""  # worker name / agent
    parent_model: Optional[str] = None  # for distilled models
    base_checkpoint: Optional[str] = None  # checkpoint this was derived from

    # Integrity
    checksum_sha256: str = ""

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def fingerprint(self) -> str:
        """Deterministic fingerprint of the model config."""
        payload = json.dumps(
            {"arch_type": self.arch_type, "config": self.config, "params": self.params},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        """Save metadata to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Auto-populate git commit
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

        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ModelMetadata":
        """Load metadata from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("fingerprint", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """
        Validate the metadata for consistency.

        Returns a list of error strings (empty = valid).
        """
        errors = []
        if not self.name:
            errors.append("Model name is empty")
        if self.params < 0:
            errors.append(f"Parameter count is negative: {self.params}")
        if self.params > 17_000_000_000:
            errors.append(
                f"Parameter count exceeds 17B cap: {self.params:,}"
            )
        if not self.arch_type:
            errors.append("Architecture type is not specified")
        return errors

    def param_budget_usage(self) -> float:
        """Fraction of the 17B parameter budget used."""
        return self.params / 17_000_000_000

    def summary(self) -> str:
        """Human-readable summary."""
        budget_pct = self.param_budget_usage() * 100
        lines = [
            f"Model: {self.name} v{self.version}",
            f"  Architecture: {self.arch_type or '(unspecified)'}",
            f"  Parameters: {self.params:,} ({budget_pct:.1f}% of 17B budget)",
        ]
        if self.benchmark_scores:
            scores = [f"{k}={v}" for k, v in self.benchmark_scores.items()]
            lines.append(f"  Benchmarks: {', '.join(scores)}")
        if self.trained_on:
            lines.append(f"  Trained on: {self.trained_on}")
        if self.git_commit:
            lines.append(f"  Git: {self.git_commit}")
        lines.append(f"  Created: {self.created_at}")
        return "\n".join(lines)


def list_models(models_dir: str | Path) -> list[ModelMetadata]:
    """List all model metadata in the models directory."""
    mdir = Path(models_dir)
    if not mdir.is_dir():
        return []

    models = []
    for child in sorted(mdir.iterdir()):
        if child.is_dir():
            meta_path = child / "metadata.json"
            if meta_path.is_file():
                try:
                    models.append(ModelMetadata.load(meta_path))
                except Exception:
                    continue
    return models
