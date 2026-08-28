"""
core/checkpoint.py — ALTTRNET checkpoint metadata & management
===============================================================
Defines the standard schema for model checkpoints so that every saved
state is self-describing and reproducible. This is a FOUNDATION module —
it defines the metadata envelope and file management without choosing
a specific model architecture or serialization format.

Usage:
    from core.checkpoint import CheckpointMeta

    ckpt = CheckpointMeta(
        name="step_2a_baseline",
        step="2A.1",
        epoch=0,
        global_step=1000,
        config={"model": "17b_dense", "lr": 3e-4},
    )
    ckpt.save("checkpoints/step_2a_baseline/")

    # Load and verify
    loaded = CheckpointMeta.load("checkpoints/step_2a_baseline/")
    print(loaded.summary())
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.exceptions import CheckpointError


@dataclass
class CheckpointMeta:
    """
    Standard metadata envelope for an Alttrnet checkpoint.

    Every checkpoint saved to `checkpoints/` should have one of these
    alongside the actual model weights / optimizer state.
    """

    # Identity
    name: str = ""
    step: str = ""  # experiment step, e.g. "2A.1"
    description: str = ""

    # Training state
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0  # expected total steps (for progress tracking)

    # Hyperparameters snapshot
    config: dict = field(default_factory=dict)
    # Example: {"lr": 3e-4, "batch_size": 8, "model_name": "17b_dense"}

    # Model metrics at save time
    metrics: dict = field(default_factory=dict)
    # Example: {"train_loss": 2.34, "eval_loss": 2.51, "perplexity": 12.3}

    # File references (relative to checkpoint dir)
    files: dict = field(default_factory=dict)
    # Example: {"model": "model.safetensors", "optimizer": "optimizer.pt", "config": "config.json"}

    # Provenance
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    parent_checkpoint: Optional[str] = None  # name of previous checkpoint

    # Integrity
    checksum_sha256: str = ""  # checksum of the primary model file

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def fingerprint(self) -> str:
        """
        Deterministic fingerprint of config + metrics.

        Two checkpoints with the same config and metrics will have
        the same fingerprint.
        """
        payload = json.dumps(
            {"config": self.config, "metrics": self.metrics},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, checkpoint_dir: str | Path) -> Path:
        """
        Save checkpoint metadata to a directory.

        Creates the directory if needed and writes `metadata.json` inside it.
        Also populates git_commit and python_version automatically.
        """
        p = Path(checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)

        meta_path = p / "metadata.json"

        # Auto-populate provenance
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

        import sys
        self.python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        meta_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return meta_path

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "CheckpointMeta":
        """Load checkpoint metadata from a directory."""
        p = Path(checkpoint_dir)
        meta_path = p / "metadata.json"
        if not meta_path.is_file():
            raise CheckpointError(
                f"No metadata.json found in {p}",
                details={"checkpoint_dir": str(p)},
            )
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        # Remove computed fields
        data.pop("fingerprint", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """
        Validate the checkpoint metadata for consistency.

        Returns a list of error strings (empty = valid).
        """
        errors = []
        if not self.name:
            errors.append("Checkpoint name is empty")
        if self.global_step < 0:
            errors.append(f"global_step is negative: {self.global_step}")
        if self.epoch < 0:
            errors.append(f"epoch is negative: {self.epoch}")
        if self.total_steps > 0 and self.global_step > self.total_steps:
            errors.append(
                f"global_step ({self.global_step}) > total_steps ({self.total_steps})"
            )
        return errors

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Checkpoint: {self.name}",
            f"  Step: {self.step or '(unspecified)'}",
            f"  Epoch: {self.epoch}, Global step: {self.global_step}",
        ]
        if self.total_steps:
            pct = 100.0 * self.global_step / self.total_steps
            lines.append(f"  Progress: {pct:.1f}%")
        if self.metrics:
            metric_strs = [f"{k}={v}" for k, v in self.metrics.items()]
            lines.append(f"  Metrics: {', '.join(metric_strs)}")
        if self.git_commit:
            lines.append(f"  Git: {self.git_commit}")
        lines.append(f"  Created: {self.created_at}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint directory management
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoints_dir: str | Path) -> list[CheckpointMeta]:
    """
    List all checkpoint metadata in the checkpoints directory.

    Each checkpoint is a subdirectory containing metadata.json.
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        return []

    checkpoints = []
    for child in sorted(ckpt_dir.iterdir()):
        if child.is_dir() and (child / "metadata.json").is_file():
            try:
                checkpoints.append(CheckpointMeta.load(child))
            except Exception:
                continue
    return checkpoints


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Optional[CheckpointMeta]:
    """Find the most recently created checkpoint."""
    checkpoints = list_checkpoints(checkpoints_dir)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: c.created_at)


def verify_checkpoint(checkpoint_dir: str | Path) -> dict:
    """
    Verify a checkpoint directory's integrity.

    Returns a dict with 'valid' (bool), 'errors' (list), and
    'files_found' (list).
    """
    p = Path(checkpoint_dir)
    result: dict[str, Any] = {"valid": True, "errors": [], "files_found": []}

    if not p.is_dir():
        result["valid"] = False
        result["errors"].append(f"Directory does not exist: {p}")
        return result

    # Check metadata
    meta_path = p / "metadata.json"
    if not meta_path.is_file():
        result["valid"] = False
        result["errors"].append("metadata.json not found")
    else:
        try:
            meta = CheckpointMeta.load(p)
            errors = meta.validate()
            if errors:
                result["valid"] = False
                result["errors"].extend(errors)
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to load metadata: {e}")

    # List files
    result["files_found"] = [f.name for f in sorted(p.iterdir())]

    return result
