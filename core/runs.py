"""
core/runs.py — ALTTRNET experiment run directory management
============================================================
Creates and manages structured run directories for experiments.

Each experiment run gets a unique directory containing:
  - metadata.json  (ExperimentMeta)
  - config.yaml    (snapshot of the config used)
  - logs/          (JSONL log files)
  - outputs/       (model outputs, predictions, etc.)
  - plots/         (generated visualizations)

Run directories are named with timestamps and experiment names for
easy identification and sorting.

Usage:
    from core.runs import RunManager

    rm = RunManager("experiments/")
    run_dir = rm.create_run(
        name="retrieval_baseline",
        experiment_step="1B.3",
        config={"chunk_size": 400, "top_k": 5},
    )
    print(run_dir)  # experiments/retrieval_baseline_20260827_143000

    # Later, list runs
    runs = rm.list_runs()
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.exceptions import ExperimentError
from core.experiment import ExperimentMeta


class RunManager:
    """
    Manages experiment run directories.

    Creates structured directories with metadata, config snapshots,
    and standard subdirectories for logs and outputs.
    """

    def __init__(self, experiments_dir: str | Path):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        name: str,
        *,
        experiment_step: str = "",
        description: str = "",
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> Path:
        """
        Create a new run directory with standard structure.

        Returns the path to the created run directory.
        """
        # Generate unique run directory name
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "_")
        dir_name = f"{safe_name}_{ts}"
        run_dir = self.experiments_dir / dir_name

        if run_dir.exists():
            raise ExperimentError(
                f"Run directory already exists: {run_dir}",
                details={"name": name, "dir_name": dir_name},
            )

        # Create subdirectories
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Create and save experiment metadata
        exp = ExperimentMeta(
            name=name,
            step=experiment_step,
            description=description,
            config=config or {},
            status="created",
        )
        exp.save(run_dir / "metadata.json")

        # Save config snapshot as YAML-compatible JSON
        if config:
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        # Save tags if provided
        if tags:
            (run_dir / "tags.json").write_text(
                json.dumps(tags, indent=2) + "\n",
                encoding="utf-8",
            )

        return run_dir

    def get_run(self, run_name: str) -> Optional[Path]:
        """Find a run directory by name (partial match)."""
        for child in sorted(self.experiments_dir.iterdir()):
            if child.is_dir() and child.name.startswith(run_name):
                return child
        return None

    def list_runs(
        self,
        *,
        name_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        List all runs in the experiments directory.

        Returns a list of dicts with run metadata.
        """
        runs = []
        for child in sorted(self.experiments_dir.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "metadata.json"
            if not meta_path.is_file():
                continue

            try:
                exp = ExperimentMeta.load(meta_path)
            except Exception:
                continue

            if name_filter and name_filter not in exp.name:
                continue
            if status_filter and exp.status != status_filter:
                continue

            runs.append({
                "dir_name": child.name,
                "name": exp.name,
                "step": exp.step,
                "status": exp.status,
                "created_at": exp.created_at,
                "fingerprint": exp.fingerprint(),
                "path": str(child),
            })

        return runs

    def update_status(self, run_dir: str | Path, status: str) -> None:
        """Update the status of a run."""
        p = Path(run_dir)
        meta_path = p / "metadata.json"
        if not meta_path.is_file():
            raise ExperimentError(f"No metadata.json found in {p}")

        exp = ExperimentMeta.load(meta_path)
        exp.set_status(status)
        exp.save(meta_path)

    def add_result(self, run_dir: str | Path, key: str, value: Any) -> None:
        """Add a result metric to a run."""
        p = Path(run_dir)
        meta_path = p / "metadata.json"
        if not meta_path.is_file():
            raise ExperimentError(f"No metadata.json found in {p}")

        exp = ExperimentMeta.load(meta_path)
        exp.add_result(key, value)
        exp.save(meta_path)

    def get_log_dir(self, run_dir: str | Path) -> Path:
        """Get the logs subdirectory for a run."""
        return Path(run_dir) / "logs"

    def get_output_dir(self, run_dir: str | Path) -> Path:
        """Get the outputs subdirectory for a run."""
        return Path(run_dir) / "outputs"

    def summary(self) -> str:
        """Human-readable summary of all runs."""
        runs = self.list_runs()
        if not runs:
            return "No experiment runs found."

        lines = [f"Experiment runs ({len(runs)} total):", ""]
        for r in runs:
            lines.append(
                f"  [{r['status']:>10}] {r['name']} "
                f"(step={r['step'] or '-'}) — {r['dir_name']}"
            )
        return "\n".join(lines)
