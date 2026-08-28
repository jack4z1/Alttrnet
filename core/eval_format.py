"""
core/eval_format.py — ALTTRNET evaluation result format
========================================================
Defines a standard data structure for evaluation results so that
different evaluation harnesses (retrieval, generation, end-to-end)
produce comparable, machine-readable output.

This does NOT define the evaluation metrics themselves — those are
specific to each experiment. This only defines the envelope.

Usage:
    from core.eval_format import EvalResult, EvalSuite

    result = EvalResult(
        name="dense_baseline",
        metrics={"precision@5": 0.83, "hit@1": 0.93, "mrr": 0.96},
        metadata={"questions": 15, "model": "nomic-embed-text"},
    )

    suite = EvalSuite(name="retrieval_step1b")
    suite.add(result)
    suite.save("artifacts/eval_results.json")
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    """
    A single evaluation result: name + metrics + metadata.

    Each eval harness creates one EvalResult per system/configuration
    it tests.
    """
    name: str = ""
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    per_question: list = field(default_factory=list)
    per_source: dict = field(default_factory=dict)
    per_type: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def add_per_question(self, qnum: int, question: str, metrics: dict) -> None:
        self.per_question.append({
            "qnum": qnum,
            "question": question,
            **metrics,
        })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalSuite:
    """
    A collection of EvalResults from a single evaluation run.

    Groups results by system/configuration for comparison.
    """
    name: str = ""
    description: str = ""
    results: list = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, result: EvalResult) -> None:
        """Add an evaluation result to this suite."""
        self.results.append(result)

    def comparison_table(self) -> str:
        """
        Generate a human-readable comparison table of all results.

        Returns a formatted string suitable for terminal output.
        """
        if not self.results:
            return "(no results)"

        # Collect all metric names
        all_metrics = []
        for r in self.results:
            for k in r.metrics:
                if k not in all_metrics:
                    all_metrics.append(k)

        # Header
        name_width = max(len(r.name) for r in self.results)
        lines = []
        header = f"{'System':<{name_width}}"
        for m in all_metrics:
            header += f" | {m:>12}"
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for r in self.results:
            row = f"{r.name:<{name_width}}"
            for m in all_metrics:
                v = r.metrics.get(m)
                if v is None:
                    row += f" | {'N/A':>12}"
                elif isinstance(v, float):
                    row += f" | {v:>12.4f}"
                else:
                    row += f" | {str(v):>12}"
            lines.append(row)

        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Save the evaluation suite to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }
        p.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EvalSuite":
        """Load an evaluation suite from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = [EvalResult(**r) for r in data.get("results", [])]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            results=results,
            timestamp=data.get("timestamp", ""),
        )
