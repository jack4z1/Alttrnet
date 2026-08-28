"""
core/datasets.py — ALTTRNET dataset manifest, schema, and validation
=====================================================================
Provides infrastructure for tracking, validating, and managing datasets
used in training and evaluation. This is a FOUNDATION module — it defines
schemas and validation logic without choosing specific datasets.

Usage:
    from core.datasets import DatasetManifest, DatasetValidator

    # Create a manifest for a new dataset
    manifest = DatasetManifest(
        name="code_alpaca_20k",
        version="1.0",
        source="https://...",
        description="Code instruction dataset",
        num_samples=20000,
        format="jsonl",
    )

    # Validate a dataset file
    validator = DatasetValidator(manifest)
    report = validator.validate("data/code_alpaca_20k.jsonl")
    if not report.is_valid:
        print(report.errors)

    # Save/load manifest
    manifest.save("datasets/code_alpaca_20k/manifest.json")
    loaded = DatasetManifest.load("datasets/code_alpaca_20k/manifest.json")
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Dataset manifest schema
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """
    Metadata schema for a single dataset version.

    Every dataset in the project should have a manifest that describes
    its provenance, size, format, and validation status.
    """

    # Identity
    name: str = ""
    version: str = "1.0"
    description: str = ""

    # Source provenance
    source: str = ""  # URL, path, or citation
    source_type: str = "unknown"  # web | file | api | synthetic | mixed
    license: str = ""
    language: str = "en"

    # Structure
    format: str = "jsonl"  # jsonl | csv | parquet | text | arrow
    num_samples: int = 0
    num_bytes: int = 0
    split: str = "train"  # train | val | test | all

    # Schema description (field names and types)
    fields: dict = field(default_factory=dict)
    # Example: {"instruction": "string", "input": "string", "output": "string"}

    # Validation
    validated: bool = False
    validation_date: str = ""
    validation_errors: list = field(default_factory=list)
    checksum_sha256: str = ""

    # Provenance
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_by: str = "alttrnet"
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Save the manifest to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        """Load a manifest from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def compute_checksum(self, file_path: str | Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self.checksum_sha256 = h.hexdigest()
        return self.checksum_sha256

    def summary(self) -> str:
        """Human-readable summary of this dataset."""
        lines = [
            f"Dataset: {self.name} v{self.version}",
            f"  Format: {self.format}",
            f"  Samples: {self.num_samples:,}",
            f"  Split: {self.split}",
            f"  Language: {self.language}",
            f"  Source: {self.source or '(not specified)'}",
            f"  Validated: {'yes' if self.validated else 'no'}",
        ]
        if self.tags:
            lines.append(f"  Tags: {', '.join(self.tags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """
    Registry of all known datasets in the project.

    Maintains a JSON index file that maps dataset names to their
    manifest locations, enabling quick lookups.
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self.index_path.is_file():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def register(self, manifest: DatasetManifest, manifest_path: str | Path) -> None:
        """Register a dataset manifest in the index."""
        self._index[manifest.name] = str(manifest_path)
        self._save_index()

    def get(self, name: str) -> Optional[DatasetManifest]:
        """Look up a dataset by name and load its manifest."""
        if name not in self._index:
            return None
        return DatasetManifest.load(self._index[name])

    def list_datasets(self) -> list[str]:
        """Return names of all registered datasets."""
        return sorted(self._index.keys())

    def remove(self, name: str) -> bool:
        """Remove a dataset from the registry (does not delete the file)."""
        if name in self._index:
            del self._index[name]
            self._save_index()
            return True
        return False


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Result of validating a dataset file against a manifest."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [f"Validation: {'PASS' if self.is_valid else 'FAIL'}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.stats:
            lines.append(f"  Stats: {self.stats}")
        return "\n".join(lines)


class DatasetValidator:
    """
    Validates dataset files against a manifest's declared schema.

    Performs basic structural validation: checks that the file exists,
      can be parsed according to the declared format, contains the
      expected fields, and has the expected number of samples.

    This is a scaffolding — concrete validation rules will be added as
    the data pipeline matures.
    """

    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest

    def validate(self, file_path: str | Path) -> ValidationReport:
        """
        Validate a dataset file against this manifest.

        Returns a ValidationReport with errors and warnings.
        """
        report = ValidationReport()
        p = Path(file_path)

        # Basic file checks
        if not p.is_file():
            report.add_error(f"File not found: {p}")
            return report

        file_size = p.stat().st_size
        report.stats["file_size_bytes"] = file_size

        if file_size == 0:
            report.add_error("File is empty")
            return report

        # Format-specific validation
        fmt = self.manifest.format.lower()
        if fmt == "jsonl":
            self._validate_jsonl(p, report)
        elif fmt == "csv":
            self._validate_csv(p, report)
        elif fmt == "text":
            self._validate_text(p, report)
        else:
            report.add_warning(
                f"No validator implemented for format {fmt!r} — "
                "skipping content validation"
            )

        # Validate checksum if available
        if self.manifest.checksum_sha256:
            actual = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    actual.update(chunk)
            if actual.hexdigest() != self.manifest.checksum_sha256:
                report.add_error(
                    f"Checksum mismatch: expected {self.manifest.checksum_sha256[:16]}..., "
                    f"got {actual.hexdigest()[:16]}..."
                )

        return report

    def _validate_jsonl(self, path: Path, report: ValidationReport) -> None:
        """Validate a JSONL file."""
        expected_fields = set(self.manifest.fields.keys())
        sample_count = 0
        field_counts: dict[str, int] = {}
        parse_errors = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    sample_count += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        if parse_errors <= 5:
                            report.add_error(f"Line {line_no}: JSON parse error: {e}")
                        continue

                    if expected_fields:
                        record_fields = set(record.keys())
                        missing = expected_fields - record_fields
                        if missing and parse_errors == 0:
                            # Only warn once for the first record
                            pass  # field presence checked below

                    for field_name in record:
                        field_counts[field_name] = field_counts.get(field_name, 0) + 1

        except UnicodeDecodeError:
            report.add_error("File is not valid UTF-8")
            return

        report.stats["sample_count"] = sample_count
        report.stats["parse_errors"] = parse_errors
        report.stats["fields_found"] = list(field_counts.keys())

        if parse_errors > 5:
            report.add_error(f"{parse_errors} total parse errors (showing first 5)")
        elif parse_errors > 0:
            report.add_error(f"{parse_errors} parse errors total")

        if expected_fields:
            found_fields = set(field_counts.keys())
            missing = expected_fields - found_fields
            if missing:
                report.add_error(f"Missing expected fields: {sorted(missing)}")
            extra = found_fields - expected_fields
            if extra:
                report.add_warning(f"Unexpected fields found: {sorted(extra)}")

        if self.manifest.num_samples > 0 and sample_count != self.manifest.num_samples:
            report.add_warning(
                f"Sample count mismatch: manifest says {self.manifest.num_samples}, "
                f"found {sample_count}"
            )

    def _validate_csv(self, path: Path, report: ValidationReport) -> None:
        """Basic CSV validation."""
        import csv
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                report.stats["sample_count"] = len(rows)
                if rows:
                    report.stats["fields_found"] = list(rows[0].keys())
        except Exception as e:
            report.add_error(f"CSV parse error: {e}")

    def _validate_text(self, path: Path, report: ValidationReport) -> None:
        """Basic text file validation."""
        try:
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            non_empty = [line for line in lines if line.strip()]
            report.stats["line_count"] = len(lines)
            report.stats["non_empty_lines"] = len(non_empty)
        except UnicodeDecodeError:
            report.add_error("File is not valid UTF-8")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_dataset_dir(
    datasets_root: str | Path,
    name: str,
    version: str = "1.0",
) -> Path:
    """
    Create a standard dataset directory structure.

    Returns the path to the created directory.
    """
    ds_dir = Path(datasets_root) / name / f"v{version}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir
