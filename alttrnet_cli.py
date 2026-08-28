"""
ALTTRNET CLI — Command-line entry points
=========================================
Provides basic CLI tools for common project operations.

Usage:
    # Run from project root
    python -m alttrnet_cli <command> [options]

    # Available commands:
    python -m alttrnet_cli info          # Show project info
    python -m alttrnet_cli validate      # Validate project config
    python -m alttrnet_cli env           # Check environment
    python -m alttrnet_cli runs          # List experiment runs
    python -m alttrnet_cli datasets      # List registered datasets
    python -m alttrnet_cli checkpoints   # List checkpoints
    python -m alttrnet_cli models        # List model metadata
    python -m alttrnet_cli test          # Run tests
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args):
    """Show project information."""
    from core.config import CHUNKING, MODELS, PATHS, PROJECT, RETRIEVAL

    print("=" * 60)
    print(f"  {PROJECT.name} v{PROJECT.version}")
    print("=" * 60)
    print(f"  Description: {PROJECT.description}")
    print(f"  Max parameters: {PROJECT.max_parameters:,}")
    print(f"  Root: {PATHS.root}")
    print()
    print("  Models:")
    print(f"    Embedding: {MODELS.embed}")
    print(f"    LLM: {MODELS.llm}")
    print(f"    Reranker: {MODELS.reranker}")
    print()
    print("  Chunking (frozen):")
    print(f"    Size: {CHUNKING.size} words")
    print(f"    Overlap: {CHUNKING.overlap} words")
    print(f"    Step: {CHUNKING.effective_step} words")
    print()
    print("  Retrieval (frozen):")
    print(f"    Dense top-K: {RETRIEVAL.dense_top_k}")
    print(f"    BM25 top-K: {RETRIEVAL.bm25_top_k}")
    print(f"    Final top-K: {RETRIEVAL.final_top_k}")
    print()


def cmd_validate(args):
    """Validate project configuration."""
    from core.config_loader import load_yaml

    config_path = args.config or "configs/default.yaml"
    print(f"Validating: {config_path}")

    try:
        data = load_yaml(config_path)
        print(f"  Loaded successfully: {len(data)} top-level keys")
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                print(f"    {key}: ({len(value)} sub-keys)")
            else:
                print(f"    {key}: {value}")
        print("\nValidation: PASS")
    except Exception as e:
        print(f"\nValidation: FAIL\n  {e}")
        return 1
    return 0


def cmd_env(args):
    """Check environment status."""
    import os

    print("Environment check:")
    print()

    # Python version
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")

    # Required packages
    packages = ["chromadb", "ollama", "rank_bm25", "torch", "numpy"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {version} [OK]")
        except (ImportError, Exception) as e:
            print(f"  {pkg}: NOT AVAILABLE ({type(e).__name__})")

    # Ollama models
    print()
    print("  Ollama models:")
    try:
        import ollama
        available = ollama.list()
        models = getattr(available, "models", [])
        model_names = []
        for m in models:
            name = getattr(m, "model", None) or m.get("name", "") if hasattr(m, "get") else ""
            model_names.append(name)

        for required in ["nomic-embed-text", "qwen3:14b"]:
            found = any(required in n for n in model_names)
            status = "[OK]" if found else "[NOT FOUND]"
            print(f"    {required}: {status}")
    except Exception:
        print("    (Ollama not reachable)")

    # Environment variables
    print()
    print("  Environment variables:")
    for var in ["ALTTRNET_API_KEY", "WANDB_API_KEY"]:
        val = os.environ.get(var)
        if val:
            print(f"    {var}: set ({val[:4]}...)")
        else:
            print(f"    {var}: not set")

    return 0


def cmd_runs(args):
    """List experiment runs."""
    from core.runs import RunManager

    rm = RunManager("experiments/")
    print(rm.summary())
    return 0


def cmd_datasets(args):
    """List registered datasets."""
    from core.datasets import DatasetRegistry

    registry_dir = Path("datasets/")
    if not registry_dir.is_dir():
        print("No datasets/ directory found.")
        print("Create it and register datasets with core.datasets.DatasetRegistry.")
        return 0

    reg = DatasetRegistry(registry_dir)
    datasets = reg.list_datasets()
    if not datasets:
        print("No datasets registered.")
        return 0

    print(f"Registered datasets ({len(datasets)}):")
    for name in datasets:
        manifest = reg.get(name)
        if manifest:
            print(f"  - {manifest.summary()}")
        else:
            print(f"  - {name} (manifest not found)")
    return 0


def cmd_checkpoints(args):
    """List checkpoints."""
    from core.checkpoint import list_checkpoints

    checkpoints = list_checkpoints("checkpoints/")
    if not checkpoints:
        print("No checkpoints found.")
        return 0

    print(f"Checkpoints ({len(checkpoints)}):")
    for ckpt in checkpoints:
        print(ckpt.summary())
        print()
    return 0


def cmd_models(args):
    """List model metadata."""
    from core.model_schema import list_models

    models = list_models("models/")
    if not models:
        print("No models found.")
        print("Create model metadata with core.model_schema.ModelMetadata.")
        return 0

    print(f"Models ({len(models)}):")
    for model in models:
        print(model.summary())
        print()
    return 0


def cmd_test(args):
    """Run the test suite."""
    import subprocess

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    if args.test_path:
        cmd = [sys.executable, "-m", "pytest", args.test_path, "-v", "--tb=short"]

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        prog="alttrnet",
        description="Alttrnet CLI — project management and diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    subparsers.add_parser("info", help="Show project information")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate project config")
    p_validate.add_argument("--config", type=str, default=None,
                            help="Config file path (default: configs/default.yaml)")

    # env
    subparsers.add_parser("env", help="Check environment status")

    # runs
    subparsers.add_parser("runs", help="List experiment runs")

    # datasets
    subparsers.add_parser("datasets", help="List registered datasets")

    # checkpoints
    subparsers.add_parser("checkpoints", help="List checkpoints")

    # models
    subparsers.add_parser("models", help="List model metadata")

    # test
    p_test = subparsers.add_parser("test", help="Run tests")
    p_test.add_argument("test_path", nargs="?", default=None,
                        help="Specific test file/directory to run")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "info": cmd_info,
        "validate": cmd_validate,
        "env": cmd_env,
        "runs": cmd_runs,
        "datasets": cmd_datasets,
        "checkpoints": cmd_checkpoints,
        "models": cmd_models,
        "test": cmd_test,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
