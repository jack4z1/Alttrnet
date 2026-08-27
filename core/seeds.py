"""
core/seeds.py — ALTTRNET reproducibility & seed management
==========================================================
Deterministic seed setting for all sources of randomness used in
experiments. Call `set_global_seed()` at the start of any experiment
that requires reproducibility.

Note: The current RAG prototype is deterministic by design (no training,
no random splits). This module is a FOUNDATION for future training and
evaluation work where seeding matters.

Usage:
    from core.seeds import set_global_seed, get_global_seed

    set_global_seed(42)   # call once at experiment start
    # All subsequent random operations use seed 42
"""

import hashlib
import os
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Global seed state
# ---------------------------------------------------------------------------

_global_seed: Optional[int] = None
SEED_DEFAULT = 42


def set_global_seed(seed: int = SEED_DEFAULT) -> int:
    """
    Set seeds for Python's random, os.environ, and (if available) numpy/torch.

    Returns the seed that was set.  Call once at the start of an experiment
    or training run.  Subsequent calls are allowed but will print a warning
    if the seed changes.
    """
    global _global_seed

    if _global_seed is not None and _global_seed != seed:
        print(f"WARNING: seed changing from {_global_seed} to {seed}")

    _global_seed = seed

    # Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy (optional)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # torch (optional)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    print(f"Global seed set to {seed}")
    return seed


def get_global_seed() -> Optional[int]:
    """Return the currently active global seed, or None if not set."""
    return _global_seed


def derive_seed(base: int, label: str) -> int:
    """
    Deterministically derive a child seed from a base seed + label.

    Useful for giving different components different but reproducible seeds:
        seed_data = derive_seed(42, "data")
        seed_model = derive_seed(42, "model")
    """
    h = hashlib.sha256(f"{base}:{label}".encode()).hexdigest()
    return int(h[:8], 16)


def experiment_seed(experiment_name: str, run_index: int = 0) -> int:
    """
    Generate a deterministic seed for a named experiment + run index.

    The seed is reproducible across machines and Python versions.
    """
    h = hashlib.sha256(f"alttrnet:{experiment_name}:run{run_index}".encode()).hexdigest()
    return int(h[:8], 16)
