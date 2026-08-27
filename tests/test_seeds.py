"""
tests/test_seeds.py — Tests for core.seeds (reproducibility)
=============================================================
"""

import random

import pytest

from core.seeds import set_global_seed, get_global_seed, derive_seed, experiment_seed


class TestSetGlobalSeed:
    def test_sets_seed(self):
        set_global_seed(42)
        assert get_global_seed() == 42

    def test_python_random_reproducible(self):
        set_global_seed(123)
        vals1 = [random.random() for _ in range(10)]
        set_global_seed(123)
        vals2 = [random.random() for _ in range(10)]
        assert vals1 == vals2

    def test_returns_seed(self):
        result = set_global_seed(99)
        assert result == 99

    def test_default_seed(self):
        result = set_global_seed()
        assert result == 42


class TestDeriveSeed:
    def test_deterministic(self):
        s1 = derive_seed(42, "data")
        s2 = derive_seed(42, "data")
        assert s1 == s2

    def test_different_labels_different_seeds(self):
        s1 = derive_seed(42, "data")
        s2 = derive_seed(42, "model")
        assert s1 != s2

    def test_different_bases_different_seeds(self):
        s1 = derive_seed(42, "data")
        s2 = derive_seed(99, "data")
        assert s1 != s2

    def test_returns_int(self):
        s = derive_seed(42, "test")
        assert isinstance(s, int)


class TestExperimentSeed:
    def test_deterministic(self):
        s1 = experiment_seed("exp_1", run_index=0)
        s2 = experiment_seed("exp_1", run_index=0)
        assert s1 == s2

    def test_different_runs(self):
        s1 = experiment_seed("exp_1", run_index=0)
        s2 = experiment_seed("exp_1", run_index=1)
        assert s1 != s2

    def test_different_experiments(self):
        s1 = experiment_seed("exp_1")
        s2 = experiment_seed("exp_2")
        assert s1 != s2

    def test_returns_int(self):
        s = experiment_seed("test")
        assert isinstance(s, int)
