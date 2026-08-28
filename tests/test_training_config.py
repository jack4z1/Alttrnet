"""
tests/test_training_config.py — Tests for core.training_config
================================================================
"""


from core.training_config import (
    DataConfig,
    ModelArchConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


class TestModelArchConfig:
    def test_defaults(self):
        arch = ModelArchConfig()
        assert arch.total_params == 17_000_000_000
        assert arch.arch_type == "dense"
        assert arch.max_seq_len == 8192

    def test_param_budget(self):
        arch = ModelArchConfig(total_params=17_000_000_000)
        remaining = arch.param_budget_remaining(embedding_params=1_000_000_000)
        assert remaining == 16_000_000_000


class TestOptimizerConfig:
    def test_defaults(self):
        opt = OptimizerConfig()
        assert opt.name == "adamw"
        assert opt.lr == 3e-4
        assert opt.weight_decay == 0.1
        assert opt.scheduler == "cosine"

    def test_custom(self):
        opt = OptimizerConfig(lr=1e-3, weight_decay=0.01)
        assert opt.lr == 1e-3
        assert opt.weight_decay == 0.01


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.architecture.arch_type == "dense"
        assert mc.optimizer.name == "adamw"
        assert mc.dtype == "bfloat16"

    def test_sub_configs(self):
        mc = ModelConfig(
            architecture=ModelArchConfig(arch_type="moe", num_experts=8),
            optimizer=OptimizerConfig(lr=1e-3),
        )
        assert mc.architecture.num_experts == 8
        assert mc.optimizer.lr == 1e-3


class TestDataConfig:
    def test_defaults(self):
        dc = DataConfig()
        assert dc.batch_size == 8
        assert dc.gradient_accumulation_steps == 4
        assert dc.effective_batch_size == 32

    def test_effective_batch_size(self):
        dc = DataConfig(batch_size=4, gradient_accumulation_steps=8)
        assert dc.effective_batch_size == 32


class TestTrainingConfig:
    def test_defaults(self):
        tc = TrainingConfig()
        assert tc.max_steps == 100000
        assert tc.seed == 42
        assert tc.deterministic is True

    def test_to_dict(self):
        tc = TrainingConfig(
            experiment_name="test_exp",
            max_steps=100,
            model=ModelConfig(name="test_model"),
        )
        d = tc.to_dict()
        assert isinstance(d, dict)
        assert d["experiment_name"] == "test_exp"
        assert d["max_steps"] == 100

    def test_summary(self):
        tc = TrainingConfig(
            experiment_name="summary_test",
            max_steps=500,
            model=ModelConfig(name="my_model"),
        )
        s = tc.summary()
        assert "summary_test" in s
        assert "500" in s
        assert "my_model" in s

    def test_nested_config(self):
        tc = TrainingConfig(
            model=ModelConfig(
                name="nested",
                architecture=ModelArchConfig(arch_type="moe"),
            ),
            data=DataConfig(train_dataset="my_dataset"),
        )
        d = tc.to_dict()
        assert d["model"]["architecture"]["arch_type"] == "moe"
        assert d["data"]["train_dataset"] == "my_dataset"
