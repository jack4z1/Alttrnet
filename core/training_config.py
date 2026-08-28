"""
core/training_config.py — ALTTRNET training configuration interfaces
=====================================================================
Defines the configuration schema for future training runs. This is a
FOUNDATION module — it defines the structure without choosing specific
hyperparameters, architectures, or strategies.

All fields have sensible defaults or are marked as required. Training
configs can be loaded from YAML files via core.config_loader.

Usage:
    from core.training_config import TrainingConfig, ModelConfig, DataConfig

    # Create from defaults
    tc = TrainingConfig(
        model=ModelConfig(name="17b_dense", params=17_000_000_000),
        data=DataConfig(train_dataset="code_alpaca_20k"),
    )

    # Load from YAML
    from core.config_loader import load_training_config
    tc = load_training_config("configs/training/base.yaml")
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArchConfig:
    """Model architecture configuration (scaffold — choices deferred to research)."""

    # Architecture type
    arch_type: str = "dense"  # dense | moe | ssa | hybrid

    # Size constraints
    total_params: int = 17_000_000_000  # hard cap
    vocab_size: int = 32000
    max_seq_len: int = 8192

    # Layer configuration (scaffold)
    num_layers: int = 0  # 0 = auto-compute from other params
    hidden_size: int = 0
    num_heads: int = 0
    intermediate_size: int = 0

    # MoE-specific (scaffold, only used if arch_type == "moe")
    num_experts: int = 0
    expert_top_k: int = 0

    def param_budget_remaining(self, embedding_params: int = 0) -> int:
        """How many parameters remain for the model body."""
        return self.total_params - embedding_params


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"  # adamw | sgd | adafactor | 8bit_adamw
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    scheduler: str = "cosine"  # cosine | linear | constant | warmup_cosine
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1  # minimum lr as fraction of peak lr

    # Optional: 8-bit optimizer
    use_8bit: bool = False


@dataclass
class ModelConfig:
    """Full model configuration (architecture + optimizer)."""

    name: str = ""  # model name for this run
    architecture: ModelArchConfig = field(default_factory=ModelArchConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Checkpointing
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 3
    eval_every_n_steps: int = 500

    # Mixed precision
    dtype: str = "bfloat16"  # float32 | float16 | bfloat16
    use_amp: bool = True


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data configuration for training."""

    # Datasets
    train_dataset: str = ""  # dataset name or path
    eval_dataset: str = ""
    test_dataset: str = ""

    # Data mixing (for multi-dataset training)
    mixture_weights: dict = field(default_factory=dict)
    # Example: {"code": 0.6, "math": 0.2, "general": 0.2}

    # Preprocessing
    max_seq_len: int = 8192
    pad_to_max: bool = True
    packing: bool = False  # pack short sequences together

    # Dataloader
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # Tokenization
    tokenizer: str = ""  # tokenizer name or path (empty = default)

    @property
    def effective_batch_size(self) -> int:
        """Total effective batch size across all GPUs and accumulation steps."""
        return self.batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    Combines model, data, and training parameters into a single
    configuration object that can be serialized, loaded, and compared.
    """

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Training parameters
    max_steps: int = 100000
    max_epochs: int = 0  # 0 = use max_steps instead
    resume_from: str = ""  # checkpoint path to resume from

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000

    # Hardware
    num_gpus: int = 1
    compile_model: bool = False  # torch.compile

    # Experiment metadata
    experiment_name: str = ""
    experiment_step: str = ""
    description: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary (for serialization)."""
        from dataclasses import asdict
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Training config: {self.experiment_name or '(unnamed)'}",
            f"  Max steps: {self.max_steps}",
            f"  Effective batch size: {self.data.effective_batch_size}",
            f"  Learning rate: {self.model.optimizer.lr}",
            f"  Seed: {self.seed}",
            f"  GPUs: {self.num_gpus}",
            f"  Model: {self.model.name or '(unnamed)'}",
            f"  Train dataset: {self.data.train_dataset or '(not set)'}",
        ]
        return "\n".join(lines)
