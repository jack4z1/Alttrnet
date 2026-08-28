"""
core/trainer.py — ALTTRNET training engine interface scaffold
==============================================================
Defines the abstract interface for the training engine. This module
specifies what the training loop must do without implementing the
actual forward/backward pass, gradient computation, or distributed
training logic.

The training engine will be implemented when the model architecture
is decided. For now, this provides:

1. Abstract Trainer class with lifecycle hooks
2. Callback system for monitoring
3. Progress tracking
4. Checkpoint integration

Usage:
    from core.trainer import Trainer, TrainerCallback

    class MyCallback(TrainerCallback):
        def on_train_step(self, state):
            print(f"Step {state.global_step}: loss={state.loss}")

    # When implemented:
    # trainer = MyTrainer(config)
    # trainer.train(dataset, callbacks=[MyCallback()])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.training_config import TrainingConfig

# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """
    Mutable state object passed to callbacks during training.

    Updated by the training loop at each step.
    """

    global_step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    metrics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def progress(self) -> float:
        """Training progress as a fraction [0, 1]."""
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.global_step / self.total_steps)

    @property
    def progress_pct(self) -> str:
        """Training progress as a percentage string."""
        return f"{self.progress * 100:.1f}%"

    def summary(self) -> str:
        """One-line summary of current state."""
        return (
            f"step={self.global_step}/{self.total_steps} "
            f"({self.progress_pct}) "
            f"epoch={self.epoch} "
            f"loss={self.loss:.4f} "
            f"lr={self.learning_rate:.2e}"
        )


# ---------------------------------------------------------------------------
# Callback interface
# ---------------------------------------------------------------------------

class TrainerCallback(ABC):
    """
    Abstract callback for training events.

    Subclass this and override methods to monitor or influence training.
    All methods have empty default implementations so subclasses only
    need to override what they care about.
    """

    def on_train_begin(self, state: TrainState, config: TrainingConfig) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, state: TrainState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, state: TrainState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, state: TrainState) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_step(self, state: TrainState) -> None:
        """Called after each training step."""
        pass

    def on_eval_begin(self, state: TrainState) -> None:
        """Called before evaluation starts."""
        pass

    def on_eval_end(self, state: TrainState, eval_metrics: dict) -> None:
        """Called after evaluation completes."""
        pass

    def on_checkpoint_save(self, state: TrainState, path: str) -> None:
        """Called when a checkpoint is saved."""
        pass

    def on_log(self, state: TrainState, metrics: dict) -> None:
        """Called when metrics are logged."""
        pass


# ---------------------------------------------------------------------------
# Concrete callbacks (examples / built-in)
# ---------------------------------------------------------------------------

class LoggingCallback(TrainerCallback):
    """Logs training progress to stderr."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every

    def on_train_step(self, state: TrainState) -> None:
        if state.global_step % self.log_every == 0:
            print(state.summary())

    def on_train_begin(self, state: TrainState, config: TrainingConfig) -> None:
        print(f"Training started: {config.summary()}")

    def on_train_end(self, state: TrainState) -> None:
        print(f"Training complete at step {state.global_step}")


class CheckpointCallback(TrainerCallback):
    """Saves checkpoints at specified intervals."""

    def __init__(self, save_every: int = 1000, checkpoint_dir: str = "checkpoints"):
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir

    def on_train_step(self, state: TrainState) -> None:
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            # Actual checkpoint saving is done by the Trainer
            # This callback signals that a save should happen
            pass


# ---------------------------------------------------------------------------
# Abstract Trainer
# ---------------------------------------------------------------------------

class Trainer(ABC):
    """
    Abstract base class for the training engine.

    Subclasses implement the actual training loop. The interface
    defines the lifecycle:
        1. __init__(config)
        2. train(dataset, callbacks)
        3. evaluate(dataset)
        4. save_checkpoint(path)
        5. load_checkpoint(path)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainState(total_steps=config.max_steps)
        self._callbacks: list[TrainerCallback] = []

    def add_callback(self, callback: TrainerCallback) -> None:
        """Register a callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: TrainerCallback) -> None:
        """Remove a registered callback."""
        self._callbacks.remove(callback)

    def _fire(self, event: str, **kwargs) -> None:
        """Fire an event on all registered callbacks."""
        for cb in self._callbacks:
            method = getattr(cb, event, None)
            if method is not None:
                method(**kwargs)

    @abstractmethod
    def train(self, dataset, callbacks: Optional[list[TrainerCallback]] = None) -> TrainState:
        """
        Run the training loop.

        Args:
            dataset: The training dataset (concrete type depends on implementation).
            callbacks: Additional callbacks for this run.

        Returns:
            The final TrainState.
        """
        ...

    @abstractmethod
    def evaluate(self, dataset) -> dict:
        """
        Run evaluation on a dataset.

        Returns:
            A dict of evaluation metrics.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save a checkpoint to the given path."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from the given path."""
        ...

    def summary(self) -> str:
        """Human-readable summary of the trainer."""
        return (
            f"Trainer: {self.__class__.__name__}\n"
            f"  Config: {self.config.experiment_name or '(unnamed)'}\n"
            f"  State: {self.state.summary()}\n"
            f"  Callbacks: {len(self._callbacks)}"
        )
