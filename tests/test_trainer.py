"""
tests/test_trainer.py — Tests for core.trainer
================================================
"""


from core.trainer import (
    CheckpointCallback,
    LoggingCallback,
    Trainer,
    TrainerCallback,
    TrainState,
)
from core.training_config import TrainingConfig


class TestTrainState:
    def test_defaults(self):
        state = TrainState()
        assert state.global_step == 0
        assert state.loss == 0.0
        assert state.progress == 0.0

    def test_progress(self):
        state = TrainState(global_step=50, total_steps=200)
        assert state.progress == 0.25

    def test_progress_pct(self):
        state = TrainState(global_step=75, total_steps=100)
        assert state.progress_pct == "75.0%"

    def test_progress_zero_total(self):
        state = TrainState(global_step=10, total_steps=0)
        assert state.progress == 0.0

    def test_progress_capped(self):
        state = TrainState(global_step=200, total_steps=100)
        assert state.progress == 1.0

    def test_summary(self):
        state = TrainState(global_step=100, total_steps=1000, loss=2.5, learning_rate=1e-4)
        s = state.summary()
        assert "100/1000" in s
        assert "10.0%" in s
        assert "2.5" in s


class TestTrainerCallback:
    def test_default_methods(self):
        """All callback methods should be callable (empty defaults)."""
        cb = TrainerCallback()
        state = TrainState()
        config = TrainingConfig()
        cb.on_train_begin(state, config)
        cb.on_train_end(state)
        cb.on_epoch_begin(state)
        cb.on_epoch_end(state)
        cb.on_train_step(state)
        cb.on_eval_begin(state)
        cb.on_eval_end(state, {"loss": 1.0})
        cb.on_checkpoint_save(state, "/tmp/ckpt")
        cb.on_log(state, {"loss": 1.0})


class TestLoggingCallback:
    def test_creation(self):
        cb = LoggingCallback(log_every=5)
        assert cb.log_every == 5

    def test_on_train_step(self, capsys):
        cb = LoggingCallback(log_every=1)
        state = TrainState(global_step=1, total_steps=10, loss=2.0)
        cb.on_train_step(state)
        captured = capsys.readouterr()
        assert "step=" in captured.err or "step=" in captured.out

    def test_on_train_begin(self, capsys):
        cb = LoggingCallback()
        config = TrainingConfig(experiment_name="test")
        cb.on_train_begin(TrainState(), config)
        captured = capsys.readouterr()
        assert "test" in captured.err or "test" in captured.out


class TestCheckpointCallback:
    def test_creation(self):
        cb = CheckpointCallback(save_every=500)
        assert cb.save_every == 500
        assert cb.checkpoint_dir == "checkpoints"


class ConcreteTrainer(Trainer):
    """Minimal concrete trainer for testing the abstract interface."""

    def train(self, dataset=None, callbacks=None):
        for cb in (callbacks or []):
            self.add_callback(cb)
        self._fire("on_train_begin", state=self.state, config=self.config)
        for step in range(1, 6):
            self.state.global_step = step
            self.state.loss = 5.0 / step
            self._fire("on_train_step", state=self.state)
        self._fire("on_train_end", state=self.state)
        return self.state

    def evaluate(self, dataset=None):
        return {"eval_loss": 1.5}

    def save_checkpoint(self, path=""):
        pass

    def load_checkpoint(self, path=""):
        pass


class TestTrainer:
    def test_creation(self):
        config = TrainingConfig(max_steps=100)
        trainer = ConcreteTrainer(config)
        assert trainer.config.max_steps == 100
        assert trainer.state.total_steps == 100

    def test_add_callback(self):
        config = TrainingConfig()
        trainer = ConcreteTrainer(config)
        cb = LoggingCallback()
        trainer.add_callback(cb)
        assert cb in trainer._callbacks

    def test_remove_callback(self):
        config = TrainingConfig()
        trainer = ConcreteTrainer(config)
        cb = LoggingCallback()
        trainer.add_callback(cb)
        trainer.remove_callback(cb)
        assert cb not in trainer._callbacks

    def test_train_runs(self):
        config = TrainingConfig(max_steps=10)
        trainer = ConcreteTrainer(config)
        final_state = trainer.train()
        assert final_state.global_step == 5

    def test_train_with_callbacks(self, capsys):
        config = TrainingConfig(max_steps=10)
        trainer = ConcreteTrainer(config)
        cb = LoggingCallback(log_every=1)
        trainer.train(callbacks=[cb])
        # Should have logged something
        captured = capsys.readouterr()
        assert len(captured.err + captured.out) > 0

    def test_evaluate(self):
        config = TrainingConfig()
        trainer = ConcreteTrainer(config)
        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] == 1.5

    def test_summary(self):
        config = TrainingConfig(experiment_name="test_exp")
        trainer = ConcreteTrainer(config)
        s = trainer.summary()
        assert "ConcreteTrainer" in s
        assert "test_exp" in s
