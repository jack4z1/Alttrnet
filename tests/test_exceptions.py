"""
tests/test_exceptions.py — Tests for core.exceptions
======================================================
"""


from core.exceptions import (
    AlttrnetError,
    CheckpointError,
    ConfigError,
    DatasetError,
    DatasetManifestError,
    DatasetValidationError,
    EnvError,
    ExperimentError,
    PipelineError,
    RetrievalError,
    TokenizerError,
    TrainingConfigError,
    TrainingError,
    ValidationError,
)


class TestAlttrnetError:
    def test_message(self):
        err = AlttrnetError("something broke")
        assert str(err) == "something broke"

    def test_with_details(self):
        err = AlttrnetError("error", details={"key": "val"})
        assert "key='val'" in str(err)

    def test_is_exception(self):
        assert issubclass(AlttrnetError, Exception)


class TestExceptionHierarchy:
    def test_config_error(self):
        assert issubclass(ConfigError, AlttrnetError)

    def test_dataset_error(self):
        assert issubclass(DatasetError, AlttrnetError)

    def test_dataset_manifest_error(self):
        assert issubclass(DatasetManifestError, DatasetError)

    def test_dataset_validation_error(self):
        assert issubclass(DatasetValidationError, DatasetError)

    def test_checkpoint_error(self):
        assert issubclass(CheckpointError, AlttrnetError)

    def test_training_error(self):
        assert issubclass(TrainingError, AlttrnetError)

    def test_training_config_error(self):
        assert issubclass(TrainingConfigError, TrainingError)
        assert issubclass(TrainingConfigError, ConfigError)

    def test_pipeline_error(self):
        assert issubclass(PipelineError, AlttrnetError)

    def test_tokenizer_error(self):
        assert issubclass(TokenizerError, PipelineError)

    def test_retrieval_error(self):
        assert issubclass(RetrievalError, AlttrnetError)

    def test_validation_error(self):
        assert issubclass(ValidationError, AlttrnetError)

    def test_experiment_error(self):
        assert issubclass(ExperimentError, AlttrnetError)

    def test_env_error(self):
        assert issubclass(EnvError, AlttrnetError)


class TestValidationError:
    def test_field_info(self):
        err = ValidationError("bad value", field="lr", value=-1.0)
        s = str(err)
        assert "bad value" in s
        assert "lr" in s
        assert "-1.0" in s

    def test_no_field(self):
        err = ValidationError("bad value")
        s = str(err)
        assert "bad value" in s
        assert "field=" not in s
