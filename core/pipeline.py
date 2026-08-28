"""
core/pipeline.py — ALTTRNET data pipeline interfaces
======================================================
Abstract base classes and protocols for the data pipeline. This defines
the contract that concrete implementations must follow without choosing
specific data formats, tokenizers, or batching strategies.

The pipeline is designed as a sequence of composable stages:
    Source -> Loader -> Preprocessor -> Tokenizer -> Batcher -> Iterator

Each stage is an abstract class that can be subclassed for specific
implementations.

Usage:
    from core.pipeline import (
        DataSource,
        DataLoader,
        Preprocessor,
        Tokenizer,
        Batcher,
    )

    # Concrete implementations subclass these
    class MyDataSource(DataSource):
        def list_samples(self):
            return [...]
        def load_sample(self, idx):
            return {...}
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """
    Abstract base for data sources.

    A data source represents a collection of samples that can be
    enumerated and loaded individually.
    """

    @abstractmethod
    def list_samples(self) -> list[Any]:
        """Return a list of sample identifiers / indices."""
        ...

    @abstractmethod
    def load_sample(self, index: Any) -> dict:
        """Load a single sample by its identifier."""
        ...

    def __len__(self) -> int:
        """Number of samples in this source."""
        return len(self.list_samples())

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all samples."""
        for idx in self.list_samples():
            yield self.load_sample(idx)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader(ABC):
    """
    Abstract base for data loaders.

    A loader takes a data source and produces an iterator of batches.
    """

    @abstractmethod
    def load(self, source: DataSource) -> Iterator[dict]:
        """
        Load data from a source and yield batches.

        Each batch is a dict with at least a 'input_ids' key.
        """
        ...


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor(ABC):
    """
    Abstract base for data preprocessors.

    A preprocessor transforms raw samples into a normalized form
    before tokenization.
    """

    @abstractmethod
    def process(self, sample: dict) -> dict:
        """
        Process a single sample.

        Input: raw sample dict from a DataSource.
        Output: normalized sample dict ready for tokenization.
        """
        ...

    def __call__(self, sample: dict) -> dict:
        return self.process(sample)


# ---------------------------------------------------------------------------
# Tokenizer interface
# ---------------------------------------------------------------------------

class Tokenizer(ABC):
    """
    Abstract base for tokenizers.

    A tokenizer converts text into integer token sequences. Concrete
    implementations will wrap specific tokenizer libraries.
    """

    @abstractmethod
    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode text into token IDs."""
        ...

    @abstractmethod
    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode token IDs back to text."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    def __call__(self, text: str, **kwargs) -> list[int]:
        return self.encode(text, **kwargs)

    def tokenize(self, text: str) -> list[str]:
        """
        Split text into subword tokens (strings, not IDs).

        Default implementation is not provided — subclasses must
        implement if needed.
        """
        raise NotImplementedError(
            "tokenize() not implemented by this tokenizer"
        )


# ---------------------------------------------------------------------------
# Batcher
# ---------------------------------------------------------------------------

class Batcher(ABC):
    """
    Abstract base for batchers.

    A batcher takes a stream of preprocessed samples and groups them
    into batches suitable for model input.
    """

    @abstractmethod
    def batch(self, samples: Iterator[dict]) -> Iterator[dict]:
        """
        Group samples into batches.

        Each yielded dict should contain batched tensors/arrays.
        """
        ...

    def __call__(self, samples: Iterator[dict]) -> Iterator[dict]:
        return self.batch(samples)


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    A composable data pipeline that chains Source -> Loader -> Preprocessor
    -> Tokenizer -> Batcher stages.

    Example:
        pipeline = Pipeline(
            source=my_source,
            loader=my_loader,
            preprocessor=my_preprocessor,
            tokenizer=my_tokenizer,
            batcher=my_batcher,
        )
        for batch in pipeline.run():
            train_step(batch)
    """

    def __init__(
        self,
        *,
        source: Optional[DataSource] = None,
        loader: Optional[DataLoader] = None,
        preprocessor: Optional[Preprocessor] = None,
        tokenizer: Optional[Tokenizer] = None,
        batcher: Optional[Batcher] = None,
    ):
        self.source = source
        self.loader = loader
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.batcher = batcher

    def run(self) -> Iterator[dict]:
        """
        Execute the full pipeline and yield batches.

        Stages are composed: each stage's output feeds into the next.
        Missing stages are skipped (pass-through).
        """
        if self.source is None:
            raise ValueError("Pipeline requires a DataSource")

        # Stage 1: Load
        if self.loader is not None:
            samples = self.loader.load(self.source)
        else:
            samples = iter(self.source)

        # Stage 2: Preprocess
        if self.preprocessor is not None:
            samples = self._apply_preprocessor(samples)

        # Stage 3: Tokenize
        if self.tokenizer is not None:
            samples = self._apply_tokenizer(samples)

        # Stage 4: Batch
        if self.batcher is not None:
            yield from self.batcher.batch(samples)
        else:
            # No batcher — yield individual samples
            yield from samples

    def _apply_preprocessor(self, samples: Iterator[dict]) -> Iterator[dict]:
        for sample in samples:
            yield self.preprocessor.process(sample)

    def _apply_tokenizer(self, samples: Iterator[dict]) -> Iterator[dict]:
        for sample in samples:
            yield self._tokenize_sample(sample)

    def _tokenize_sample(self, sample: dict) -> dict:
        """Tokenize the text fields of a sample."""
        result = dict(sample)
        # Tokenize 'text' field if present
        if "text" in result:
            result["input_ids"] = self.tokenizer.encode(result["text"])
        # Tokenize 'input' field if present (instruction datasets)
        if "input" in result and result["input"]:
            result["input_ids"] = self.tokenizer.encode(result["input"])
        # Tokenize 'output' field if present
        if "output" in result:
            result["labels"] = self.tokenizer.encode(result["output"])
        return result
