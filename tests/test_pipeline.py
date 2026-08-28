"""
tests/test_pipeline.py — Tests for core.pipeline
==================================================
"""

import pytest

from core.pipeline import (
    Batcher,
    DataSource,
    Pipeline,
    Preprocessor,
)
from core.tokenizer import PlaceholderTokenizer


class SimpleDataSource(DataSource):
    """Test data source."""
    def __init__(self, samples):
        self._samples = samples

    def list_samples(self):
        return list(range(len(self._samples)))

    def load_sample(self, index):
        return self._samples[index]


class SimplePreprocessor(Preprocessor):
    """Uppercases the 'text' field."""
    def process(self, sample):
        result = dict(sample)
        if "text" in result:
            result["text"] = result["text"].upper()
        return result


class SimpleBatcher(Batcher):
    """Groups samples into fixed-size batches."""
    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def batch(self, samples):
        batch = []
        for sample in samples:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield {"samples": batch}
                batch = []
        if batch:
            yield {"samples": batch}


class TestPlaceholderTokenizer:
    def test_encode_decode(self):
        tok = PlaceholderTokenizer(vocab_size=256)
        encoded = tok.encode("hello")
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)
        decoded = tok.decode(encoded)
        assert decoded == "hello"

    def test_vocab_size(self):
        tok = PlaceholderTokenizer(vocab_size=100)
        assert tok.vocab_size() == 100

    def test_callable(self):
        tok = PlaceholderTokenizer(vocab_size=256)
        result = tok("test")
        assert isinstance(result, list)


class TestDataSource:
    def test_list_samples(self):
        source = SimpleDataSource([{"a": 1}, {"a": 2}, {"a": 3}])
        assert source.list_samples() == [0, 1, 2]

    def test_load_sample(self):
        source = SimpleDataSource([{"a": 1}, {"a": 2}])
        assert source.load_sample(0) == {"a": 1}
        assert source.load_sample(1) == {"a": 2}

    def test_len(self):
        source = SimpleDataSource([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(source) == 3

    def test_iteration(self):
        source = SimpleDataSource([{"a": 1}, {"a": 2}])
        samples = list(source)
        assert len(samples) == 2
        assert samples[0] == {"a": 1}


class TestPreprocessor:
    def test_process(self):
        pp = SimplePreprocessor()
        result = pp.process({"text": "hello"})
        assert result["text"] == "HELLO"

    def test_callable(self):
        pp = SimplePreprocessor()
        result = pp({"text": "world"})
        assert result["text"] == "WORLD"

    def test_preserves_other_fields(self):
        pp = SimplePreprocessor()
        result = pp.process({"text": "hello", "label": 1})
        assert result["label"] == 1


class TestBatcher:
    def test_batching(self):
        batcher = SimpleBatcher(batch_size=2)
        samples = [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}]
        batches = list(batcher.batch(iter(samples)))
        assert len(batches) == 3  # [2, 2, 1]
        assert len(batches[0]["samples"]) == 2
        assert len(batches[1]["samples"]) == 2
        assert len(batches[2]["samples"]) == 1

    def test_exact_batch_size(self):
        batcher = SimpleBatcher(batch_size=3)
        samples = [{"i": 0}, {"i": 1}, {"i": 2}]
        batches = list(batcher.batch(iter(samples)))
        assert len(batches) == 1
        assert len(batches[0]["samples"]) == 3


class TestPipeline:
    def test_source_only(self):
        source = SimpleDataSource([{"text": "hello"}, {"text": "world"}])
        pipeline = Pipeline(source=source)
        results = list(pipeline.run())
        assert len(results) == 2
        assert results[0] == {"text": "hello"}

    def test_with_preprocessor(self):
        source = SimpleDataSource([{"text": "hello"}])
        pipeline = Pipeline(
            source=source,
            preprocessor=SimplePreprocessor(),
        )
        results = list(pipeline.run())
        assert results[0]["text"] == "HELLO"

    def test_with_batcher(self):
        source = SimpleDataSource([{"text": "a"}, {"text": "b"}, {"text": "c"}])
        pipeline = Pipeline(
            source=source,
            batcher=SimpleBatcher(batch_size=2),
        )
        batches = list(pipeline.run())
        assert len(batches) == 2
        assert len(batches[0]["samples"]) == 2

    def test_full_pipeline(self):
        source = SimpleDataSource([{"text": "hello"}, {"text": "world"}])
        pipeline = Pipeline(
            source=source,
            preprocessor=SimplePreprocessor(),
            batcher=SimpleBatcher(batch_size=2),
        )
        batches = list(pipeline.run())
        assert len(batches) == 1
        assert batches[0]["samples"][0]["text"] == "HELLO"
        assert batches[0]["samples"][1]["text"] == "WORLD"

    def test_no_source_raises(self):
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="requires a DataSource"):
            list(pipeline.run())

    def test_with_tokenizer(self):
        source = SimpleDataSource([{"text": "hello"}])
        tok = PlaceholderTokenizer(vocab_size=256)
        pipeline = Pipeline(
            source=source,
            tokenizer=tok,
        )
        results = list(pipeline.run())
        assert "input_ids" in results[0]
        assert isinstance(results[0]["input_ids"], list)
