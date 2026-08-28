"""
tests/test_logging.py — Tests for core.logging
================================================
"""

import json

from core.logging import get_logger


class TestExperimentLogger:
    def test_creation(self):
        log = get_logger("test_log")
        assert log.name == "test_log"
        log.close()

    def test_info_does_not_raise(self, capsys):
        log = get_logger("test")
        log.info("test message", key="value")
        log.close()

    def test_metric_records(self):
        log = get_logger("test")
        log.metric("accuracy", 0.95)
        assert len(log._metrics) == 1
        assert log._metrics[0]["name"] == "accuracy"
        assert log._metrics[0]["value"] == 0.95
        log.close()

    def test_summary(self):
        log = get_logger("test")
        log.metric("p@5", 0.83)
        log.metric("mrr", 0.96)
        summary = log.summary()
        assert summary["name"] == "test"
        assert len(summary["metrics"]) == 2
        log.close()

    def test_section_does_not_raise(self):
        log = get_logger("test")
        log.section("TEST SECTION")
        log.close()

    def test_context_manager(self):
        with get_logger("test") as log:
            log.info("inside context")
        assert log._log_file is None

    def test_jsonl_output(self, tmp_path):
        log = get_logger("test_jsonl", log_dir=tmp_path)
        log.info("test message")
        log.metric("acc", 0.9)
        log.close()

        # Find the JSONL file
        jsonl_files = list(tmp_path.glob("test_jsonl_*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text().strip().split("\n")
        assert len(lines) == 2  # one info + one metric

        entry = json.loads(lines[0])
        assert entry["level"] == "INFO"
        assert entry["message"] == "test message"
        assert entry["name"] == "test_jsonl"

        metric_entry = json.loads(lines[1])
        assert metric_entry["level"] == "METRIC"
