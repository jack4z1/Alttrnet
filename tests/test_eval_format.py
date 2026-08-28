"""
tests/test_eval_format.py — Tests for core.eval_format
=======================================================
"""



from core.eval_format import EvalResult, EvalSuite


class TestEvalResult:
    def test_creation(self):
        r = EvalResult(name="dense_baseline")
        assert r.name == "dense_baseline"
        assert r.metrics == {}
        assert r.per_question == []

    def test_add_metric(self):
        r = EvalResult(name="test")
        r.add_metric("p@5", 0.93)
        assert r.metrics["p@5"] == 0.93

    def test_add_per_question(self):
        r = EvalResult(name="test")
        r.add_per_question(1, "What is Python?", {"p@5": 1.0})
        assert len(r.per_question) == 1
        assert r.per_question[0]["qnum"] == 1

    def test_to_dict(self):
        r = EvalResult(name="test", metrics={"acc": 0.9})
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["metrics"]["acc"] == 0.9

    def test_timestamp_set(self):
        r = EvalResult(name="test")
        assert r.timestamp  # non-empty


class TestEvalSuite:
    def test_creation(self):
        suite = EvalSuite(name="test_suite")
        assert suite.name == "test_suite"
        assert suite.results == []

    def test_add_result(self):
        suite = EvalSuite(name="test")
        r = EvalResult(name="dense", metrics={"p@5": 0.83})
        suite.add(r)
        assert len(suite.results) == 1

    def test_comparison_table(self):
        suite = EvalSuite(name="comparison")
        suite.add(EvalResult(name="dense", metrics={"p@5": 0.83, "mrr": 0.96}))
        suite.add(EvalResult(name="hybrid", metrics={"p@5": 0.87, "mrr": 0.97}))
        table = suite.comparison_table()
        assert "dense" in table
        assert "hybrid" in table
        assert "p@5" in table

    def test_comparison_table_empty(self):
        suite = EvalSuite(name="empty")
        assert suite.comparison_table() == "(no results)"

    def test_save_and_load(self, tmp_path):
        suite = EvalSuite(name="test_suite", description="Test")
        suite.add(EvalResult(name="dense", metrics={"p@5": 0.83}))
        suite.add(EvalResult(name="hybrid", metrics={"p@5": 0.87}))

        path = tmp_path / "results.json"
        suite.save(path)
        assert path.exists()

        loaded = EvalSuite.load(path)
        assert loaded.name == "test_suite"
        assert loaded.description == "Test"
        assert len(loaded.results) == 2
        assert loaded.results[0].name == "dense"
        assert loaded.results[1].metrics["p@5"] == 0.87
