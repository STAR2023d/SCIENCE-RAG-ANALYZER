import pytest

from science_rag.evaluation import EvalCase, evaluate, load_cases
from science_rag.models import Retrieval


class StubRetriever:
    def __init__(self, answers):
        self.answers = answers

    def retrieve(self, question, top_k=None):
        return Retrieval(contexts=self.answers[question][:top_k], sources=[])


def test_hit_rate_and_mrr():
    retriever = StubRetriever(
        {
            "q1": ["irrelevant", "contains CHLOROPHYLL here"],  # found at rank 2
            "q2": ["magma rises"],  # found at rank 1
            "q3": ["nothing useful"],  # missed
        }
    )
    cases = [
        EvalCase("q1", "chlorophyll"),
        EvalCase("q2", "magma"),
        EvalCase("q3", "solid"),
    ]
    metrics = evaluate(retriever, cases, k=5)
    assert metrics["hit_rate@5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_k_limits_what_counts_as_a_hit():
    retriever = StubRetriever({"q": ["no", "no", "target is here"]})
    assert evaluate(retriever, [EvalCase("q", "target")], k=2)["hit_rate@2"] == 0.0
    assert evaluate(retriever, [EvalCase("q", "target")], k=3)["hit_rate@3"] == 1.0


def test_empty_case_list_is_an_error():
    with pytest.raises(ValueError):
        evaluate(StubRetriever({}), [])


def test_load_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "a?", "expected_substring": "b"}\n\n', encoding="utf-8")
    assert load_cases(path) == [EvalCase("a?", "b")]
