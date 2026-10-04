"""Retrieval evaluation: measure instead of guessing.

A case is a question plus a phrase that a correct chunk must contain. We report
hit rate @k (was a correct chunk retrieved?) and MRR (how high was it ranked?)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from science_rag.models import Retrieval


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int | None = None) -> Retrieval: ...


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_substring: str


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read cases from a JSONL file: {"question": "...", "expected_substring": "..."}."""
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(EvalCase(row["question"], row["expected_substring"]))
    return cases


def evaluate(retriever: Retriever, cases: Iterable[EvalCase], k: int = 5) -> dict[str, float]:
    cases = list(cases)
    if not cases:
        raise ValueError("no evaluation cases given")
    hits = 0
    reciprocal_ranks = 0.0
    for case in cases:
        contexts = retriever.retrieve(case.question, top_k=k).contexts
        needle = case.expected_substring.lower()
        rank = next((i for i, c in enumerate(contexts) if needle in c.lower()), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / (rank + 1)
    n = len(cases)
    return {"cases": float(n), f"hit_rate@{k}": hits / n, "mrr": reciprocal_ranks / n}
