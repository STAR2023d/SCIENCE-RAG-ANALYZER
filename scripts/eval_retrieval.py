"""Measure retrieval quality on your own question set.

    uv run python scripts/eval_retrieval.py eval/cases.jsonl -k 5

Each line of the file: {"question": "...", "expected_substring": "text a correct chunk contains"}
"""

from __future__ import annotations

import argparse

from science_rag.evaluation import evaluate, load_cases
from science_rag.service import get_service

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("cases", help="JSONL file of evaluation cases")
parser.add_argument("-k", type=int, default=5)
args = parser.parse_args()

metrics = evaluate(get_service(), load_cases(args.cases), k=args.k)
for name, value in metrics.items():
    print(f"{name:>12}: {value:.3f}" if name != "cases" else f"{name:>12}: {int(value)}")
