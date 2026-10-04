"""Run retrieval from the command line and print what was found (no LLM involved).

uv run python scripts/query.py "how does the water cycle work"
"""

from __future__ import annotations

import sys

from science_rag.service import get_service

question = " ".join(sys.argv[1:]).strip()
if not question:
    raise SystemExit('Usage: python scripts/query.py "your question"')

result = get_service().retrieve(question)
print(f"\nSources: {', '.join(result.sources) or '(none)'}")
for i, context in enumerate(result.contexts, start=1):
    print(f"\n[{i}] {context[:300].replace(chr(10), ' ')}")
print("\nFigures:")
for fig in result.images:
    print(f"  {fig.score:.3f} | {fig.source} p{fig.page} | {fig.caption[:90]}")
if not result.images:
    print("  (none above the score threshold)")
