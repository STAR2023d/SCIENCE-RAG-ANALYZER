"""Prompt construction: pure functions, easy to unit test."""

from __future__ import annotations

from science_rag.models import ImageHit, Retrieval

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context does not contain the answer, say you could not find it "
    "in the provided notes. Do not use outside knowledge. "
    "Figures related to the question may be listed after the context. "
    "They are displayed to the user next to your answer, so you may refer to them "
    "(for example 'see the figure below'), but only describe what their listed "
    "descriptions say."
)


def format_figures(images: list[ImageHit]) -> str:
    if not images:
        return ""
    lines = [
        f"- [Figure from {h.source}" + (f", page {h.page}" if h.page else "") + f"] {h.caption}"
        for h in images
    ]
    return "\n\nRelated figures (shown to the user):\n" + "\n".join(lines)


def build_user_prompt(question: str, retrieval: Retrieval) -> str:
    context_block = "\n\n".join(f"- {c}" for c in retrieval.contexts)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}{format_figures(retrieval.images)}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )


def build_messages(question: str, retrieval: Retrieval) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, retrieval)},
    ]
