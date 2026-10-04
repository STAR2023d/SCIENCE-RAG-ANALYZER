"""Typed data passed between services and Inngest steps.

Inngest persists every step result as JSON, so step outputs are small, serialisable
Pydantic models (vectors are never part of them)."""

from __future__ import annotations

import pydantic


class ExtractedImage(pydantic.BaseModel):
    path: str
    page: int
    index: int


class ImageHit(pydantic.BaseModel):
    path: str
    caption: str
    source: str
    page: int
    score: float


class Retrieval(pydantic.BaseModel):
    """Everything found for one question."""

    contexts: list[str]
    sources: list[str]
    images: list[ImageHit] = []


class ChunkBatch(pydantic.BaseModel):
    source_id: str
    chunks: list[str]


class ImageList(pydantic.BaseModel):
    source_id: str
    images: list[ExtractedImage]


class CountResult(pydantic.BaseModel):
    count: int
