"""Text embeddings. The rest of the code depends only on the Embedder protocol, so a
different provider (or a fake in tests) can be swapped in without touching the pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import InternalServerError, OpenAI, RateLimitError

from science_rag.config import Settings
from science_rag.retry import with_backoff


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """Gemini embeddings through its OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings, batch_size: int = 20) -> None:
        self.dim = settings.embed_dim
        self._model = settings.embed_model
        self._batch_size = batch_size
        # max_retries=0: the SDK's instant retries only burn quota; we back off ourselves
        self._client = OpenAI(
            api_key=settings.require_gemini_key(),
            base_url=settings.gemini_base_url,
            max_retries=0,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = with_backoff(
                lambda b=batch: self._client.embeddings.create(model=self._model, input=b),
                retry_on=(RateLimitError, InternalServerError),
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors
