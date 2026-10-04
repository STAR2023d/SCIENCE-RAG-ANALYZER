"""Thin, typed wrapper around one Qdrant collection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    score: float  # cosine similarity: higher is closer
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_dimension()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        try:  # speeds up delete/filter by source on large collections
            self.client.create_payload_index(
                self.collection, field_name="source", field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:  # noqa: BLE001 - index is an optimisation only
            logger.debug("could not create payload index", exc_info=True)

    def _check_dimension(self) -> None:
        """Fail loudly if the collection was built with a different embedding model."""
        vectors = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(vectors, "size", None)
        if size is not None and size != self.dim:
            raise ValueError(
                f"Collection '{self.collection}' stores {size}-dimensional vectors but the "
                f"embedder produces {self.dim}. Use a new collection name or delete the old one."
            )

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        points = [
            PointStruct(id=ids[i], vector=list(vectors[i]), payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: Sequence[float], limit: int = 5, min_score: float = 0.0) -> list[Hit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            with_payload=True,
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
        )
        return [Hit(score=p.score, payload=p.payload or {}) for p in response.points]

    def delete_by_source(self, source: str) -> None:
        """Remove every point that came from one document (makes re-ingestion clean)."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False
