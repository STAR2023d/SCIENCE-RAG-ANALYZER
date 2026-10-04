from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(
        self,
        url="http://localhost:6333",
        collection="docs",
        dim=3072,
        client: QdrantClient | None = None,
    ):
        self.client = client if client is not None else QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        """Text search: returns chunk texts and the set of source names."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        contexts = []
        sources = set()

        for r in response.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def search_points(self, query_vector, top_k: int = 5, min_score: float = 0.0):
        """Raw search: returns [{"score": float, "payload": dict}, ...].
        Cosine similarity: higher is closer. min_score drops weak matches."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
        )
        return [{"score": r.score, "payload": r.payload or {}} for r in response.points]