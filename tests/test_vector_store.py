import pytest
from qdrant_client import QdrantClient

from science_rag.vector_store import VectorStore


@pytest.fixture
def store():
    return VectorStore(QdrantClient(":memory:"), "t", dim=3)


def _fill(store):
    store.upsert(
        [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
        ],
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]],
        [
            {"source": "a.pdf", "text": "x"},
            {"source": "a.pdf", "text": "y"},
            {"source": "b.pdf", "text": "z"},
        ],
    )


def test_search_orders_by_similarity(store):
    _fill(store)
    hits = store.search([1, 0, 0], limit=3)
    assert [h.payload["text"] for h in hits] == ["x", "y", "z"]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_min_score_filters_weak_matches(store):
    _fill(store)
    hits = store.search([1, 0, 0], limit=3, min_score=0.8)
    assert {h.payload["text"] for h in hits} == {"x", "y"}


def test_delete_by_source_removes_only_that_document(store):
    _fill(store)
    store.delete_by_source("a.pdf")
    assert store.count() == 1
    assert store.search([0, 1, 0])[0].payload["source"] == "b.pdf"


def test_upsert_same_id_overwrites(store):
    _fill(store)
    store.upsert(
        ["00000000-0000-0000-0000-000000000001"], [[1, 0, 0]], [{"source": "a.pdf", "text": "new"}]
    )
    assert store.count() == 3


def test_dimension_mismatch_fails_loudly():
    client = QdrantClient(":memory:")
    VectorStore(client, "t", dim=3)
    with pytest.raises(ValueError, match="3-dimensional"):
        VectorStore(client, "t", dim=8)


def test_ping(store):
    assert store.ping() is True
