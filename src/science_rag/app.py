"""FastAPI application: serves the Inngest endpoint and a health check.

Run with:  uvicorn science_rag.app:app --reload"""

from __future__ import annotations

import logging

import inngest.fast_api
from fastapi import FastAPI

from science_rag.config import get_settings
from science_rag.service import get_service
from science_rag.workflows import FUNCTIONS, inngest_client

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

app = FastAPI(title="Science RAG Analyzer", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness plus a check that the vector database is reachable."""
    try:
        qdrant_up = get_service().text_store.ping()
    except Exception:  # noqa: BLE001 - health must never raise
        qdrant_up = False
    return {"status": "ok" if qdrant_up else "degraded", "qdrant": "up" if qdrant_up else "down"}


inngest.fast_api.serve(app, inngest_client, FUNCTIONS)
