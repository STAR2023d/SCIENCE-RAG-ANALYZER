"""Framework-agnostic business logic.

RagService knows nothing about Inngest, FastAPI or Streamlit. It receives its
collaborators (embedder, captioner, vector stores) through the constructor, which keeps
the pipeline unit-testable with fakes and lets any provider be swapped in one place."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient

from science_rag.config import Settings, get_settings
from science_rag.embeddings import Embedder, GeminiEmbedder
from science_rag.models import ExtractedImage, ImageHit, Retrieval
from science_rag.pdf_images import extract_pdf_images, slugify
from science_rag.pdf_text import load_and_chunk_pdf
from science_rag.vector_store import VectorStore
from science_rag.vision import Captioner, GeminiCaptioner, is_decorative

logger = logging.getLogger(__name__)


class NoTextExtractedError(ValueError):
    """The PDF has no text layer (probably a scan) and needs OCR first."""


def point_id(kind: str, source_id: str, *parts: object) -> str:
    """Deterministic ID: re-ingesting the same document overwrites instead of duplicating."""
    key = ":".join([kind, source_id, *map(str, parts)])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        captioner: Captioner,
        text_store: VectorStore,
        image_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.captioner = captioner
        self.text_store = text_store
        self.image_store = image_store

    # ------------------------------------------------------------------ text ingestion
    def chunk_pdf(self, pdf_path: str | Path) -> list[str]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        chunks = load_and_chunk_pdf(path, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise NoTextExtractedError(
                f"No text extracted from {path.name} (scanned PDF? run OCR first)"
            )
        return chunks

    def store_chunks(self, chunks: list[str], source_id: str) -> int:
        """Embed and store chunks. Embedding happens first, so a provider failure never
        leaves the document half-deleted."""
        vectors = self.embedder.embed(chunks)
        ids = [point_id("txt", source_id, i) for i in range(len(chunks))]
        payloads = [
            {"source": source_id, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)
        ]
        self.text_store.delete_by_source(source_id)  # drop stale chunks from older versions
        self.text_store.upsert(ids, vectors, payloads)
        logger.info("stored %d text chunks for %s", len(chunks), source_id)
        return len(chunks)

    # ----------------------------------------------------------------- image ingestion
    def extract_images(self, pdf_path: str | Path, source_id: str) -> list[ExtractedImage]:
        """Save a PDF's pictures to disk and clear that document's old image records."""
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        self.image_store.delete_by_source(source_id)
        s = self.settings
        return extract_pdf_images(
            path,
            s.images_dir.resolve() / slugify(source_id),
            min_side_px=s.min_image_side_px,
            max_repeat_pages=s.max_image_repeat_pages,
            render_drawing_pages=s.render_drawing_pages,
            drawing_threshold=s.drawing_threshold,
            render_dpi=s.render_dpi,
        )

    def store_image(
        self, image_path: str | Path, source_id: str, page: int = 0, index: int = 0
    ) -> bool:
        """Describe one image, embed the description and store it.
        Returns False when the vision model judged the image decorative."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        caption = self.captioner.caption(path)
        if is_decorative(caption):
            return False
        vector = self.embedder.embed([caption])[0]
        self.image_store.upsert(
            [point_id("img", source_id, page, index)],
            [vector],
            [
                {
                    "type": "image",
                    "source": source_id,
                    "page": page,
                    "index": index,
                    "path": str(path.resolve()),
                    "caption": caption,
                }
            ],
        )
        return True

    # --------------------------------------------------------------------- retrieval
    def retrieve(self, question: str, top_k: int | None = None) -> Retrieval:
        """Embed the question once and search both collections."""
        s = self.settings
        vector = self.embedder.embed([question])[0]

        contexts: list[str] = []
        sources: list[str] = []
        for hit in self.text_store.search(vector, limit=top_k or s.top_k):
            text = hit.payload.get("text", "")
            if not text:
                continue
            contexts.append(text)
            source = hit.payload.get("source", "")
            if source and source not in sources:
                sources.append(source)

        images: list[ImageHit] = []
        for hit in self.image_store.search(
            vector, limit=s.image_top_k, min_score=s.min_image_score
        ):
            p = hit.payload
            path = p.get("path", "")
            if not path or not Path(path).is_file():  # file deleted or moved
                continue
            images.append(
                ImageHit(
                    path=path,
                    caption=p.get("caption", ""),
                    source=p.get("source", ""),
                    page=int(p.get("page", 0)),
                    score=float(hit.score),
                )
            )
        return Retrieval(contexts=contexts, sources=sources, images=images)


def build_service(settings: Settings | None = None) -> RagService:
    settings = settings or get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    embedder = GeminiEmbedder(settings)
    return RagService(
        settings=settings,
        embedder=embedder,
        captioner=GeminiCaptioner(settings),
        text_store=VectorStore(client, settings.text_collection, embedder.dim),
        image_store=VectorStore(client, settings.image_collection, embedder.dim),
    )


@lru_cache(maxsize=1)
def get_service() -> RagService:
    """Lazily built singleton (nothing connects at import time)."""
    return build_service()
