"""PDF text extraction and chunking."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader


@lru_cache(maxsize=8)
def _splitter(chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_and_chunk_pdf(path: str | Path, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Read a PDF page by page and split the text into overlapping, sentence-aware chunks
    (sizes are in tokens). Returns [] for PDFs without a text layer (scans)."""
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks: list[str] = []
    for doc in PDFReader().load_data(file=Path(path)):
        text = getattr(doc, "text", "") or ""
        if text.strip():
            chunks.extend(splitter.split_text(text))
    return chunks
