"""Ingest PDFs from the command line, without Inngest (useful for bulk loading and debugging).

uv run python scripts/ingest.py path/to/file.pdf
uv run python scripts/ingest.py path/to/folder --images-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from science_rag.config import get_settings
from science_rag.service import RagService, get_service


def ingest_one(service: RagService, pdf: Path, text: bool, images: bool, pause: int) -> str:
    parts = []
    if text:
        chunks = service.chunk_pdf(pdf)
        parts.append(f"{service.store_chunks(chunks, pdf.name)} chunks")
    if images:
        found = service.extract_images(pdf, pdf.name)
        stored = 0
        for n, img in enumerate(found, start=1):
            print(f"    image {n}/{len(found)} (page {img.page})", flush=True)
            stored += service.store_image(img.path, pdf.name, img.page, img.index)
            time.sleep(pause)
        parts.append(f"{stored}/{len(found)} images")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", type=Path, help="a PDF file or a folder of PDFs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text-only", action="store_true")
    group.add_argument("--images-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = args.path.expanduser().resolve()
    pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not pdfs or not all(p.is_file() for p in pdfs):
        print(f"No PDF found at {target}", file=sys.stderr)
        return 1

    service = get_service()
    pause = get_settings().image_pause_seconds
    failures = 0
    for pdf in pdfs:
        print(f"Ingesting {pdf.name}")
        try:
            summary = ingest_one(service, pdf, not args.images_only, not args.text_only, pause)
            print(f"  OK: {summary}")
        except Exception as exc:  # noqa: BLE001 - report and continue with the next file
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
