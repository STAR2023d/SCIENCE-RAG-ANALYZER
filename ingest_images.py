"""Ingest the pictures of PDFs WITHOUT Inngest (handy for testing and bulk loading).

Usage:
    uv run python ingest_images.py "/path/to/file.pdf"
    uv run python ingest_images.py "/path/to/folder"
"""
import sys
import time
from pathlib import Path

from image_loader import extract_pdf_images, caption_embed_store

PAUSE_SECONDS = 4


def ingest_pdf_images(pdf: Path) -> str:
    items = extract_pdf_images(str(pdf), pdf.name)
    print(f"  found {len(items)} usable images")
    stored = 0
    for n, it in enumerate(items, start=1):
        print(f"  [{n}/{len(items)}] page {it['page']} ...", end=" ", flush=True)
        caption = caption_embed_store(it["path"], pdf.name, it["page"], it["index"])
        if caption is None:
            print("decorative, skipped")
        else:
            stored += 1
            print("stored:", caption[:70].replace("\n", " "))
        time.sleep(PAUSE_SECONDS)
    return f"{stored} stored of {len(items)} found"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python ingest_images.py "/path/to/file.pdf or folder"')
        sys.exit(1)

    target = Path(sys.argv[1]).expanduser().resolve()
    if target.is_dir():
        pdfs = sorted(p for p in target.iterdir() if p.suffix.lower() == ".pdf")
    elif target.is_file():
        pdfs = [target]
    else:
        print(f"ERROR: not found: {target}")
        sys.exit(1)

    summary = []
    for pdf in pdfs:
        print(f"Processing {pdf.name}")
        try:
            summary.append((pdf.name, ingest_pdf_images(pdf)))
        except Exception as e:
            summary.append((pdf.name, f"FAILED: {type(e).__name__}: {e}"))

    print("\n=== Summary ===")
    for name, status in summary:
        print(f"{name[:60]:60} {status}")


if __name__ == "__main__":
    main()