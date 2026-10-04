"""Extract the useful pictures from a PDF and save them as PNG files."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

from science_rag.models import ExtractedImage


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem) or "doc"


def extract_pdf_images(
    pdf_path: str | Path,
    out_dir: Path,
    *,
    min_side_px: int = 120,
    max_repeat_pages: int = 2,
    render_drawing_pages: bool = False,
    drawing_threshold: int = 40,
    render_dpi: int = 110,
) -> list[ExtractedImage]:
    """Save the embedded images of a PDF into out_dir.

    Filters out icons (smaller than min_side_px) and images repeated on more than
    max_repeat_pages pages (logos, watermarks). With render_drawing_pages=True, pages
    that hold no embedded image but many vector paths (graphs drawn as shapes) are
    rendered to a PNG instead."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    try:
        found: list[tuple[int, int, str, bytes]] = []  # (page, index, digest, png bytes)
        pages_by_digest: dict[str, set[int]] = defaultdict(set)

        for page_no, page in enumerate(doc.pages(), start=1):
            seen_xrefs: set[int] = set()
            for idx, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < min_side_px or pix.height < min_side_px:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK and friends -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.alpha:  # drop transparency
                        pix = pymupdf.Pixmap(pix, 0)
                    data = pix.tobytes("png")
                except Exception:  # noqa: BLE001 - one unreadable image must not stop the PDF
                    continue
                digest = hashlib.sha1(data).hexdigest()  # noqa: S324 - dedup key, not security
                pages_by_digest[digest].add(page_no)
                found.append((page_no, idx, digest, data))

        results: list[ExtractedImage] = []
        pages_with_images: set[int] = set()
        for page_no, idx, digest, data in found:
            if len(pages_by_digest[digest]) > max_repeat_pages:
                continue
            path = out_dir / f"p{page_no:03d}_i{idx}.png"
            path.write_bytes(data)
            results.append(ExtractedImage(path=str(path), page=page_no, index=idx))
            pages_with_images.add(page_no)

        if render_drawing_pages:
            for page_no, page in enumerate(doc.pages(), start=1):
                if page_no in pages_with_images:
                    continue
                if len(page.get_drawings()) >= drawing_threshold:
                    path = out_dir / f"p{page_no:03d}_render.png"
                    page.get_pixmap(dpi=render_dpi).save(str(path))
                    results.append(ExtractedImage(path=str(path), page=page_no, index=99))

        results.sort(key=lambda r: (r.page, r.index))
        return results
    finally:
        doc.close()
