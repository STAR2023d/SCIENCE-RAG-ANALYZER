"""Image ingestion: extract images from PDFs, describe them with a vision model,
embed the description, and store it in the 'images' Qdrant collection.

Why captions? Search is text-to-text: a vision model turns each picture into a rich
description (type, labels, axes, concept), we embed that text with the SAME embedder
used for your notes, and a student's question then finds the matching figure.
"""
import base64
import hashlib
import io
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pymupdf  # PyMuPDF
from PIL import Image
from openai import OpenAI, RateLimitError, InternalServerError
from dotenv import load_dotenv

from data_loader import embed_texts, EMBED_DIM
from vector_db import QdrantStorage

load_dotenv()

# ---------------- settings ----------------
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VISION_MODEL = "gemini-3.8-flash"     # must accept images; verify with test_vision.py
IMAGE_COLLECTION = "images"
IMAGES_DIR = Path("images").resolve()  # extracted pictures are saved here

MIN_SIDE_PX = 120          # skip icons / bullets smaller than this
MAX_REPEAT_PAGES = 2       # an image on more than this many pages = logo/watermark, skip
RENDER_DRAWING_PAGES = False  # True: also render pages whose graphs are vector drawings
DRAWING_THRESHOLD = 40     # vector paths needed before a page counts as "has a drawing"
RENDER_DPI = 110

vision_client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url=GEMINI_BASE_URL,
    max_retries=0,
)

CAPTION_PROMPT = (
    "You are indexing an image from a school science textbook so that students can find "
    "it by searching. Describe it in 80 to 150 words of plain text. Start with its type "
    "(diagram, graph, chart, table, photograph, drawing or map). Then state the topic, "
    "transcribe every visible title, label and axis name with units, describe any trend "
    "or relationship shown, and say which concept it teaches. "
    "If the image is only a logo, decoration, border or blank, reply with exactly: DECORATIVE"
)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem) or "doc"


# ---------------- extraction ----------------
def extract_pdf_images(pdf_path: str, source_id: str) -> list[dict]:
    """Save the useful images of a PDF as PNG files.
    Returns [{"path": str, "page": int, "index": int}, ...]."""
    out_dir = IMAGES_DIR / _slug(source_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        found = []  # (page_no, idx, digest, png_bytes)
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
                    if pix.width < MIN_SIDE_PX or pix.height < MIN_SIDE_PX:
                        continue
                    if pix.n - pix.alpha >= 4:          # CMYK etc. -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.alpha:                        # drop transparency
                        pix = pymupdf.Pixmap(pix, 0)
                    data = pix.tobytes("png")
                except Exception:
                    continue                             # unreadable image: skip it
                digest = hashlib.sha1(data).hexdigest()
                pages_by_digest[digest].add(page_no)
                found.append((page_no, idx, digest, data))

        results: list[dict] = []
        pages_with_images: set[int] = set()
        for page_no, idx, digest, data in found:
            if len(pages_by_digest[digest]) > MAX_REPEAT_PAGES:
                continue                                 # repeated logo / watermark
            path = out_dir / f"p{page_no:03d}_i{idx}.png"
            path.write_bytes(data)
            results.append({"path": str(path), "page": page_no, "index": idx})
            pages_with_images.add(page_no)

        if RENDER_DRAWING_PAGES:
            for page_no, page in enumerate(doc.pages(), start=1):
                if page_no in pages_with_images:
                    continue
                if len(page.get_drawings()) >= DRAWING_THRESHOLD:
                    path = out_dir / f"p{page_no:03d}_render.png"
                    page.get_pixmap(dpi=RENDER_DPI).save(str(path))
                    results.append({"path": str(path), "page": page_no, "index": 99})

        results.sort(key=lambda r: (r["page"], r["index"]))
        return results
    finally:
        doc.close()


# ---------------- captioning ----------------
def _image_to_data_url(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1600, 1600))                       # keep requests small
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def caption_image(path: str) -> str:
    """Ask the vision model to describe one image. Returns text, or 'DECORATIVE'."""
    data_url = _image_to_data_url(path)
    for attempt in range(4):
        try:
            resp = vision_client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=1024,   # headroom in case the model "thinks" first
                temperature=0.2,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("vision model returned an empty caption")
            return text
        except (RateLimitError, InternalServerError):
            time.sleep(15 * (attempt + 1))               # 429 / 503: wait, retry
    raise RuntimeError("vision model unavailable: gave up after 4 attempts")


# ---------------- store ----------------
def caption_embed_store(image_path: str, source_id: str, page: int, index: int) -> str | None:
    """Caption one image, embed the caption, save it to Qdrant.
    Returns the caption, or None if the image was judged decorative."""
    caption = caption_image(image_path)
    if caption.strip().upper().startswith("DECORATIVE"):
        return None
    vec = embed_texts([caption])[0]
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"img:{source_id}:{page}:{index}"))
    store = QdrantStorage(collection=IMAGE_COLLECTION, dim=EMBED_DIM)
    store.upsert(
        [point_id],
        [vec],
        [{
            "type": "image",
            "source": source_id,
            "page": page,
            "index": index,
            "path": str(Path(image_path).resolve()),
            "caption": caption,
        }],
    )
    return caption