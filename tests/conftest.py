"""Shared fixtures: a fake embedder/captioner and an in-memory Qdrant, so the whole
pipeline is tested offline, fast, and without API keys."""

from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient
from reportlab.pdfgen import canvas

from science_rag.config import Settings
from science_rag.service import RagService
from science_rag.vector_store import VectorStore

DIM = 256


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    dim = DIM

    def embed(self, texts):
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            if len(word) > 3:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        if norm == 1.0 and not any(vec):
            vec[0] = 1.0
        return [x / norm for x in vec]


class FakeCaptioner:
    """Returns a canned caption chosen by the image file name."""

    def __init__(self, captions: dict[str, str] | None = None, default: str = "DECORATIVE"):
        self.captions = captions or {}
        self.default = default
        self.calls = 0

    def caption(self, image_path) -> str:
        self.calls += 1
        name = Path(image_path).name
        return next((c for key, c in self.captions.items() if key in name), self.default)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embed_dim=DIM,
        chunk_size=60,
        chunk_overlap=5,
        min_image_score=0.3,
        images_dir=tmp_path / "images",
        uploads_dir=tmp_path / "uploads",
    )


@pytest.fixture
def qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def make_service(settings, qdrant):
    def _make(captioner: FakeCaptioner | None = None) -> RagService:
        return RagService(
            settings=settings,
            embedder=FakeEmbedder(),
            captioner=captioner or FakeCaptioner(),
            text_store=VectorStore(qdrant, "docs", DIM),
            image_store=VectorStore(qdrant, "images", DIM),
        )

    return _make


def _picture(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    rng = random.Random(str(path))
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for _ in range(80):  # noise, so every picture has distinct bytes
        x, y = rng.randint(0, size[0]), rng.randint(0, size[1])
        draw.ellipse([x, y, x + 15, y + 15], fill=(rng.randint(0, 255),) * 3)
    image.save(path)
    return path


@pytest.fixture
def text_pdf(tmp_path) -> Path:
    """Three pages on three topics."""
    path = tmp_path / "notes.pdf"
    pages = [
        "Photosynthesis happens in chloroplasts. Chlorophyll absorbs sunlight and plants "
        "convert carbon dioxide and water into glucose and oxygen.",
        "A volcano erupts when magma rises through the crust. Lava and ash are released "
        "and the magma chamber empties during an eruption.",
        "Matter exists as a solid, a liquid or a gas. Heating a solid causes melting and "
        "heating a liquid causes evaporation into a gas.",
    ]
    c = canvas.Canvas(str(path))
    for text in pages:
        t = c.beginText(72, 760)
        words = text.split()
        for i in range(0, len(words), 9):
            t.textLine(" ".join(words[i : i + 9]))
        c.drawText(t)
        c.showPage()
    c.save()
    return path


@pytest.fixture
def image_pdf(tmp_path) -> Path:
    """Page 1: water picture. Page 2: plant picture. Every page: the same logo and a tiny icon."""
    water = _picture(tmp_path / "water.png", (400, 300), (30, 90, 200))
    plant = _picture(tmp_path / "plant.png", (400, 300), (30, 160, 60))
    logo = _picture(tmp_path / "logo.png", (200, 200), (200, 30, 30))
    icon = _picture(tmp_path / "icon.png", (50, 50), (0, 0, 0))

    path = tmp_path / "figures.pdf"
    c = canvas.Canvas(str(path))
    for page in (1, 2, 3):
        c.drawString(72, 800, f"Page {page}")
        c.drawImage(str(logo), 450, 700, 80, 80)
        c.drawImage(str(icon), 72, 700, 20, 20)
        if page == 1:
            c.drawImage(str(water), 72, 400, 300, 225)
        if page == 2:
            c.drawImage(str(plant), 72, 400, 300, 225)
        c.showPage()
    c.save()
    return path


@pytest.fixture
def blank_pdf(tmp_path) -> Path:
    path = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(path))
    c.showPage()
    c.save()
    return path
