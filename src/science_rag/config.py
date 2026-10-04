"""Central configuration. Everything tunable lives here and can be set via environment
variables (or a .env file). Nothing in this module talks to the network, and API keys
are only required at the moment they are actually used."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when a required setting (for example an API key) is missing."""


def _env[T](name: str, default: T, cast: Callable[[str], T] | None = None) -> T:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return cast(raw) if cast else raw  # type: ignore[return-value]


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- secrets (optional at import time, required when used) ---
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    # --- models and endpoints ---
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 3072
    vision_model: str = "gemini-3.8-flash"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    chat_model: str = "openai/gpt-oss-120b"

    # --- vector database ---
    qdrant_url: str = "http://localhost:6333"
    text_collection: str = "docs"
    image_collection: str = "images"

    # --- text chunking and retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    image_top_k: int = 3
    min_image_score: float = 0.55

    # --- image extraction ---
    min_image_side_px: int = 120
    max_image_repeat_pages: int = 2
    render_drawing_pages: bool = False
    drawing_threshold: int = 40
    render_dpi: int = 110
    image_pause_seconds: int = 4

    # --- files and logging ---
    images_dir: Path = Path("data/images")
    uploads_dir: Path = Path("data/uploads")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        d = cls()
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY", None),
            groq_api_key=_env("GROQ_API_KEY", None),
            gemini_base_url=_env("GEMINI_BASE_URL", d.gemini_base_url),
            embed_model=_env("EMBED_MODEL", d.embed_model),
            embed_dim=_env("EMBED_DIM", d.embed_dim, int),
            vision_model=_env("VISION_MODEL", d.vision_model),
            groq_base_url=_env("GROQ_BASE_URL", d.groq_base_url),
            chat_model=_env("CHAT_MODEL", d.chat_model),
            qdrant_url=_env("QDRANT_URL", d.qdrant_url),
            text_collection=_env("TEXT_COLLECTION", d.text_collection),
            image_collection=_env("IMAGE_COLLECTION", d.image_collection),
            chunk_size=_env("CHUNK_SIZE", d.chunk_size, int),
            chunk_overlap=_env("CHUNK_OVERLAP", d.chunk_overlap, int),
            top_k=_env("TOP_K", d.top_k, int),
            image_top_k=_env("IMAGE_TOP_K", d.image_top_k, int),
            min_image_score=_env("MIN_IMAGE_SCORE", d.min_image_score, float),
            min_image_side_px=_env("MIN_IMAGE_SIDE_PX", d.min_image_side_px, int),
            max_image_repeat_pages=_env("MAX_IMAGE_REPEAT_PAGES", d.max_image_repeat_pages, int),
            render_drawing_pages=_env("RENDER_DRAWING_PAGES", d.render_drawing_pages, _as_bool),
            drawing_threshold=_env("DRAWING_THRESHOLD", d.drawing_threshold, int),
            render_dpi=_env("RENDER_DPI", d.render_dpi, int),
            image_pause_seconds=_env("IMAGE_PAUSE_SECONDS", d.image_pause_seconds, int),
            images_dir=_env("IMAGES_DIR", d.images_dir, Path),
            uploads_dir=_env("UPLOADS_DIR", d.uploads_dir, Path),
            log_level=_env("LOG_LEVEL", d.log_level),
        )

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (see .env.example)")
        return self.gemini_api_key

    def require_groq_key(self) -> str:
        if not self.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set (see .env.example)")
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
