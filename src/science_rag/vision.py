"""Image understanding: a vision model turns a picture into a searchable description."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Protocol

from openai import InternalServerError, OpenAI, RateLimitError
from PIL import Image

from science_rag.config import Settings
from science_rag.retry import with_backoff

DECORATIVE_MARKER = "DECORATIVE"

CAPTION_PROMPT = (
    "You are indexing an image from a school science textbook so that students can find "
    "it by searching. Describe it in 80 to 150 words of plain text. Start with its type "
    "(diagram, graph, chart, table, photograph, drawing or map). Then state the topic, "
    "transcribe every visible title, label and axis name with units, describe any trend "
    "or relationship shown, and say which concept it teaches. "
    "If the image is only a logo, decoration, border or blank, "
    f"reply with exactly: {DECORATIVE_MARKER}"
)


class Captioner(Protocol):
    def caption(self, image_path: str | Path) -> str: ...


def is_decorative(caption: str) -> bool:
    return caption.strip().upper().startswith(DECORATIVE_MARKER)


def image_to_data_url(path: str | Path, max_side: int = 1600) -> str:
    """Re-encode any image as a size-limited PNG data URL (keeps requests small)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class GeminiCaptioner:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.vision_model
        self._client = OpenAI(
            api_key=settings.require_gemini_key(),
            base_url=settings.gemini_base_url,
            max_retries=0,
        )

    def caption(self, image_path: str | Path) -> str:
        data_url = image_to_data_url(image_path)

        def call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,  # headroom in case the model "thinks" first
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("vision model returned an empty caption")
            return text

        return with_backoff(call, retry_on=(RateLimitError, InternalServerError), attempts=4)
