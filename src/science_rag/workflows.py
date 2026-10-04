"""Inngest durable workflows: the thin orchestration layer.

Each ctx.step.run is checkpointed, so a failure in a late step never repeats earlier
work, and every run is inspectable in the Inngest dashboard. Blocking work (PDF parsing,
network calls) runs in worker threads so the event loop stays responsive."""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, cast

import inngest
from inngest.experimental import ai

from science_rag import events
from science_rag.config import get_settings
from science_rag.models import ChunkBatch, CountResult, ImageList, Retrieval
from science_rag.prompts import build_messages
from science_rag.service import NoTextExtractedError, get_service

logger = logging.getLogger("science_rag.workflows")

inngest_client = inngest.Inngest(
    app_id=events.APP_ID,
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


def _pdf_event(ctx: inngest.Context) -> tuple[str, str]:
    pdf_path = str(ctx.event.data["pdf_path"])
    source_id = str(ctx.event.data.get("source_id", pdf_path))
    return pdf_path, source_id


def _permanent(exc: Exception) -> inngest.NonRetriableError:
    """Errors that retrying cannot fix (bad path, scanned PDF) must not be retried."""
    return inngest.NonRetriableError(str(exc))


# ============================================================ text ingestion
@inngest_client.create_function(
    fn_id="ingest-pdf-text",
    retries=1,
    trigger=inngest.TriggerEvent(event=events.INGEST_PDF),
)
async def ingest_pdf_text(ctx: inngest.Context) -> dict[str, int]:
    pdf_path, source_id = _pdf_event(ctx)

    async def _chunk() -> ChunkBatch:
        try:
            chunks = await asyncio.to_thread(get_service().chunk_pdf, pdf_path)
        except (FileNotFoundError, NoTextExtractedError) as exc:
            raise _permanent(exc) from exc
        return ChunkBatch(source_id=source_id, chunks=chunks)

    batch = await ctx.step.run("chunk-pdf", _chunk, output_type=ChunkBatch)

    async def _store() -> CountResult:
        count = await asyncio.to_thread(get_service().store_chunks, batch.chunks, batch.source_id)
        return CountResult(count=count)

    stored = await ctx.step.run("embed-and-store", _store, output_type=CountResult)
    return {"chunks_stored": stored.count}


# ============================================================ image ingestion
# Fan-out: the same "rag/ingest_pdf" event also triggers this function, so one upload
# ingests both the text and the pictures of a PDF.
@inngest_client.create_function(
    fn_id="ingest-pdf-images",
    retries=1,
    trigger=inngest.TriggerEvent(event=events.INGEST_PDF),
)
async def ingest_pdf_images(ctx: inngest.Context) -> dict[str, int]:
    pdf_path, source_id = _pdf_event(ctx)

    async def _extract() -> ImageList:
        try:
            images = await asyncio.to_thread(get_service().extract_images, pdf_path, source_id)
        except FileNotFoundError as exc:
            raise _permanent(exc) from exc
        return ImageList(source_id=source_id, images=images)

    listing = await ctx.step.run("extract-images", _extract, output_type=ImageList)

    pause = datetime.timedelta(seconds=get_settings().image_pause_seconds)
    stored = 0
    total = len(listing.images)
    for n, image in enumerate(listing.images):

        async def _describe(image=image) -> CountResult:
            ok = await asyncio.to_thread(
                get_service().store_image, image.path, listing.source_id, image.page, image.index
            )
            return CountResult(count=int(ok))

        result = await ctx.step.run(f"describe-image-{n}", _describe, output_type=CountResult)
        stored += result.count
        if n < total - 1:
            await ctx.step.sleep(f"pause-{n}", pause)  # stay under free-tier rate limits

    return {"images_found": total, "images_stored": stored}


@inngest_client.create_function(
    fn_id="ingest-image",
    retries=2,
    trigger=inngest.TriggerEvent(event=events.INGEST_IMAGE),
)
async def ingest_image(ctx: inngest.Context) -> dict[str, bool]:
    image_path = str(ctx.event.data["image_path"])
    source_id = str(ctx.event.data.get("source_id", image_path.rsplit("/", 1)[-1]))

    async def _run() -> CountResult:
        try:
            ok = await asyncio.to_thread(get_service().store_image, image_path, source_id)
        except FileNotFoundError as exc:
            raise _permanent(exc) from exc
        return CountResult(count=int(ok))

    result = await ctx.step.run("describe-and-store", _run, output_type=CountResult)
    return {"stored": bool(result.count)}


# ============================================================ question answering
@inngest_client.create_function(
    fn_id="answer-question",
    retries=4,
    trigger=inngest.TriggerEvent(event=events.ASK_QUESTION),
)
async def answer_question(ctx: inngest.Context) -> dict[str, Any]:
    settings = get_settings()
    question = str(ctx.event.data["question"])
    raw_top_k = ctx.event.data.get("top_k", settings.top_k)
    top_k = int(raw_top_k) if isinstance(raw_top_k, (int, float, str)) else settings.top_k

    async def _retrieve() -> Retrieval:
        return await asyncio.to_thread(get_service().retrieve, question, top_k)

    retrieval = await ctx.step.run("retrieve", _retrieve, output_type=Retrieval)
    figures = [h.model_dump() for h in retrieval.images]

    # Nothing retrieved from the notes: skip the LLM call (saves tokens, avoids invention)
    if not retrieval.contexts:
        return {
            "answer": "I couldn't find anything relevant in the ingested notes.",
            "sources": [],
            "num_contexts": 0,
            "images": figures,
        }

    # step.ai.infer is executed by the Inngest server, not by this process
    adapter = ai.openai.Adapter(
        auth_key=settings.require_groq_key(),
        base_url=settings.groq_base_url,
        model=settings.chat_model,
    )
    response = cast(
        dict[str, Any],
        await ctx.step.ai.infer(
            "generate-answer",
            adapter=adapter,
            body={
                "max_tokens": 2048,
                "temperature": 0.2,
                "messages": build_messages(question, retrieval),
            },
        ),
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": retrieval.sources,
        "num_contexts": len(retrieval.contexts),
        "images": figures,
    }


FUNCTIONS = [ingest_pdf_text, ingest_pdf_images, ingest_image, answer_question]
