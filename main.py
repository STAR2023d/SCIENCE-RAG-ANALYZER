import asyncio
import datetime
import logging
import os
import traceback
import uuid
from typing import Any, cast

from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts, EMBED_DIM
from vector_db import QdrantStorage
from image_loader import (
    IMAGE_COLLECTION,
    extract_pdf_images,
    caption_embed_store,
)
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGImageHit,
    RAGImageList,
    ExtractedImage,
)

load_dotenv()

logger = logging.getLogger("uvicorn")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"

IMAGE_TOP_K = 3            # max figures returned with an answer
MIN_IMAGE_SCORE = 0.55     # cosine similarity cutoff. TUNE with check_images.py
IMAGE_PAUSE_SECONDS = 4    # pause between images to respect free-tier limits

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context does not contain the answer, say you could not find it "
    "in the provided notes. Do not use outside knowledge. "
    "Figures related to the question may be listed after the context. "
    "They are displayed to the user next to your answer, so you may refer to them "
    "(for example 'see the figure below'), but only describe what their listed "
    "descriptions say."
)

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# =====================================================================
# TEXT INGESTION
# =====================================================================
def _embed_and_store(chunks: list[str], source_id: str) -> int:
    """Blocking work: connect to Qdrant, embed, upsert. Run via asyncio.to_thread."""
    try:
        logger.info("[UPSERT] connecting to Qdrant")
        store = QdrantStorage()
    except Exception:
        logger.error("[UPSERT] Qdrant connection FAILED\n" + traceback.format_exc())
        raise

    try:
        logger.info(f"[EMBED] embedding {len(chunks)} chunks")
        vecs = embed_texts(chunks)
        logger.info(f"[EMBED] done, {len(vecs)} vectors")
    except Exception:
        logger.error("[EMBED] FAILED\n" + traceback.format_exc())
        raise

    try:
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]
        logger.info("[UPSERT] writing to Qdrant")
        store.upsert(ids, vecs, payloads)
        logger.info("[UPSERT] done")
    except Exception:
        logger.error("[UPSERT] FAILED\n" + traceback.format_exc())
        raise

    return len(chunks)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    retries=0,
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load() -> RAGChunkAndSrc:
        try:
            pdf_path = str(ctx.event.data["pdf_path"])
            source_id = str(ctx.event.data.get("source_id", pdf_path))

            if not os.path.isfile(pdf_path):
                raise inngest.NonRetriableError(f"PDF not found: {pdf_path}")

            logger.info(f"[LOAD] reading {pdf_path}")
            chunks = await asyncio.to_thread(load_and_chunk_pdf, pdf_path)
            logger.info(f"[LOAD] got {len(chunks)} chunks")
            if not chunks:
                raise inngest.NonRetriableError(
                    "No text extracted from PDF (scanned/image-only PDF?)"
                )
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
        except Exception:
            logger.error("[LOAD] FAILED\n" + traceback.format_exc())
            raise

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        count = await asyncio.to_thread(
            _embed_and_store, chunks_and_src.chunks, chunks_and_src.source_id
        )
        return RAGUpsertResult(ingested=count)

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        _load,
        output_type=RAGChunkAndSrc,
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )
    return ingested.model_dump()


# =====================================================================
# IMAGE INGESTION
# Fan-out: the SAME "rag/ingest_pdf" event also triggers this function, so
# uploading a PDF in Streamlit ingests its text AND its pictures, with no UI change.
# =====================================================================
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF Images",
    retries=1,
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf_images(ctx: inngest.Context):
    async def _extract() -> RAGImageList:
        try:
            pdf_path = str(ctx.event.data["pdf_path"])
            source_id = str(ctx.event.data.get("source_id", pdf_path))

            if not os.path.isfile(pdf_path):
                raise inngest.NonRetriableError(f"PDF not found: {pdf_path}")

            logger.info(f"[IMG-EXTRACT] scanning {pdf_path}")
            items = await asyncio.to_thread(extract_pdf_images, pdf_path, source_id)
            logger.info(f"[IMG-EXTRACT] found {len(items)} usable images")
            return RAGImageList(
                source_id=source_id,
                images=[ExtractedImage(**i) for i in items],
            )
        except Exception:
            logger.error("[IMG-EXTRACT] FAILED\n" + traceback.format_exc())
            raise

    listing = await ctx.step.run("extract-images", _extract, output_type=RAGImageList)

    stored = 0
    total = len(listing.images)
    for n, img in enumerate(listing.images):

        async def _caption_and_store(img=img) -> RAGUpsertResult:
            try:
                logger.info(f"[IMG] {n + 1}/{total} captioning page {img.page}")
                caption = await asyncio.to_thread(
                    caption_embed_store,
                    img.path,
                    listing.source_id,
                    img.page,
                    img.index,
                )
                return RAGUpsertResult(ingested=0 if caption is None else 1)
            except Exception:
                logger.error(f"[IMG {n}] FAILED\n" + traceback.format_exc())
                raise

        result = await ctx.step.run(
            f"caption-{n}", _caption_and_store, output_type=RAGUpsertResult
        )
        stored += result.ingested

        if n < total - 1:
            await ctx.step.sleep(
                f"pause-{n}", datetime.timedelta(seconds=IMAGE_PAUSE_SECONDS)
            )

    return {"images_found": total, "images_stored": stored}


@inngest_client.create_function(
    fn_id="RAG: Ingest Image",
    retries=2,
    trigger=inngest.TriggerEvent(event="rag/ingest_image"),
)
async def rag_ingest_image(ctx: inngest.Context):
    async def _run() -> RAGUpsertResult:
        try:
            image_path = str(ctx.event.data["image_path"])
            source_id = str(ctx.event.data.get("source_id", os.path.basename(image_path)))
            if not os.path.isfile(image_path):
                raise inngest.NonRetriableError(f"Image not found: {image_path}")
            logger.info(f"[IMG] captioning {image_path}")
            caption = await asyncio.to_thread(
                caption_embed_store, image_path, source_id, 0, 0
            )
            return RAGUpsertResult(ingested=0 if caption is None else 1)
        except Exception:
            logger.error("[IMG] FAILED\n" + traceback.format_exc())
            raise

    result = await ctx.step.run("caption-and-store", _run, output_type=RAGUpsertResult)
    return result.model_dump()


# =====================================================================
# QUERY
# =====================================================================
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    retries=4,
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = str(ctx.event.data["question"])

    raw_top_k = ctx.event.data.get("top_k", 5)
    top_k = int(raw_top_k) if isinstance(raw_top_k, (int, float, str)) else 5

    def _search_sync() -> RAGSearchResult:
        # ONE embedding call serves both searches
        query_vec = embed_texts([question])[0]
        found = QdrantStorage().search(query_vec, top_k)

        hits: list[RAGImageHit] = []
        image_store = QdrantStorage(collection=IMAGE_COLLECTION, dim=EMBED_DIM)
        for p in image_store.search_points(query_vec, IMAGE_TOP_K, MIN_IMAGE_SCORE):
            payload = p["payload"]
            path = payload.get("path", "")
            if not path or not os.path.isfile(path):
                continue                                   # file was deleted/moved
            hits.append(
                RAGImageHit(
                    path=path,
                    caption=payload.get("caption", ""),
                    source=payload.get("source", ""),
                    page=int(payload.get("page", 0)),
                    score=float(p["score"]),
                )
            )
        return RAGSearchResult(
            contexts=found["contexts"], sources=found["sources"], images=hits
        )

    async def _search() -> RAGSearchResult:
        try:
            return await asyncio.to_thread(_search_sync)
        except Exception:
            logger.error("[SEARCH] FAILED\n" + traceback.format_exc())
            raise

    found = await ctx.step.run(
        "embed-and-search",
        _search,
        output_type=RAGSearchResult,
    )
    images_out = [h.model_dump() for h in found.images]

    # Nothing retrieved from the notes: skip the LLM call (saves Groq tokens)
    if not found.contexts:
        return {
            "answer": "I couldn't find anything relevant in the ingested notes.",
            "sources": [],
            "num_contexts": 0,
            "images": images_out,
        }

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    figures_block = ""
    if found.images:
        lines = "\n".join(
            f"- [Figure from {h.source}" + (f", page {h.page}" if h.page else "") + f"] {h.caption}"
            for h in found.images
        )
        figures_block = f"\n\nRelated figures (shown to the user):\n{lines}"

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}{figures_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
        model=GROQ_CHAT_MODEL,
    )

    res = cast(
        dict[str, Any],
        await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 2048,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
        ),
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
        "images": images_out,
    }


app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_ingest_pdf_images, rag_ingest_image, rag_query_pdf_ai],
)