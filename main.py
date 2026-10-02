import logging
import uuid
import os
import traceback
from typing import Any, cast

from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGQueryResult, RAGSearchResult

load_dotenv()

logger = logging.getLogger("uvicorn")

##GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
##GEMINI_CHAT_MODEL = "gemini-3.8-flash"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

BATCH_SIZE = 32
PAUSE_SECONDS = 0


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
            logger.info(f"[LOAD] reading {pdf_path}")
            chunks = load_and_chunk_pdf(pdf_path)
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
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Connect to Qdrant first so a missing database fails before any embedding calls
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
                {"source": source_id, "text": chunks[i]}
                for i in range(len(chunks))
            ]
            logger.info("[UPSERT] writing to Qdrant")
            store.upsert(ids, vecs, payloads)
            logger.info("[UPSERT] done")
        except Exception:
            logger.error("[UPSERT] FAILED\n" + traceback.format_exc())
            raise
        return RAGUpsertResult(ingested=len(chunks))

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


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    retries=4,
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = str(ctx.event.data["question"])

    raw_top_k = ctx.event.data.get("top_k", 5)
    top_k = int(raw_top_k) if isinstance(raw_top_k, (int, float, str)) else 5

    async def _search() -> RAGSearchResult:
        try:
            query_vec = embed_texts([question])[0]
            store = QdrantStorage()
            found = store.search(query_vec, top_k)
            return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
        except Exception:
            logger.error("[SEARCH] FAILED\n" + traceback.format_exc())
            raise

    found = await ctx.step.run(
        "embed-and-search",
        _search,
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        ##auth_key=os.environ["GEMINI_API_KEY"],
        ##base_url=GEMINI_BASE_URL,
        ##model=GEMINI_CHAT_MODEL,

        auth_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
        model=GROQ_CHAT_MODEL

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
                    {
                        "role": "system",
                        "content": "You answer questions using only the provided context.",
                    },
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
    }


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])