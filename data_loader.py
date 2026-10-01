import os
import time
from pathlib import Path

from fastembed import TextEmbedding
from openai import OpenAI, RateLimitError
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 768

#stored inside the project so the download isn't lost when /tmp is cleadned
_embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=".fastembed_cache")

splitter = SentenceSplitter(chunk_size=400, chunk_overlap=80)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url=GEMINI_BASE_URL,
    max_retries=0,
)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=Path(path))
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                response = client.embeddings.create(model=EMBED_MODEL, input=batch)
                break
            except RateLimitError:
                # Gemini free tier has tight per-minute limits; wait and retry
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError("Gemini embedding rate limit: gave up after 5 retries")
        vectors.extend(item.embedding for item in response.data)
    return vectors