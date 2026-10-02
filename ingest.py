import sys
import uuid
from pathlib import Path

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python ingest.py "/path/to/file.pdf"')
        sys.exit(1)

    pdf = Path(sys.argv[1]).expanduser().resolve()
    if not pdf.is_file():
        print(f"ERROR: file not found: {pdf}")
        print("Check the exact filename with: ls", pdf.parent)
        sys.exit(1)

    source_id = pdf.name

    print(f"[1/4] Reading and chunking {pdf.name}")
    chunks = load_and_chunk_pdf(str(pdf))
    print(f"      {len(chunks)} chunks")
    if not chunks:
        print("ERROR: no text extracted. This PDF is probably a scan and needs OCR.")
        sys.exit(1)

    print("[2/4] Embedding locally")
    vecs = embed_texts(chunks)
    print(f"      {len(vecs)} vectors of size {len(vecs[0])}")

    print("[3/4] Connecting to Qdrant")
    store = QdrantStorage()

    print("[4/4] Upserting")
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
        for i in range(len(chunks))
    ]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    store.upsert(ids, vecs, payloads)

    total = store.client.count(collection_name=store.collection, exact=True).count
    print(f"Done. Stored {len(chunks)} chunks from {source_id}. Collection now has {total} points.")


if __name__ == "__main__":
    main()