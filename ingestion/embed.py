"""
Embeds every chunk in data/processed/chunks.jsonl with a local sentence-
transformers model and upserts them into Qdrant - a local file-backed store
by default, or a real Qdrant server if QDRANT_URL is set (see
retrieval/config.py; docker-compose.yml sets this to point at its qdrant
service, since the file-backed store only allows one process to open it at
a time).

Usage:
    python ingestion/embed.py
    QDRANT_URL=http://localhost:6333 python ingestion/embed.py   # index into a running server instead
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.config import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
)


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found - run ingestion/chunk.py first")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    # bge models are trained to embed passages plainly and queries with an
    # instruction prefix - we apply the query-side prefix at query time only.
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

    if QDRANT_URL:
        print(f"Writing to Qdrant server at {QDRANT_URL} ...")
        client = QdrantClient(url=QDRANT_URL)
    else:
        print(f"Writing to local Qdrant store at {QDRANT_PATH} ...")
        client = QdrantClient(path=str(QDRANT_PATH))

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )

    points = [
        qmodels.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload=chunks[i],  # full chunk incl. text + provenance for retrieval-time citation
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print(f"Indexed {len(points)} chunks into Qdrant collection '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
