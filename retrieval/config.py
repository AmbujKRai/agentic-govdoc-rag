"""Shared constants for the retrieval layer, so embed.py and the query-time
retriever can never drift apart on model name, collection name, or paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small output dimension

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

# Local, file-backed Qdrant (no separate server needed for development).
# docker-compose swaps this for a real Qdrant service at deploy time.
QDRANT_PATH = ROOT / "data" / "qdrant_db"
QDRANT_COLLECTION = "govdoc_chunks"

BM25_INDEX_PATH = ROOT / "data" / "bm25_index.pkl"
