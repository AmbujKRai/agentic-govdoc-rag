"""Shared constants for the retrieval layer, so embed.py and the query-time
retriever can never drift apart on model name, collection name, or paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small output dimension

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

# Local, file-backed Qdrant by default (no separate server needed for
# solo development) - but its single-process file lock has caused real
# pain (Windows PermissionError whenever two processes touch it at once,
# see eval/run_eval.py history). Setting QDRANT_URL switches to a real
# Qdrant server instead (what docker-compose.yml runs), which also
# sidesteps that lock entirely.
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PATH = ROOT / "data" / "qdrant_db"
QDRANT_COLLECTION = "govdoc_chunks"

BM25_INDEX_PATH = ROOT / "data" / "bm25_index.pkl"
