"""
Builds a BM25 keyword index over the same chunks used for dense retrieval,
and pickles it alongside the chunk list (so retrieval/hybrid_search.py can
map BM25 ranks back to chunk payloads without re-reading chunks.jsonl).

Usage:
    python ingestion/bm25_index.py
"""

import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.config import BM25_INDEX_PATH, CHUNKS_PATH

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found - run ingestion/chunk.py first")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"Tokenizing {len(chunks)} chunks ...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    print("Building BM25 index ...")
    bm25 = BM25Okapi(tokenized_corpus)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Saved BM25 index for {len(chunks)} chunks -> {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
