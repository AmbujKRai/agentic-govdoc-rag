"""
Hybrid retrieval: dense (Qdrant) + keyword (BM25), combined with Reciprocal
Rank Fusion (RRF). This is the fix for the exact gap the naive baseline
exposed - a query can match strongly on domain-specific terms (e.g. "Table
2", "Annexure D") that BM25 catches well but dense embeddings can under-
weight, or vice versa for paraphrased/semantic queries.

Usage (standalone test):
    python retrieval/hybrid_search.py "your question here"
"""

import pickle
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.config import BM25_INDEX_PATH, QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL
from retrieval.embedder import embed_query

TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60  # standard RRF damping constant

_bm25_data = None
_qdrant_client: QdrantClient | None = None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _load_bm25():
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
        else:
            _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    return _qdrant_client


def dense_search(query: str, top_k: int = 20, doc_type_filter: str | None = None) -> list[tuple[str, dict]]:
    vector = embed_query(query)
    query_filter = None
    if doc_type_filter:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type_filter))]
        )
    hits = _get_qdrant_client().query_points(
        collection_name=QDRANT_COLLECTION, query=vector, limit=top_k, query_filter=query_filter
    ).points
    return [(hit.payload["chunk_id"], hit.payload) for hit in hits]


def bm25_search(query: str, top_k: int = 20, doc_type_filter: str | None = None) -> list[tuple[str, dict]]:
    data = _load_bm25()
    bm25, chunks = data["bm25"], data["chunks"]
    scores = bm25.get_scores(tokenize(query))
    indices = range(len(scores))
    if doc_type_filter:
        indices = [i for i in indices if chunks[i]["doc_type"] == doc_type_filter]
    ranked_idx = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i]["chunk_id"], chunks[i]) for i in ranked_idx if scores[i] > 0]


def reciprocal_rank_fusion(*ranked_lists: list[tuple[str, dict]], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, payload) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            payloads[chunk_id] = payload
    fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [payloads[cid] for cid in fused_ids]


def hybrid_search(
    query: str, top_k: int = 10, dense_k: int = 20, bm25_k: int = 20, doc_type_filter: str | None = None
) -> list[dict]:
    dense = dense_search(query, top_k=dense_k, doc_type_filter=doc_type_filter)
    bm25 = bm25_search(query, top_k=bm25_k, doc_type_filter=doc_type_filter)
    fused = reciprocal_rank_fusion(dense, bm25)
    return fused[:top_k]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python retrieval/hybrid_search.py "your question here"')
        sys.exit(1)
    query = sys.argv[1]

    print(f"Query: {query}\n")

    print("--- dense only (top 5) ---")
    for chunk_id, payload in dense_search(query, top_k=5):
        print(f"  [{payload['doc_id']}] {payload['text'][:90]!r}")

    print("\n--- bm25 only (top 5) ---")
    for chunk_id, payload in bm25_search(query, top_k=5):
        print(f"  [{payload['doc_id']}] {payload['text'][:90]!r}")

    print("\n--- fused (RRF, top 5) ---")
    for payload in hybrid_search(query, top_k=5):
        print(f"  [{payload['doc_id']}] {payload['text'][:90]!r}")
