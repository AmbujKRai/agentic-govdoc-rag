"""
The single retrieval entry point everything downstream should call:
hybrid search (dense + BM25 + RRF) to get a broad candidate set, then
cross-encoder reranking to pick the final top-k. This is what the agent
graph (Day 6-7) uses as its retrieval tool, and what the eval harness
(Day 8-9) scores as "the agentic system"'s retrieval step.

RERANK_ENABLED (env var, default true) can disable the cross-encoder step -
loading it alongside the embedding model pushed the deployed Render free
instance (512MB RAM) over the edge and got the process OOM-killed mid-
request. Disabled specifically on that deployment; still on by default for
local dev where RAM isn't the constraint.

Usage (standalone test):
    python retrieval/retriever.py "your question here"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() not in ("false", "0", "no")


def retrieve(
    query: str, hybrid_k: int = 20, final_k: int = 5, doc_type_filter: str | None = None
) -> list[dict]:
    candidates = hybrid_search(query, top_k=hybrid_k, doc_type_filter=doc_type_filter)
    if not RERANK_ENABLED:
        return candidates[:final_k]
    return rerank(query, candidates, top_k=final_k)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python retrieval/retriever.py "your question here"')
        sys.exit(1)
    query = sys.argv[1]

    results = retrieve(query)
    print(f"Query: {query}\n")
    print(f"Top {len(results)} after hybrid search + rerank:\n")
    for i, c in enumerate(results, 1):
        page_info = f", page {c['page']}" if c.get("page") else ""
        print(f"{i}. [{c['doc_id']}{page_info}]")
        print(f"   {c['text'][:200].strip()!r}\n")
