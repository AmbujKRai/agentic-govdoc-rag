"""
Cross-encoder reranking: hybrid_search.py's RRF fusion is a cheap, coarse
combination of two rankers, good for narrowing hundreds of chunks down to
~20 candidates. A cross-encoder scores each (query, chunk) pair jointly
(more expensive, more accurate) to pick the final top-k that actually goes
to the LLM.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    scores = get_reranker().predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]
