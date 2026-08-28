"""Shared query-embedding helper - used by hybrid_search.py now, and by
the agent graph and eval harness later, so the model is loaded once and
the bge query-instruction prefix lives in exactly one place."""

from sentence_transformers import SentenceTransformer

from retrieval.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(text: str) -> list[float]:
    # bge models are trained to expect this instruction prefix on the query
    # side only - passages are embedded plainly (see ingestion/embed.py).
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return get_model().encode(prefixed).tolist()
