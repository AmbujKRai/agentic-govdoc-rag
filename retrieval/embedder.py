"""Shared query-embedding helper - used by hybrid_search.py now, and by
the agent graph and eval harness later, so the model is loaded once and
the bge query-instruction prefix lives in exactly one place.

`sentence_transformers` is imported lazily, inside get_model(), rather than
at module level - importing it (which pulls in torch/transformers) has a
real memory cost on its own, before any model weights ever load, and this
module gets imported transitively just by importing api/main.py. On a
memory-constrained deployment (512MB), paying that cost merely to start the
web server and bind a port - before it's ever actually needed - starved
the process before Render's port scan even found it listening."""

from retrieval.config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(text: str) -> list[float]:
    # bge models are trained to expect this instruction prefix on the query
    # side only - passages are embedded plainly (see ingestion/embed.py).
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return get_model().encode(prefixed).tolist()
