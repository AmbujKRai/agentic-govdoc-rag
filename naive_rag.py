"""
Day 3 baseline: the simplest possible RAG - single dense retrieval, no
reranking, no query routing, no multi-hop. This exists specifically to be
BEATEN by the agentic system later; eval/run_eval.py scores both and
reports the before/after delta.

Usage:
    python naive_rag.py "I'm a 17 year old applying for my first passport, what do I need?"
"""

import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters LLM
# output commonly contains (curly quotes, non-breaking hyphens, etc).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval.config import EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_PATH

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

SYSTEM_PROMPT = """You are an assistant that answers questions about Indian \
government document processes (passport, driving license, PAN) and welfare \
schemes (scholarships, PM-KISAN) using ONLY the provided source excerpts.

Rules:
- Answer ONLY using the excerpts below. Do not use outside knowledge.
- Every claim must cite its source using the [doc_id] tag shown with the excerpt.
- If the excerpts do not fully answer the question, say exactly what is \
missing and tell the user to verify at the relevant official source - do \
NOT guess or fill gaps with assumptions.
"""


def embed_query(model: SentenceTransformer, query: str):
    # bge models expect this instruction prefix on the query side only.
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode(prefixed).tolist()


def retrieve(client: QdrantClient, model: SentenceTransformer, query: str, top_k: int = TOP_K):
    vector = embed_query(model, query)
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
    ).points
    return [hit.payload for hit in hits]


def build_prompt(query: str, chunks: list[dict]) -> str:
    excerpts = []
    for c in chunks:
        page_info = f", page {c['page']}" if c.get("page") else ""
        excerpts.append(
            f"[{c['doc_id']}] (source: {c['title']}{page_info}, {c['source_url']})\n{c['text']}"
        )
    excerpt_block = "\n\n---\n\n".join(excerpts)
    return f"Source excerpts:\n\n{excerpt_block}\n\n---\n\nQuestion: {query}"


def generate_answer(groq_client: Groq, query: str, chunks: list[dict]) -> str:
    user_prompt = build_prompt(query, chunks)
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage: python naive_rag.py "your question here"')
        sys.exit(1)
    query = sys.argv[1]

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Copy .env.example to .env and add your free Groq API key.")
        sys.exit(1)

    print("Loading embedding model and Qdrant collection...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=str(QDRANT_PATH))
    groq_client = Groq()

    print(f"\nQuery: {query}\n")
    chunks = retrieve(client, model, query)

    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - [{c['doc_id']}] {c['title'][:70]}")

    print("\nGenerating answer...\n")
    answer = generate_answer(groq_client, query, chunks)
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
