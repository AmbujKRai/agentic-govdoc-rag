---
title: GovDoc Copilot
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# GovDoc Copilot

Agentic RAG over Indian government document processes (passport, driving licence, PAN) and welfare schemes (scholarships, PM-KISAN) — grounded in official sources, cited, and honest about what it doesn't know instead of guessing.

## Why this exists

People routinely submit incomplete applications because eligibility rules are conditional and scattered across contradictory sources, miss renewal deadlines with real consequences, and miss out on benefits they never discover. That problem needs a system that can chase down a multi-hop answer across linked eligibility clauses and say "verify at the source" when it genuinely doesn't know — not a chatbot that confidently guesses.

Naive single-shot retrieval provably cannot answer the hardest questions in this corpus. See the [full engineering field guide](https://claude.ai/code/artifact/4ca25ae2-536c-4390-9f4c-11fe9cc76563) for the architecture, every module, the day-by-day build journey (including the real bugs hit and fixed), and evaluation results.

## Architecture

```
Offline:  raw sources -> fetch (+ Playwright for JS-rendered pages) -> parse -> chunk -> embed + BM25 index
Online:   query -> route (classify doc type) -> hybrid search (dense + BM25 + RRF) + rerank
                -> sufficiency check -> (loop, up to 2 hops) -> generate (cited answer)
```

- **Retrieval**: dense embeddings (`sentence-transformers`) + BM25 keyword search, fused with Reciprocal Rank Fusion, then cross-encoder reranking.
- **Agentic loop**: [LangGraph](https://github.com/langchain-ai/langgraph) state machine that detects when retrieved context *references* something (e.g. "see Table 2") without containing it, and issues a targeted follow-up search instead of guessing.
- **Eval harness**: hand-verified golden set, reference-free faithfulness scoring, naive-vs-agentic comparison.
- **Tracker**: document validity/renewal tracking with per-document-type lead times.
- **LLM**: [Groq](https://groq.com) free tier.

## Run locally

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # Windows; use venv/bin/pip on macOS/Linux
cp .env.example .env   # add your Groq API key

python ingestion/fetch_corpus.py
python ingestion/parse.py
python ingestion/chunk.py
python ingestion/embed.py
python ingestion/bm25_index.py

uvicorn api.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API, or run a query directly:

```bash
python agent/graph.py "I am 17 years old applying for my first passport, what documents do I need?"
```

## Disclaimer

Not official or legal advice. Always verify requirements at the official source before relying on an answer from this system.
