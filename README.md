# GovDoc Copilot

**[Live demo](https://agentic-govdoc-rag.onrender.com/docs)** · Agentic RAG over Indian government document processes and welfare schemes

Answers questions about passports, driving licences, PAN cards, and welfare schemes (scholarships, PM-KISAN) — grounded in official sources, with citations, and an honest "I don't have that, verify here" instead of a confident guess.

> **Note:** the live demo runs on a free instance that sleeps after 15 minutes of inactivity. The first request after a sleep takes ~60s (cold start + model load); subsequent requests take ~10s.

---

## Why this exists

Government document requirements are conditional, scattered across contradictory sources, and revised without notice. Three real consequences follow: people submit incomplete applications and get rejected, they miss renewal deadlines, and they never discover benefits they qualify for.

That problem is a poor fit for naive RAG, and this repo demonstrates why rather than asserting it. The passport instruction booklet doesn't list required documents by name — it points you to *Table 2*, which maps your applicant category to document *numbers*, which only mean something once you cross-reference *Table 3*. Answering "what do I need?" requires resolving both tables in sequence.

Single-shot retrieval cannot do this, and I measured it: for the benchmark query, the chunk containing the needed table content ranks **~11th of 50** candidates before reranking and **~23rd of 50** after. No amount of top-k tuning fixes that. What fixes it is a system that notices its own retrieved context references a table it doesn't contain, and goes and fetches that specific thing.

---

## Architecture

```
Offline (build the index once):
  raw sources → fetch (Playwright for JS-rendered pages) → parse → chunk → embed + BM25 index

Online (answer a query):
  query → contextualize (rewrite follow-ups into standalone questions)
        → route (classify doc type)
        → retrieve: hybrid search (dense + BM25 → RRF) → cross-encoder rerank
        → check sufficiency ──insufficient──→ targeted follow-up query (max 2 hops) ──┐
                │                                                                     │
                └──sufficient──→ generate cited answer                                │
                                          ▲──────────────────────────────────────────┘
```

**On multi-turn:** retrieval needs a *self-contained* query — embedding `"what about for a minor?"` retrieves noise, because that string means almost nothing alone. So conversation history isn't fed to the retriever; it's used to rewrite the turn into a standalone question first (history-aware retrieval):

```
history:  "What documents do I need for a passport?"
new turn: "What about for a minor?"
   → rewrite → "What documents does a minor need for a passport?" → route + retrieve
```

Pass a `session_id` to `/chat` to enable this; omit it for stateless one-off queries. A genuine topic change is deliberately left unrewritten rather than forced to relate to the previous turn.

| Component | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit state machine; the loop-back edge is the whole point |
| Embeddings | `bge-small-en-v1.5` (local) | Free, CPU-friendly, no API dependency |
| Vector store | Qdrant | Local file mode for dev, server mode via `QDRANT_URL` |
| Keyword search | BM25 (`rank_bm25`) | Catches exact codes ("Table 2") that dense retrieval under-weights |
| Fusion | Reciprocal Rank Fusion | Standard, no tuning weights to overfit |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | Cleans up fusion noise |
| LLM | Groq (`openai/gpt-oss-120b`) | Genuinely free tier, fast enough for a multi-call agent loop |

**The sufficiency check** is the core of the system. It's prompted to catch one specific pattern: retrieved excerpts that *name* a table, section, or annexure without containing it. When that happens, the required fix is a follow-up query for that exact named thing — not a rephrasing of the original question.

---

## Results

Evaluated with a hand-verified golden set (10 questions across all 5 document categories). Every `must_mention` fact was grepped against the source text before being added — never taken from model output, which would make the eval circular.

Two metrics, both reference-free:
- **Faithfulness** — an LLM judge decomposes each answer into atomic claims and checks each against the actually-retrieved context. Catches hallucination.
- **Coverage** — a plain substring check for the hand-verified facts. Deliberately not LLM-judged, so anyone can audit it by reading the code.

| Run | Scored | Faithfulness (naive → agentic) | Coverage (naive → agentic) |
|---|---|---|---|
| 2026-08-28 | 9/10 | 0.98 → **1.00** | 0.56 → 0.56 |
| 2026-08-29 | 5/10 | 0.83 → **1.00** | 0.60 → 0.60 |

**Faithfulness is a consistent, real win.** The agentic system scored a perfect 1.00 on all 14 scored questions across both runs. Naive dipped as low as 0.56 — meaning nearly half its claims on that question weren't supported by what it actually retrieved.

**Coverage tied, and that's an honest finding worth explaining.** The passport golden-set fact was originally the literal string `"Table 3"`. That was fragile — which table gets cited by name varies between runs — so I swapped it for `"Birth Certificate"`. That fixed the fragility but also made the check *easier*, since naive retrieval surfaces that fact through a shallower path too. The multi-hop advantage is real and reproducible (see the rank-probing numbers above), it just isn't what this particular metric ends up measuring. Good eval design trades robustness against discriminative power, and it's easy to accidentally trade away the thing you meant to measure.

**Model quality matters as much as architecture.** One run used a smaller fallback model after exhausting the primary model's daily quota. The agentic advantage visibly shrank — on one question it scored *worse* than naive. The loop only helps if the underlying model can write a precise follow-up query and synthesize across hops.

Raw results: [`eval/results/`](eval/results/).

---

## Postmortem: bugs worth documenting

Retrieval pipelines fail *silently*. Most of these raised no exception — the system kept running and produced plausible-looking output. Each was found by inspecting intermediate artifacts, not by a crash.

**1. Four sources returned empty page shells.** Plain HTTP GET on the Parivahan and Income Tax portals returned navigation chrome and no content — they're JS-rendered SPAs. A 200 response is not evidence of real data. *Fix: flag those sources `render_js: true` in the manifest and fetch them through a headless browser.*

**2. An entire government table became one 3,279-character chunk.** PDF table rows have no blank lines between them, and my chunker split on blank lines — so the Table 2 content (the exact thing the benchmark query needed) was one unsplittable blob that couldn't rank against anything. The same root cause had already bitten in a different disguise: extracted HTML has no paragraph breaks either. *Fix: a line-level packer for oversized paragraphs. Corpus went from 225 to 266 well-formed chunks.*

**3. An import statement consumed 85% of the memory budget.** The deployed service kept getting OOM-killed, then failed to bind a port at all for 15 minutes. `sentence_transformers` was imported at module level in two retrieval files, which `api/main.py` imports transitively — so merely *starting the web server* paid the full cost. Measured: **433MB RSS from the import alone**, before loading a single model weight, on a 512MB instance. *Fix: move the imports inside the functions that need them. `import api.main` dropped to ~135MB.*

**4. A crash erased every result before it.** The eval harness only wrote results at the end of the run, so a failure on question 10 discarded the nine that had already succeeded. *Fix: save after every question.*

**5. Local Qdrant allows one connection per process.** The eval harness's naive path opened its own client instead of reusing the shared one, deadlocking on Windows every time. *Fix: share the singleton.*

**6. `requirements.txt` had invalid syntax, untested for the entire project.** `torch --index-url ...` works as a CLI argument but not inside a requirements file. Every install until deployment day had been package-by-package, so it was never exercised. *Fix: `--extra-index-url` as a proper directive — caught by installing the file into a throwaway venv before trusting the Docker build.*

**7. Groq's daily token quota is a rolling 24h window, not a calendar-day reset.** "Wait until tomorrow" did not produce a clean refill. *Fix: parse the `Xm Ys` wait hint from the error, retry with backoff, and fail fast past 90s rather than silently blocking on a 20-minute wait.*

---

## Known limitations

- **The corpus is manually curated.** Staying current requires a human to notice a source changed. A scheduled content-hash diff against `manifest.json` would flag staleness without the risk of auto-updating the index unattended.
- **The golden set is small** (10 questions) and, as above, one fact ended up less discriminative than intended.
- **Reranking is disabled in the live deployment** (`RERANK_ENABLED=false`) to fit in 512MB. Retrieval quality on the demo is slightly below what the code does locally.
- **The tracker has no notification delivery.** It computes renewal status correctly; nothing pushes it to the user.
- **PDF tables are handled by patching around text extraction.** [ColPali](https://huggingface.co/blog/manu/colpali)-style multi-vector page-image retrieval would fix bug #2 structurally — embedding pages as images with per-patch vectors, so table layout never has to survive serialization to text. It's the right answer given more compute budget than a free CPU tier.

---

## Running it

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt   # bin/pip on macOS/Linux
cp .env.example .env    # add your Groq API key (console.groq.com, free)

python ingestion/fetch_corpus.py    # download sources listed in manifest.json
python ingestion/parse.py           # → clean text
python ingestion/chunk.py           # → chunks.jsonl
python ingestion/embed.py           # → Qdrant
python ingestion/bm25_index.py      # → BM25 index

uvicorn api.main:app --reload       # http://localhost:8000/docs
```

Ask a question directly:

```bash
python agent/graph.py "I am 17 years old applying for my first passport, what documents do I need?"
```

Compare against the naive baseline, or run the eval:

```bash
python naive_rag.py "same question here"
python eval/run_eval.py
```

Optional local tracing (Phoenix UI at `localhost:6006`):

```bash
python observability/phoenix_setup.py
```

### Project layout

```
ingestion/      fetch → parse → chunk → embed + BM25 index
retrieval/      hybrid search (dense + BM25 + RRF), cross-encoder rerank
agent/          LangGraph state machine, query router, conversation memory, Groq retry wrapper
eval/           golden set, coverage + faithfulness scoring, naive-vs-agentic harness
tracker/        SQLite document validity tracking, per-doc-type renewal windows
observability/  Phoenix tracing, JSONL token/latency log
api/            FastAPI: /chat, /documents, /documents/alerts, /health
```

---

## Disclaimer

Not official or legal advice. Always verify requirements at the official source before acting on an answer from this system. The corpus reflects what was published when it was last fetched (see `retrieved_date` in `data/raw/manifest.json`) and government rules change without notice.
