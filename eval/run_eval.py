"""
Runs the golden set through naive_rag.py and agent/graph.py, scores both
with a coverage check (did the answer actually contain the key facts?) and
a reference-free faithfulness check (does the answer's claims actually
follow from the retrieved context, i.e. is it hallucinating?), and
prints/saves a before/after comparison table.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --limit 3     # quick smoke test on first 3 questions
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import run as run_agentic
from eval.coverage import check_coverage
from eval.faithfulness import score_faithfulness
from naive_rag import generate_answer as naive_generate_answer
from naive_rag import retrieve as naive_retrieve
from retrieval.embedder import get_model as get_embedding_model
from retrieval.hybrid_search import _get_qdrant_client

load_dotenv()

GOLDEN_SET_PATH = ROOT / "eval" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_naive(question: str) -> tuple[str, list[dict]]:
    # Reuse the same cached Qdrant client the agentic path uses (via
    # retrieval/hybrid_search.py) - the local file-backed store only allows
    # one open handle per process, so a second independent QdrantClient
    # instance deadlocks on Windows (portalocker AlreadyLocked).
    embed_model = get_embedding_model()
    client = _get_qdrant_client()
    groq_client = Groq()
    chunks = naive_retrieve(client, embed_model, question)
    answer = naive_generate_answer(groq_client, question, chunks)
    return answer, chunks


def save_results(results: list[dict], out_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    golden_set = load_golden_set()
    if args.limit:
        golden_set = golden_set[: args.limit]

    groq_client = Groq()
    out_path = RESULTS_DIR / f"eval_{date.today().isoformat()}.json"

    results = []
    for i, item in enumerate(golden_set):
        print(f"\n[{i+1}/{len(golden_set)}] {item['id']}: {item['question']}")

        try:
            print("  running naive...")
            naive_answer, naive_chunks = run_naive(item["question"])
            naive_coverage = check_coverage(naive_answer, item["must_mention"])
            naive_faith = score_faithfulness(
                groq_client, item["question"], naive_answer, [c["text"] for c in naive_chunks]
            )
            print(f"    naive: coverage={naive_coverage['coverage']:.2f} faithfulness={naive_faith['score']}")

            print("  running agentic...")
            agentic_state = run_agentic(item["question"])
            agentic_answer = agentic_state["final_answer"] or ""
            agentic_chunks = agentic_state["retrieved_chunks"]
            agentic_coverage = check_coverage(agentic_answer, item["must_mention"])
            agentic_faith = score_faithfulness(
                groq_client, item["question"], agentic_answer, [c["text"] for c in agentic_chunks]
            )
            print(f"    agentic: coverage={agentic_coverage['coverage']:.2f} faithfulness={agentic_faith['score']} hops={agentic_state['hop_count']}")

            results.append({
                "id": item["id"],
                "question": item["question"],
                "naive": {
                    "answer": naive_answer,
                    "coverage": naive_coverage,
                    "faithfulness": naive_faith,
                    "n_chunks": len(naive_chunks),
                },
                "agentic": {
                    "answer": agentic_answer,
                    "coverage": agentic_coverage,
                    "faithfulness": agentic_faith,
                    "n_chunks": len(agentic_chunks),
                    "hops": agentic_state["hop_count"],
                },
            })
        except Exception as e:
            # A single stubborn failure (e.g. a daily token quota that won't
            # clear for the retry wrapper's max wait) shouldn't lose every
            # result already collected - record it and move on.
            print(f"    [FAILED: {e}]")
            results.append({"id": item["id"], "question": item["question"], "error": str(e)})

        # Save after every question, not just at the end - a batch this
        # sensitive to external rate limits will eventually get interrupted
        # mid-run, and losing 9 completed results because the 10th failed
        # is exactly the kind of thing a real eval harness shouldn't do.
        save_results(results, out_path)
        time.sleep(2)  # be gentle on the free-tier rate limit across questions

    print("\n" + "=" * 78)
    print(f"{'id':<30} {'naive cov':>10} {'naive faith':>12} {'agentic cov':>12} {'agentic faith':>14}")
    print("-" * 78)
    n_cov_sum, a_cov_sum, n_scored = 0, 0, 0
    n_faith_vals, a_faith_vals = [], []
    failed_ids = []
    for r in results:
        if "error" in r:
            failed_ids.append(r["id"])
            print(f"{r['id']:<30} {'FAILED: ' + r['error'][:50]:>60}")
            continue
        n_scored += 1
        n_cov, a_cov = r["naive"]["coverage"]["coverage"], r["agentic"]["coverage"]["coverage"]
        n_faith, a_faith = r["naive"]["faithfulness"]["score"], r["agentic"]["faithfulness"]["score"]
        n_cov_sum += n_cov
        a_cov_sum += a_cov
        if n_faith is not None:
            n_faith_vals.append(n_faith)
        if a_faith is not None:
            a_faith_vals.append(a_faith)
        print(f"{r['id']:<30} {n_cov:>10.2f} {str(round(n_faith,2)) if n_faith is not None else 'n/a':>12} "
              f"{a_cov:>12.2f} {str(round(a_faith,2)) if a_faith is not None else 'n/a':>14}")

    print("-" * 78)
    if failed_ids:
        print(f"({len(failed_ids)} question(s) failed and are excluded from averages: {', '.join(failed_ids)})")
    avg_n_cov = n_cov_sum / n_scored if n_scored else 0
    avg_a_cov = a_cov_sum / n_scored if n_scored else 0
    avg_n_faith = sum(n_faith_vals) / len(n_faith_vals) if n_faith_vals else None
    avg_a_faith = sum(a_faith_vals) / len(a_faith_vals) if a_faith_vals else None
    print(f"{'AVERAGE':<30} {avg_n_cov:>10.2f} "
          f"{str(round(avg_n_faith,2)) if avg_n_faith is not None else 'n/a':>12} "
          f"{avg_a_cov:>12.2f} "
          f"{str(round(avg_a_faith,2)) if avg_a_faith is not None else 'n/a':>14}")
    print("=" * 78)
    print(f"\nFull results saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
