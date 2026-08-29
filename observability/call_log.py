"""
A per-call token/latency log, independent of whether Phoenix is running.
On the free tier there's no real dollar cost to track, so token count and
latency are the closest thing to a "cost dashboard" - and unlike Phoenix's
UI, this is a plain JSONL file anyone can open and read without spinning up
a server, which matters for a portfolio repo someone is skimming.

Usage: called from agent/groq_utils.py around every Groq call.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "call_log.jsonl"


@contextmanager
def log_call(purpose: str, model: str):
    """Wraps a Groq call; on exit, records latency + token usage (if the
    caller attaches a `.response` attribute to the yielded holder)."""
    start = time.perf_counter()
    holder = {"response": None, "error": None}
    try:
        yield holder
    except Exception as e:
        holder["error"] = str(e)
        raise
    finally:
        latency_s = time.perf_counter() - start
        usage = getattr(holder["response"], "usage", None)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": model,
            "latency_s": round(latency_s, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "error": holder["error"],
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarize(log_path: Path = LOG_PATH) -> dict:
    """Quick aggregate stats over the call log - total calls, total tokens,
    avg latency, breakdown by purpose. Used by observability/dashboard.py."""
    if not log_path.exists():
        return {"total_calls": 0}

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    total_tokens = sum(e["total_tokens"] or 0 for e in entries)
    total_latency = sum(e["latency_s"] for e in entries)
    errors = sum(1 for e in entries if e["error"])

    by_purpose = {}
    for e in entries:
        p = e["purpose"]
        by_purpose.setdefault(p, {"calls": 0, "tokens": 0, "latency_s": 0.0})
        by_purpose[p]["calls"] += 1
        by_purpose[p]["tokens"] += e["total_tokens"] or 0
        by_purpose[p]["latency_s"] += e["latency_s"]

    return {
        "total_calls": len(entries),
        "total_tokens": total_tokens,
        "avg_latency_s": round(total_latency / len(entries), 3) if entries else 0,
        "errors": errors,
        "by_purpose": by_purpose,
    }


if __name__ == "__main__":
    stats = summarize()
    print(json.dumps(stats, indent=2))
