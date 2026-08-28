"""
Classifies which document category a query is most likely about, so
retrieval can optionally be scoped to that category for higher precision -
this is what fixed the "irrelevant PAN chunks leaking into a passport
query" noise we saw during Day 4-5 testing, without needing the reranker
to clean it up after the fact.

Only applied when the classifier is confident - "low" confidence falls
back to unfiltered search, since a wrong filter is worse than no filter.
"""

import json
import sys
from pathlib import Path

from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.groq_utils import chat_completion_with_retry

DOC_TYPES = ["passport", "driving_license", "pan", "scheme_scholarship", "scheme_subsidy"]

ROUTER_MODEL = "openai/gpt-oss-20b"  # small/fast is enough for classification

SYSTEM_PROMPT = f"""Classify which category of Indian government document or \
scheme a question is about.

Categories: {", ".join(DOC_TYPES)}

If the question could plausibly span multiple categories, or you are not \
reasonably sure, use "unknown" rather than guessing.

Respond with strict JSON only, no other text:
{{"doc_type": "<one of the categories above, or unknown>", "confidence": "high" or "low"}}"""


def classify_query(groq_client: Groq, query: str) -> dict:
    resp = chat_completion_with_retry(
        groq_client,
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"doc_type": "unknown", "confidence": "low"}

    if result.get("doc_type") not in DOC_TYPES:
        result["doc_type"] = "unknown"
    if result.get("confidence") not in ("high", "low"):
        result["confidence"] = "low"
    return result
