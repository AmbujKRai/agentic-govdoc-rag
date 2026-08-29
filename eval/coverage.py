"""
A deliberately simple, fully transparent metric: for each golden question,
did the generated answer actually mention the key fact(s) it needs to?
Each golden-set entry's "must_mention" is a list of facts, each fact itself
a list of acceptable phrasings (any one matching counts as present) - e.g.
["01.02.2019", "1 February 2019"] for a date that might be phrased either
way.

This is intentionally not LLM-judged: every fact was hand-verified against
the source text before being added to golden_set.jsonl (see project notes),
so a plain case-insensitive substring check is honest, cheap, and lets
anyone reading the eval results verify exactly what was checked and why -
unlike an LLM-judged score, which is opaque without re-running it yourself.
"""


def check_coverage(answer: str, must_mention: list[list[str]]) -> dict:
    answer_lower = answer.lower()
    hits = []
    for fact_variants in must_mention:
        found = any(variant.lower() in answer_lower for variant in fact_variants)
        hits.append({"fact": fact_variants[0], "found": found})

    n_found = sum(h["found"] for h in hits)
    return {
        "coverage": n_found / len(hits) if hits else 1.0,
        "n_found": n_found,
        "n_total": len(hits),
        "details": hits,
    }
