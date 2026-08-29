"""
Reference-free faithfulness scoring: does the generated answer's claims
actually follow from the retrieved context, or is it hallucinating?

This is the same decompose-then-verify methodology Ragas' Faithfulness
metric uses internally (break the answer into atomic factual claims, check
each against the retrieved context). Implemented directly against Groq
rather than through the `ragas` package - that package's LLM factory does
an unconditional `from langchain_community.chat_models.vertexai import
ChatVertexAI` at import time, which fails without the (heavy, unrelated)
Google Cloud Vertex AI SDK installed. Pulling that in just to satisfy an
import path for a provider we never use wasn't worth it - this gives the
same signal with a dependency chain we actually control.
"""

import json
import os

from groq import Groq

from agent.groq_utils import chat_completion_with_retry

FAITHFULNESS_MODEL = os.environ.get("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You will be given a QUESTION, an ANSWER, and the CONTEXT \
excerpts the answer was supposed to be grounded in.

Break the ANSWER down into its individual factual claims, then judge each \
claim: is it actually supported by the CONTEXT, or does it go beyond what \
the CONTEXT states (unsupported / hallucinated)?

Notes:
- A claim that says information is missing, uncertain, or recommends the \
user verify elsewhere is itself a true, supportable statement if the \
CONTEXT genuinely doesn't cover that point - mark it supported.
- Purely conversational/formatting text (headers, "here is a summary", \
table labels with no factual content) does not need to be listed as a claim.

Respond with strict JSON only:
{"claims": [{"claim": "<short paraphrase of the claim>", "supported": true or false}, ...]}"""


def score_faithfulness(groq_client: Groq, question: str, answer: str, contexts: list[str]) -> dict:
    context_block = "\n\n---\n\n".join(contexts)
    user_prompt = f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context_block}"

    resp = chat_completion_with_retry(
        groq_client,
        purpose="faithfulness_score",
        model=FAITHFULNESS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(resp.choices[0].message.content)
        claims = result.get("claims", [])
    except (json.JSONDecodeError, TypeError):
        return {"score": None, "claims": [], "error": "failed to parse judge response"}

    if not claims:
        return {"score": None, "claims": [], "error": "no claims extracted"}

    n_supported = sum(1 for c in claims if c.get("supported"))
    return {"score": n_supported / len(claims), "claims": claims, "error": None}
