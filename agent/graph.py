"""
The agentic RAG loop: route -> retrieve -> check sufficiency -> (re-query up
to MAX_HOPS times) -> generate. This exists specifically to fix the gap the
naive baseline and hybrid+rerank both hit: a query like "I'm 17, applying for
my first passport, what documents do I need?" retrieves chunks that
EXPLICITLY REFERENCE "Table 2"/"Table 3" by name but don't contain them -
and single-shot retrieval (even hybrid+reranked, even at top-50) can't
surface those tables because the conversational query phrasing doesn't
lexically/semantically resemble their dense coded content.

The check_sufficiency step is built to specifically catch that pattern: a
reference to named content that isn't itself present counts as insufficient,
and the fix is a targeted follow-up query for that named content - not a
rephrasing of the original question.

Usage:
    python agent/graph.py "your question here"
"""

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.groq_utils import chat_completion_with_retry
from agent.router import classify_query
from retrieval.retriever import retrieve

load_dotenv()

GENERATION_MODEL = "openai/gpt-oss-120b"
MAX_HOPS = 2

_groq_client: Groq | None = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()
    return _groq_client


def merge_chunks(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {c["chunk_id"] for c in existing}
    merged = list(existing)
    for c in new:
        if c["chunk_id"] not in seen:
            merged.append(c)
            seen.add(c["chunk_id"])
    return merged


class AgentState(TypedDict):
    query: str
    doc_type_filter: str | None
    retrieved_chunks: list[dict]
    hop_count: int
    sufficient: bool
    missing_info: str | None
    follow_up_query: str | None
    final_answer: str | None


SUFFICIENCY_SYSTEM_PROMPT = """You are evaluating whether retrieved excerpts are \
enough to fully and specifically answer a user's question about Indian \
government documents or welfare schemes.

Judge strictly:
- If the excerpts only give partial or general instructions but explicitly \
reference another table, section, or annexure BY NAME (e.g. "refer Table 2", \
"see Annexure D", "Table 3: Overall List of Documents") that would contain \
the specific answer, and that referenced content is NOT itself present in the \
excerpts, this is INSUFFICIENT. A reference to information is not the same \
as the information itself.
- If nothing in the excerpts is relevant at all, this is also INSUFFICIENT.
- Only mark sufficient if the excerpts actually contain enough concrete, \
specific detail (not just a pointer to where it might be) to answer the question.

If insufficient, propose exactly ONE targeted follow-up search query that \
would find the missing content. If a table/section/annexure is referenced by \
name, search for that exact name - do not just rephrase the original question.

Respond with strict JSON only, no other text:
{"sufficient": true or false, "missing_info": "<what's missing, or null if sufficient>", "follow_up_query": "<specific search query, or null if sufficient>"}"""

GENERATE_SYSTEM_PROMPT = """You are an assistant that answers questions about Indian \
government document processes (passport, driving license, PAN) and welfare \
schemes (scholarships, PM-KISAN) using ONLY the provided source excerpts.

Rules:
- Answer ONLY using the excerpts below. Do not use outside knowledge.
- Every claim must cite its source using the [doc_id] tag shown with the excerpt.
- If the excerpts do not fully answer the question even after multiple search \
attempts, say exactly what is missing and tell the user to verify at the \
relevant official source - do NOT guess or fill gaps with assumptions."""


def format_chunks(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        page_info = f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[{c['doc_id']}] (source: {c['title']}{page_info}, {c['source_url']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def route_node(state: AgentState) -> dict:
    result = classify_query(get_groq(), state["query"])
    doc_type_filter = result["doc_type"] if result["confidence"] == "high" else None
    return {"doc_type_filter": doc_type_filter}


def retrieve_node(state: AgentState) -> dict:
    query = state.get("follow_up_query") or state["query"]
    results = retrieve(query, doc_type_filter=state.get("doc_type_filter"))
    return {
        "retrieved_chunks": merge_chunks(state.get("retrieved_chunks", []), results),
        "hop_count": state.get("hop_count", 0) + 1,
    }


def check_sufficiency_node(state: AgentState) -> dict:
    context = format_chunks(state["retrieved_chunks"])
    user_prompt = f"Question: {state['query']}\n\nRetrieved excerpts so far:\n\n{context}"
    resp = chat_completion_with_retry(
        get_groq(),
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SUFFICIENCY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        result = {"sufficient": True, "missing_info": None, "follow_up_query": None}

    return {
        "sufficient": bool(result.get("sufficient", True)),
        "missing_info": result.get("missing_info"),
        "follow_up_query": result.get("follow_up_query"),
    }


def should_continue(state: AgentState) -> str:
    if state["sufficient"]:
        return "generate"
    if state["hop_count"] >= MAX_HOPS:
        return "generate"
    if not state.get("follow_up_query"):
        return "generate"
    return "retrieve"


def generate_node(state: AgentState) -> dict:
    context = format_chunks(state["retrieved_chunks"])
    user_prompt = f"Question: {state['query']}\n\nSource excerpts:\n\n{context}"
    resp = chat_completion_with_retry(
        get_groq(),
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return {"final_answer": resp.choices[0].message.content}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_sufficiency", check_sufficiency_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency", should_continue, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("generate", END)
    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run(query: str) -> AgentState:
    initial_state: AgentState = {
        "query": query,
        "doc_type_filter": None,
        "retrieved_chunks": [],
        "hop_count": 0,
        "sufficient": False,
        "missing_info": None,
        "follow_up_query": None,
        "final_answer": None,
    }
    return get_app().invoke(initial_state)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent/graph.py "your question here"')
        sys.exit(1)
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Add it to .env.")
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    final_state = run(query)

    print(f"Doc-type filter used: {final_state['doc_type_filter']}")
    print(f"Hops taken: {final_state['hop_count']}")
    print(f"Total chunks retrieved: {len(final_state['retrieved_chunks'])}")
    print(f"Final sufficiency verdict: {final_state['sufficient']}")
    if final_state.get("missing_info"):
        print(f"Missing info noted: {final_state['missing_info']}")
    print()
    print("=" * 70)
    print(final_state["final_answer"])
    print("=" * 70)
