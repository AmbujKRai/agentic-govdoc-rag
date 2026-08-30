"""
Conversation memory for multi-turn chat.

The non-obvious part is NOT storing the history - it's that retrieval needs a
SELF-CONTAINED query. Embedding "what about for a minor?" retrieves garbage,
because that string means almost nothing on its own. So history is used to
REWRITE the turn into a standalone question BEFORE it ever reaches the
router or the retriever:

    history:  "What documents do I need for a passport?"
    new turn: "What about for a minor?"
       -> rewrite -> "What documents does a minor need for a passport?"
       -> route + retrieve normally

This is history-aware retrieval (a.k.a. contextual query rewriting). Only
the rewritten query touches retrieval; the raw history is never embedded.

Storage is SQLite (reusing the tracker's database file) so sessions survive
a process restart - an in-memory dict would silently lose every conversation
on the free-tier instance's frequent sleep/wake cycles.
"""

import sys
from datetime import datetime
from pathlib import Path

from groq import Groq
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.groq_utils import chat_completion_with_retry

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tracker.db"

Base = declarative_base()

# How many prior turns to feed the rewriter. Two is plenty for resolving
# pronouns and ellipsis, and keeps the prompt (and token spend) small.
HISTORY_TURNS = 2


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=engine)


def init_memory_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_history(db: Session, session_id: str, turns: int = HISTORY_TURNS) -> list[ChatTurn]:
    """Most recent `turns` user+assistant pairs, oldest first."""
    rows = (
        db.query(ChatTurn)
        .filter(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.id.desc())
        .limit(turns * 2)
        .all()
    )
    return list(reversed(rows))


def append_turn(db: Session, session_id: str, role: str, content: str) -> None:
    db.add(ChatTurn(session_id=session_id, role=role, content=content))
    db.commit()


def clear_session(db: Session, session_id: str) -> int:
    n = db.query(ChatTurn).filter(ChatTurn.session_id == session_id).delete()
    db.commit()
    return n


REWRITE_SYSTEM_PROMPT = """Rewrite the user's latest message into a standalone \
question that makes full sense with no conversation history.

Rules:
- Resolve pronouns and ellipsis using the history ("what about for a minor?" \
after a passport question becomes "What documents does a minor need for a passport?").
- If the latest message is ALREADY self-contained, return it unchanged.
- If it changes the subject entirely, return it unchanged - do not force it to \
relate to the previous topic.
- Return ONLY the rewritten question. No preamble, no quotes, no explanation."""


def rewrite_query(groq_client: Groq, model: str, history: list[ChatTurn], query: str) -> str:
    """Turn a context-dependent follow-up into a standalone question.

    Returns `query` unchanged when there's no history to resolve against, so
    a first turn costs no extra LLM call.
    """
    if not history:
        return query

    transcript = "\n".join(f"{t.role}: {t.content}" for t in history)
    user_prompt = f"Conversation so far:\n{transcript}\n\nLatest message: {query}\n\nStandalone question:"

    try:
        resp = chat_completion_with_retry(
            groq_client,
            purpose="query_rewrite",
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
    except Exception:
        # A failed rewrite must not take down the whole request - falling back
        # to the raw query gives a worse answer, not no answer.
        return query

    # Guard against a model that ignores instructions and returns prose or an
    # empty string; a wildly long "question" is a rewrite that went wrong.
    if not rewritten or len(rewritten) > 4 * len(query) + 200:
        return query
    return rewritten
