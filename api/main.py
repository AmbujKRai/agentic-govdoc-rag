"""
FastAPI app wrapping the agent graph (/chat) and the document tracker
(/documents). This is what turns the collection of CLI scripts built so far
into an actual runnable product.

Run locally:
    uvicorn api.main:app --reload

Endpoints are plain `def` (not `async def`) - the agent graph and Groq calls
are synchronous/blocking, and FastAPI automatically runs sync path
operations in a threadpool, which is the correct way to wrap blocking code
here without adding async complexity that buys nothing (Groq's SDK isn't
async in this project anyway).
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agent.graph import run as run_agentic
from tracker.models import SessionLocal, add_document, delete_document, init_db
from tracker.reminder import check_all, get_alerts

app = FastAPI(
    title="GovDoc Copilot",
    description="Agentic RAG over Indian government document processes and welfare schemes.",
)


@app.on_event("startup")
def on_startup():
    init_db()


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------- /chat ----------

class ChatRequest(BaseModel):
    query: str


class Source(BaseModel):
    doc_id: str
    title: str
    source_url: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    hops: int
    sufficient: bool
    doc_type_filter: str | None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    state = run_agentic(req.query)

    seen_doc_ids = set()
    sources = []
    for c in state["retrieved_chunks"]:
        if c["doc_id"] in seen_doc_ids:
            continue
        seen_doc_ids.add(c["doc_id"])
        sources.append(Source(doc_id=c["doc_id"], title=c["title"], source_url=c["source_url"], page=c.get("page")))

    return ChatResponse(
        answer=state["final_answer"] or "",
        sources=sources,
        hops=state["hop_count"],
        sufficient=state["sufficient"],
        doc_type_filter=state["doc_type_filter"],
    )


# ---------- /documents ----------

class DocumentCreate(BaseModel):
    owner_name: str
    doc_type: str
    expiry_date: date
    issue_date: date | None = None
    document_number: str | None = None
    notes: str | None = None


class DocumentStatus(BaseModel):
    id: int
    owner_name: str
    doc_type: str
    document_number: str | None
    expiry_date: str
    status: str
    days_left: int
    lead_days: int


@app.post("/documents", response_model=DocumentStatus, status_code=201)
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    created = add_document(
        db,
        owner_name=doc.owner_name,
        doc_type=doc.doc_type,
        expiry_date=doc.expiry_date,
        issue_date=doc.issue_date,
        document_number=doc.document_number,
        notes=doc.notes,
    )
    matches = [r for r in check_all(db) if r["id"] == created.id]
    return matches[0]


@app.get("/documents", response_model=list[DocumentStatus])
def list_documents_endpoint(owner_name: str | None = None, db: Session = Depends(get_db)):
    return check_all(db, owner_name=owner_name)


@app.get("/documents/alerts", response_model=list[DocumentStatus])
def alerts_endpoint(owner_name: str | None = None, db: Session = Depends(get_db)):
    return get_alerts(db, owner_name=owner_name)


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document_endpoint(doc_id: int, db: Session = Depends(get_db)):
    if not delete_document(db, doc_id):
        raise HTTPException(status_code=404, detail="document not found")


# ---------- health ----------

@app.get("/health")
def health():
    return {"status": "ok"}
