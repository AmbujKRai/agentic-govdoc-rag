"""
SQLite-backed document validity tracker. Deliberately minimal - no auth, no
notification infra (per the project plan). This exists to prove the
proactive-not-just-reactive part of the motive: the system should be able
to tell you "your DL needs renewing in 3 weeks" without you having to ask,
not just answer questions when prompted.

owner_name is free text rather than a real user account, so one person can
track documents for their whole household (parents, grandparents) - matches
the "helping family members navigate paperwork" part of the motive.
"""

from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Column, Date, DateTime, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tracker.db"

Base = declarative_base()


class TrackedDocument(Base):
    __tablename__ = "tracked_documents"

    id = Column(Integer, primary_key=True)
    owner_name = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)  # passport | driving_license | pan
    document_number = Column(String, nullable=True)
    issue_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def add_document(
    session: Session,
    owner_name: str,
    doc_type: str,
    expiry_date: date,
    issue_date: date | None = None,
    document_number: str | None = None,
    notes: str | None = None,
) -> TrackedDocument:
    doc = TrackedDocument(
        owner_name=owner_name,
        doc_type=doc_type,
        expiry_date=expiry_date,
        issue_date=issue_date,
        document_number=document_number,
        notes=notes,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def list_documents(session: Session, owner_name: str | None = None) -> list[TrackedDocument]:
    query = session.query(TrackedDocument)
    if owner_name:
        query = query.filter(TrackedDocument.owner_name == owner_name)
    return query.order_by(TrackedDocument.expiry_date).all()


def delete_document(session: Session, doc_id: int) -> bool:
    doc = session.get(TrackedDocument, doc_id)
    if doc is None:
        return False
    session.delete(doc)
    session.commit()
    return True
