"""
Renewal-window logic: how far ahead of expiry a document should be flagged
as "renew soon" depends on the process's real-world lead time - a passport
renewal (especially with police verification) takes far longer than
reprinting a PAN card, so a single fixed "30 days" threshold for everything
would either nag too early for fast processes or too late for slow ones.
"""

from datetime import date

from sqlalchemy.orm import Session

from tracker.models import TrackedDocument, list_documents

# Days before expiry to start flagging "renew soon", based on realistic
# processing time for each process type (see corpus notes: passport
# involves police verification and can take weeks to months; DL/PAN are
# comparatively fast).
RENEWAL_LEAD_DAYS = {
    "passport": 180,
    "driving_license": 30,
    "pan": 14,
    "scheme_scholarship": 30,
    "scheme_subsidy": 30,
}
DEFAULT_LEAD_DAYS = 30


def get_status(doc: TrackedDocument, today: date | None = None) -> dict:
    today = today or date.today()
    days_left = (doc.expiry_date - today).days
    lead_days = RENEWAL_LEAD_DAYS.get(doc.doc_type, DEFAULT_LEAD_DAYS)

    if days_left < 0:
        status = "expired"
    elif days_left <= lead_days:
        status = "renew_soon"
    else:
        status = "valid"

    return {"status": status, "days_left": days_left, "lead_days": lead_days}


def check_all(session: Session, owner_name: str | None = None) -> list[dict]:
    docs = list_documents(session, owner_name=owner_name)
    results = []
    for doc in docs:
        info = get_status(doc)
        results.append({
            "id": doc.id,
            "owner_name": doc.owner_name,
            "doc_type": doc.doc_type,
            "document_number": doc.document_number,
            "expiry_date": doc.expiry_date.isoformat(),
            **info,
        })
    return results


def get_alerts(session: Session, owner_name: str | None = None) -> list[dict]:
    """Just the documents that actually need attention - what a dashboard's
    'action needed' section, or a future notification job, would show."""
    return [r for r in check_all(session, owner_name) if r["status"] in ("expired", "renew_soon")]
