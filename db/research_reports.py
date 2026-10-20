"""Results notes and the account's report details (PLAN.md R8).

One note per covered company: what the analyst typed for it and the linking
sentences last written (never a figure: those are read from the saved model
when the note is drawn). The firm name and disclosures are the account's,
typed once. Every function matches the caller's subject in the statement
that reads or writes; another account's company is ``NotFound``. Each change
writes one audit entry (``report_saved`` with the names of the fields
changed, ``report_details_saved``), none when nothing changed.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db import audit
from db.coverage import NoAccount, NotFound
from db.engine import connect, transaction
from db.models import CoveredCompany, ResearchReport, ResearchReportDetails, User, utc_now

MAX_FIRM = 200
MAX_DISCLOSURES = 8000
FIELD_NAMES = ("argument", "rating", "target", "note_date", "compare_version")
# A note nothing is typed for yet (an empty text and a missing one are the same)
EMPTY = {"argument": "", "rating": "", "target": None, "note_date": None, "compare_version": None}


def _owner(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def _owned(conn, subject: str, company_id: uuid.UUID):
    row = conn.execute(select(CoveredCompany.id, CoveredCompany.owner_id).where(
        CoveredCompany.id == company_id, CoveredCompany.owner_id == _owner(subject))).first()
    if row is None:
        raise NotFound(str(company_id))
    return row


def get_report(subject: str, company_id: uuid.UUID) -> tuple[dict, Optional[dict]]:
    """The note's fields and sentences; empty fields for a company with no note yet."""
    with connect() as conn:
        _owned(conn, subject, company_id)
        row = conn.execute(select(ResearchReport.fields, ResearchReport.sentences).where(
            ResearchReport.company_id == company_id, ResearchReport.owner_id == _owner(subject))).first()
    return (dict(row.fields), row.sentences) if row else ({}, None)


_KEEP = object()


def save_report(subject: str, company_id: uuid.UUID, fields=_KEEP, sentences=_KEEP) -> list[str]:
    """Keep the note's fields and its sentences (either left as stored when not given: the sentences are written
    while the analyst may be typing, so they never carry the fields along). Answers the names of what changed."""
    now = utc_now()
    with transaction() as conn:
        owned = _owned(conn, subject, company_id)
        row = conn.execute(select(ResearchReport.id, ResearchReport.fields, ResearchReport.sentences).where(
            ResearchReport.company_id == company_id).with_for_update()).first()
        old_fields = dict(row.fields) if row else dict(EMPTY)
        if fields is _KEEP:
            fields = old_fields
        fields = {k: EMPTY[k] if fields.get(k) is None else fields.get(k) for k in FIELD_NAMES}
        old_sentences = row.sentences if row else None
        new_sentences = old_sentences if sentences is _KEEP else sentences
        changed = [k for k in FIELD_NAMES if old_fields.get(k, EMPTY[k]) != fields[k]]
        if new_sentences != old_sentences:
            changed.append("sentences")
        if not changed and row is not None:
            return []
        if row is None:
            conn.execute(insert(ResearchReport).values(id=uuid.uuid4(), company_id=company_id,
                                                       owner_id=owned.owner_id, fields=fields,
                                                       sentences=new_sentences, created_at=now, updated_at=now))
        else:
            conn.execute(update(ResearchReport).where(ResearchReport.id == row.id).values(
                fields=fields, sentences=new_sentences, updated_at=now))
        audit.record(conn, owned.owner_id, "report_saved", now=now, detail={"fields": changed or ["created"]})
        return changed or ["created"]


def get_details(subject: str) -> tuple[str, str]:
    with connect() as conn:
        row = conn.execute(select(ResearchReportDetails.firm, ResearchReportDetails.disclosures).where(
            ResearchReportDetails.owner_id == _owner(subject))).first()
    return (row.firm, row.disclosures) if row else ("", "")


def save_details(subject: str, firm: str, disclosures: str) -> bool:
    """Keep the account's firm name and disclosures; True when they changed."""
    firm, disclosures = " ".join((firm or "").split())[:MAX_FIRM], (disclosures or "")[:MAX_DISCLOSURES]
    now = utc_now()
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject)).scalar()
        if owner is None:
            raise NoAccount(subject)
        old = conn.execute(select(ResearchReportDetails.firm, ResearchReportDetails.disclosures).where(
            ResearchReportDetails.owner_id == owner).with_for_update()).first()
        if old is not None and (old.firm, old.disclosures) == (firm, disclosures):
            return False
        conn.execute(pg_insert(ResearchReportDetails).values(owner_id=owner, firm=firm, disclosures=disclosures,
                                                             updated_at=now)
                     .on_conflict_do_update(index_elements=[ResearchReportDetails.owner_id],
                                            set_={"firm": firm, "disclosures": disclosures, "updated_at": now}))
        audit.record(conn, owner, "report_details_saved", now=now)
        return True


__all__ = ["EMPTY", "FIELD_NAMES", "NoAccount", "NotFound", "get_details", "get_report", "save_details", "save_report"]
