"""Reference transactions and their two-person review, stored (PLAN.md 4.5b).

``reference_deals`` holds every proposal and ``reference_reviews`` every
administrator's verdict on one. The rules about *who* may review live here,
inside the transaction that records the verdict, so two administrators
acting at once can't both be the deciding second approval or approve
something already rejected:

- a proposal is decided by administrators other than its proposer (the
  repository's proposals have no proposer, so any two administrators);
- each administrator gives one verdict per proposal;
- the second approval makes it ``approved`` (and supersedes an older approved
  version of the same transaction); the first rejection makes it
  ``rejected``;
- an approval is refused while the inclusion rules find a problem
  (``check`` returns them; ``library/references.py`` holds the rules).

Whether the caller *is* an administrator is ``library.switch.is_admin``,
checked by ``library/review.py`` before anything here runs. Every proposal
and verdict writes one audit entry (``reference_proposed``,
``reference_reviewed``) in the same transaction. Public filing data, so
nothing personal is stored beyond which account proposed or reviewed.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db import audit
from db.engine import connect, transaction
from db.flags import NoAccount
from db.models import ReferenceDeal, ReferenceReview, User, utc_now

APPROVALS_NEEDED = 2
# Proposals kept at most (each a few KB): the free 0.5 GB stays safe
MAX_PROPOSALS = 2000


class NotFound(LookupError):
    """No such proposal."""


class AlreadyDecided(RuntimeError):
    """The proposal was already approved, rejected or superseded."""


class OwnProposal(PermissionError):
    """A proposer can't review their own proposal."""


class AlreadyReviewed(RuntimeError):
    """This administrator already gave a verdict on it."""


class Duplicate(RuntimeError):
    """The same content is already proposed or in the library."""


class TooMany(RuntimeError):
    """The table holds ``MAX_PROPOSALS`` already."""


class Blocked(RuntimeError):
    """The inclusion rules found problems; it can't be approved."""

    def __init__(self, problems: list[dict]):
        super().__init__("the inclusion rules refuse it")
        self.problems = problems


@dataclass(frozen=True)
class Proposal:
    id: uuid.UUID
    key: str
    content: dict
    origin: str
    status: str
    created_at: datetime
    decided_at: Optional[datetime]
    approvals: int
    rejections: int
    reasons: tuple[str, ...]
    mine: bool
    my_verdict: Optional[str]


def _user_id(conn, subject: str) -> int:
    uid = conn.execute(select(User.id).where(User.subject == subject)).scalar()
    if uid is None:
        raise NoAccount(subject)
    return uid


def sync_repository(deals: list[tuple[str, str, dict]], now: Optional[datetime] = None) -> int:
    """Propose the repository's transactions (``(key, content_hash,
    content)``) that are not stored yet; returns how many were added. One
    statement, so it is safe to call on every read of the review queue."""
    if not deals:
        return 0
    rows = [{"id": uuid.uuid4(), "key": k, "content_hash": h, "content": c, "origin": "repository",
             "status": "proposed", "created_at": now or utc_now()} for k, h, c in deals]
    with transaction() as conn:
        added = conn.execute(pg_insert(ReferenceDeal).values(rows).on_conflict_do_nothing(
            index_elements=[ReferenceDeal.key, ReferenceDeal.content_hash]).returning(ReferenceDeal.id)).all()
    return len(added)


def propose(subject: str, key: str, content_hash: str, content: dict, now: Optional[datetime] = None) -> uuid.UUID:
    """An administrator's proposal; one ``reference_proposed`` audit entry."""
    now = now or utc_now()
    with transaction() as conn:
        uid = _user_id(conn, subject)
        if conn.execute(select(func.count()).select_from(ReferenceDeal)).scalar_one() >= MAX_PROPOSALS:
            raise TooMany()
        new_id = uuid.uuid4()
        added = conn.execute(pg_insert(ReferenceDeal).values(
            id=new_id, key=key, content_hash=content_hash, content=content, origin="user", status="proposed",
            proposed_by=uid, created_at=now,
        ).on_conflict_do_nothing(index_elements=[ReferenceDeal.key, ReferenceDeal.content_hash])
            .returning(ReferenceDeal.id)).first()
        if added is None:
            raise Duplicate(key)
        audit.record(conn, uid, "reference_proposed", now=now, detail={"reference": str(new_id)})
    return new_id


def review(subject: str, reference_id: uuid.UUID, verdict: str, *, reason: Optional[str] = None,
           check: Callable[[dict], list[dict]], now: Optional[datetime] = None) -> str:
    """Record ``subject``'s verdict and answer the proposal's status after it."""
    now = now or utc_now()
    with transaction() as conn:
        uid = _user_id(conn, subject)
        row = conn.execute(select(ReferenceDeal.key, ReferenceDeal.content, ReferenceDeal.status,
                                  ReferenceDeal.proposed_by)
                           .where(ReferenceDeal.id == reference_id).with_for_update()).first()
        if row is None:
            raise NotFound(reference_id)
        if row.status != "proposed":
            raise AlreadyDecided(row.status)
        if row.proposed_by == uid:
            raise OwnProposal()
        if conn.execute(select(ReferenceReview.id).where(
                ReferenceReview.reference_id == reference_id, ReferenceReview.reviewer_id == uid)).first():
            raise AlreadyReviewed()
        if verdict == "approve":
            found = check(row.content)
            if found:
                raise Blocked(found)
        conn.execute(insert(ReferenceReview).values(
            reference_id=reference_id, reviewer_id=uid, verdict=verdict,
            reason=reason if verdict == "reject" else None, created_at=now))
        audit.record(conn, uid, "reference_reviewed", now=now,
                     detail={"reference": str(reference_id), "verdict": verdict})
        status = "proposed"
        if verdict == "reject":
            status = "rejected"
        elif conn.execute(select(func.count()).select_from(ReferenceReview).where(
                ReferenceReview.reference_id == reference_id,
                ReferenceReview.verdict == "approve")).scalar_one() >= APPROVALS_NEEDED:
            status = "approved"
            conn.execute(update(ReferenceDeal).where(
                ReferenceDeal.key == row.key, ReferenceDeal.status == "approved",
                ReferenceDeal.id != reference_id).values(status="superseded", decided_at=now))
        if status != "proposed":
            conn.execute(update(ReferenceDeal).where(ReferenceDeal.id == reference_id)
                         .values(status=status, decided_at=now))
    return status


def _tally():
    """Per proposal: approvals, rejections and the rejection reasons."""
    r = ReferenceReview
    return (select(r.reference_id,
                   func.count().filter(r.verdict == "approve").label("approvals"),
                   func.count().filter(r.verdict == "reject").label("rejections"),
                   func.array_remove(func.array_agg(r.reason), None).label("reasons"))
            .group_by(r.reference_id).subquery())


def proposals(subject: Optional[str], statuses: tuple[str, ...], *,
              only: Optional[uuid.UUID] = None) -> list[Proposal]:
    """Proposals in ``statuses``, oldest first, each with its tally and what
    the caller has to do with it (their own? already reviewed?)."""
    tally = _tally()
    me = select(User.id).where(User.subject == subject).scalar_subquery() if subject else None
    mine_review = (select(ReferenceReview.verdict)
                   .where(ReferenceReview.reference_id == ReferenceDeal.id, ReferenceReview.reviewer_id == me)
                   .scalar_subquery()) if subject else None
    cols = [ReferenceDeal.id, ReferenceDeal.key, ReferenceDeal.content, ReferenceDeal.origin, ReferenceDeal.status,
            ReferenceDeal.created_at, ReferenceDeal.decided_at,
            func.coalesce(tally.c.approvals, 0).label("approvals"),
            func.coalesce(tally.c.rejections, 0).label("rejections"), tally.c.reasons]
    if subject:
        cols += [and_(ReferenceDeal.proposed_by.is_not(None), ReferenceDeal.proposed_by == me).label("mine"),
                 mine_review.label("my_verdict")]
    query = (select(*cols).select_from(ReferenceDeal)
             .outerjoin(tally, tally.c.reference_id == ReferenceDeal.id)
             .where(ReferenceDeal.status.in_(statuses))
             .order_by(ReferenceDeal.created_at, ReferenceDeal.key))
    if only is not None:
        query = query.where(ReferenceDeal.id == only)
    with connect() as conn:
        rows = conn.execute(query).all()
    return [Proposal(id=r.id, key=r.key, content=r.content, origin=r.origin, status=r.status,
                     created_at=r.created_at, decided_at=r.decided_at, approvals=r.approvals,
                     rejections=r.rejections, reasons=tuple(r.reasons or ()),
                     mine=bool(getattr(r, "mine", False)), my_verdict=getattr(r, "my_verdict", None))
            for r in rows]


def approved() -> list[dict]:
    """The library's transactions: the content of every approved proposal,
    with when it was approved."""
    with connect() as conn:
        rows = conn.execute(select(ReferenceDeal.id, ReferenceDeal.content, ReferenceDeal.decided_at)
                            .where(ReferenceDeal.status == "approved").order_by(ReferenceDeal.key)).all()
    return [{**r.content, "id": str(r.id), "approved_at": r.decided_at} for r in rows]


def counts() -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute(select(ReferenceDeal.status, func.count()).group_by(ReferenceDeal.status)).all()
    return {status: n for status, n in rows}
