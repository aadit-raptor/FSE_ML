"""Library -> Review: proposing reference transactions and the two-person approval (PLAN.md 4.5b).

Only administrators (``library.switch.is_admin``) propose or review. The
repository's own proposals (``reference_deals.json``) are added to the queue
the first time an administrator opens it, and again whenever the file gains
a transaction or a corrected version of one. Each proposal comes with what
the inclusion rules find (blocking) and what the balance rules say
(advisory) against the library as it stands; the stored state and who may
do what are ``db/references.py``.
"""
from __future__ import annotations

import uuid
from typing import Optional

from db import references as store
from library import references, switch


class NotAdmin(PermissionError):
    """Only an administrator proposes or reviews reference transactions."""


def _require_admin(subject: str) -> None:
    if not switch.is_admin(subject):
        raise NotAdmin(subject)


def sync() -> int:
    """Queue the repository's transactions that are not stored yet."""
    deals = references.repository_deals()
    return store.sync_repository([(d["key"], references.content_hash(d), d) for d in deals])


def library_deals() -> list[dict]:
    return store.approved()


def _view(p: store.Proposal, library: list[dict]) -> dict:
    others = [d for d in library if d["key"] != p.key]
    return {
        "id": str(p.id), "key": p.key, "origin": p.origin, "status": p.status,
        "proposed_at": p.created_at, "decided_at": p.decided_at,
        "approvals": p.approvals, "approvals_needed": store.APPROVALS_NEEDED, "rejections": p.rejections,
        "reasons": list(p.reasons), "mine": p.mine, "my_verdict": p.my_verdict,
        "replaces_approved": len(others) < len(library),
        "deal": references.summary(p.content),
        "problems": references.problems(p.content),
        "balance": references.balance(others, p.content),
    }


def queue(subject: str) -> dict:
    """What an administrator reviews: every open proposal, oldest first,
    and the ones decided lately (so a decision stays visible)."""
    _require_admin(subject)
    sync()
    library = library_deals()
    open_ = store.proposals(subject, ("proposed",))
    decided = store.proposals(subject, ("approved", "rejected"))
    return {
        "proposals": [_view(p, library) for p in open_],
        "decided": [_view(p, library) for p in reversed(decided[-20:])],
        "library_size": len(library),
    }


def propose(subject: str, deal: dict) -> dict:
    """Store an administrator's proposal; answers it as the queue shows it.
    A proposal that breaks a rule is still stored (the reviewers see why),
    but it can't be approved until corrected and proposed again."""
    _require_admin(subject)
    new_id = store.propose(subject, deal["key"], references.content_hash(deal), deal)
    return find(subject, new_id)


def decide(subject: str, reference_id: uuid.UUID, verdict: str, reason: Optional[str]) -> dict:
    _require_admin(subject)
    store.review(subject, reference_id, verdict, reason=reason, check=references.problems)
    return find(subject, reference_id)


def find(subject: str, reference_id: uuid.UUID) -> dict:
    found = store.proposals(subject, ("proposed", "approved", "rejected", "superseded"), only=reference_id)
    if not found:
        raise store.NotFound(reference_id)
    return _view(found[0], library_deals())
