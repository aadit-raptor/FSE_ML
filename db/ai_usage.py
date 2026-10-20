"""What each person's AI requests used, a day (PLAN.md R9; table ``ai_usage``, migration 0023).

``record`` adds a read's or a writing's pages, requests, tokens and cost to the caller's row for the day in
**one statement** (an upsert keyed on the person, the day, the kind and the tier, pooler-safe), the person
matched by subject inside it: a subject with no account adds nothing. ``own`` reads only the caller's rows.
``totals`` sums a day for everyone without naming anyone (the nightly summary); ``sweep`` deletes rows past
``KEEP_DAYS``. Nothing from a document, a model's answer or a file's name is ever stored here.
"""
from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Iterable, Optional

from sqlalchemy import delete, func, literal, select
from sqlalchemy.dialects.postgresql import insert

from db.engine import connect, transaction
from db.models import AiUsage, User, utc_now
from research import ai_costs

KEEP_DAYS = 400          # a year of results seasons and a quarter to compare with
COLUMNS = ("day", "kind", "tier", "requests", "pages", "input_tokens", "output_tokens", "unpriced", "cost")


def _sums(answers: Iterable, day: datetime.date) -> dict:
    out = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0, "cost_amount": Decimal(0)}
    for answer in answers:
        given, made = ai_costs.tokens(answer.usage)
        priced = ai_costs.cost(answer.model, answer.usage, day)
        out["requests"] += 1
        out["input_tokens"] += given
        out["output_tokens"] += made
        if priced is None:
            out["unpriced"] += 1
        else:
            out["cost_amount"] += priced
    return out


def record(subject: str, kind: str, tier: str, *, pages: int = 0, answers: Iterable = (),
           day: Optional[datetime.date] = None) -> None:
    """Add pages and the answers' requests, tokens and cost to the person's row for ``day`` (today, UTC)."""
    day = day or utc_now().date()
    sums = _sums(answers, day)
    if not pages and not sums["requests"]:
        return
    values = {"pages": int(pages), **sums}
    chosen = select(User.id, literal(day), literal(kind), literal(tier),
                    *[literal(v) for v in values.values()], literal(ai_costs.CURRENCY), literal(utc_now())
                    ).where(User.subject == subject)
    stmt = insert(AiUsage).from_select(
        ["owner_id", "day", "kind", "tier", *values, "cost_currency", "updated_at"], chosen)
    stmt = stmt.on_conflict_do_update(
        index_elements=[AiUsage.owner_id, AiUsage.day, AiUsage.kind, AiUsage.tier],
        set_={**{name: getattr(AiUsage, name) + getattr(stmt.excluded, name) for name in values},
              "updated_at": stmt.excluded.updated_at})
    with transaction() as conn:
        conn.execute(stmt)


def own(subject: str, since: datetime.date) -> list[dict]:
    """The caller's rows from ``since`` on, newest first."""
    columns = [getattr(AiUsage, name) for name in COLUMNS[:-1]] + [AiUsage.cost_amount]
    query = select(*columns).join(User, User.id == AiUsage.owner_id).where(
        User.subject == subject, AiUsage.day >= since).order_by(AiUsage.day.desc(), AiUsage.kind, AiUsage.tier)
    with connect() as conn:
        return [dict(zip(COLUMNS, row)) for row in conn.execute(query)]


def totals(day: datetime.date) -> dict:
    """A day for everyone together: requests, pages, cost and how many people (nobody named)."""
    query = select(func.coalesce(func.sum(AiUsage.requests), 0), func.coalesce(func.sum(AiUsage.pages), 0),
                   func.coalesce(func.sum(AiUsage.cost_amount), 0), func.count(func.distinct(AiUsage.owner_id))
                   ).where(AiUsage.day == day)
    with connect() as conn:
        requests, pages, cost, people = conn.execute(query).one()
    return {"requests": int(requests), "pages": int(pages), "cost": Decimal(cost), "people": int(people)}


def sweep(today: Optional[datetime.date] = None) -> int:
    """Delete rows older than ``KEEP_DAYS``; how many went."""
    cutoff = (today or utc_now().date()) - datetime.timedelta(days=KEEP_DAYS)
    with transaction() as conn:
        return conn.execute(delete(AiUsage).where(AiUsage.day < cutoff)).rowcount or 0
