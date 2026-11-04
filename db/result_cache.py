"""Cached model results in the database: the fallback when Upstash Redis can't
be used (PLAN.md 8.1, api/result_cache.py).

The rows are opaque: a hash for a key and an encrypted answer for a value.
Every write deletes the expired rows and then, oldest first, whatever puts
the table over its byte budget, in one transaction, so the table never holds
more than the budget plus one answer.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from db.engine import transaction
from db.models import CachedResult, utc_now


def get(key: str, now: Optional[datetime] = None) -> Optional[bytes]:
    """The value stored under ``key``, or None when it is missing or expired."""
    now = now or utc_now()
    with transaction() as conn:
        return conn.execute(select(CachedResult.value).where(
            CachedResult.key == key, CachedResult.expires_at > now)).scalar_one_or_none()


def put(key: str, value: bytes, ttl_s: int, budget_bytes: int, now: Optional[datetime] = None) -> None:
    """Store ``value`` under ``key`` for ``ttl_s`` seconds, keeping the table within ``budget_bytes``."""
    now = now or utc_now()
    row = {"key": key, "value": value, "size": len(value), "created_at": now,
           "expires_at": now + timedelta(seconds=int(ttl_s))}
    stmt = insert(CachedResult).values(row)
    stmt = stmt.on_conflict_do_update(index_elements=[CachedResult.key], set_={
        "value": stmt.excluded.value, "size": stmt.excluded.size,
        "created_at": stmt.excluded.created_at, "expires_at": stmt.excluded.expires_at})
    # The rows whose newer neighbours (this one included) already fill the budget
    newer = func.sum(CachedResult.size).over(order_by=(CachedResult.created_at.desc(), CachedResult.key.desc()))
    ranked = select(CachedResult.key, newer.label("running")).subquery()
    over = select(ranked.c.key).where(ranked.c.running > budget_bytes)
    with transaction() as conn:
        conn.execute(delete(CachedResult).where(CachedResult.expires_at <= now))
        conn.execute(stmt)
        conn.execute(delete(CachedResult).where(CachedResult.key.in_(over)))


def usage() -> dict:
    """How many rows the table holds and their bytes (tests and the health check)."""
    with transaction() as conn:
        rows, size = conn.execute(select(func.count(), func.coalesce(func.sum(CachedResult.size), 0))).one()
    return {"rows": int(rows), "bytes": int(size)}
