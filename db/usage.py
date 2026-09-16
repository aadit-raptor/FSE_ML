"""Usage counters in the database: the fallback when Upstash Redis can't be
used (PLAN.md 1.6, api/usage.py).

One statement adds a whole batch of increments and returns the new totals;
expired counters are deleted in the same transaction, so the table never
grows past the counters that are live.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert

from db.engine import transaction
from db.models import UsageCounter, utc_now


def add_counts(batch: Sequence[tuple[str, int, int]], now: Optional[datetime] = None) -> dict[str, int]:
    """Add ``(key, increment, seconds to keep)`` to each counter; the new totals by key."""
    if not batch:
        return {}
    now = now or utc_now()
    rows = [{"key": key, "count": int(delta), "expires_at": now + timedelta(seconds=int(ttl))}
            for key, delta, ttl in batch]
    stmt = insert(UsageCounter).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[UsageCounter.key],
        set_={"count": UsageCounter.count + stmt.excluded.count,
              "expires_at": stmt.excluded.expires_at},
    ).returning(UsageCounter.key, UsageCounter.count)
    with transaction() as conn:
        conn.execute(delete(UsageCounter).where(UsageCounter.expires_at <= now))
        return {key: int(count) for key, count in conn.execute(stmt)}
