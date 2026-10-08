"""Stored industry averages and country tax rates (PLAN.md 4.3).

Public data, shared by every account: nothing here records who read it.

**Staying inside the free 0.5 GB.** Forty-one tables -- five data sets for
eight regions, and the country tax rates -- each the few figures a deal
starts from for about 85 industries, plus the history the risk ranges are
measured on (PLAN.md 4.4, benchmarks/history.py): one table per group with
three figures an industry a year since 2011, the economic history and the
archive's read log. About a megabyte against ``BUDGET_BYTES``, which
``/api/health/database`` reports (``usage_on``). A refresh replaces a table
whole; the history grows by one year each January.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from benchmarks.damodaran import Table
from db.engine import connect, transaction
from db.models import BenchmarkTable, utc_now

BUDGET_BYTES = 8 * 1024 * 1024
BUDGET_WARN_FRACTION = 0.8
TABLES = ("benchmark_tables",)


def save(tables: list[Table], now: Optional[datetime] = None) -> int:
    """Store each table, replacing what was kept under its name, as read at
    ``now`` (the refresh's own clock, which also judges what is fresh)."""
    if not tables:
        return 0
    now = now or utc_now()
    stmt = insert(BenchmarkTable).values([dict(name=t.name, published=t.published, url=t.url[:300],
                                               rows=dict(t.rows), refreshed_at=now) for t in tables])
    stmt = stmt.on_conflict_do_update(index_elements=["name"], set_={
        c: stmt.excluded[c] for c in ("published", "url", "rows", "refreshed_at")})
    with transaction() as conn:
        conn.execute(stmt)
    return len(tables)


HISTORY_PREFIX = "history."


def all_tables(history: bool = False, history_groups: Optional[Iterable[str]] = None
               ) -> tuple[dict[str, Table], Optional[datetime]]:
    """Every stored table by name, and when the oldest current one was
    refreshed (a refresh reads them all again once that is old enough).
    The history tables (``history.*``) only when asked for: they are the
    larger ones, and only the risk ranges and the multiple predictor read
    them; ``history_groups`` reads only those groups' ``history.<group>``."""
    query = select(BenchmarkTable)
    if history_groups is not None:
        wanted = [f"{HISTORY_PREFIX}{g}" for g in history_groups]
        query = query.where(~BenchmarkTable.name.startswith(HISTORY_PREFIX) | BenchmarkTable.name.in_(wanted))
    elif not history:
        query = query.where(~BenchmarkTable.name.startswith(HISTORY_PREFIX))
    with connect() as conn:
        rows = conn.execute(query).all()
    tables = {r.name: Table(r.name, r.published, r.url, dict(r.rows)) for r in rows}
    current = [r.refreshed_at for r in rows if not r.name.startswith(HISTORY_PREFIX)]
    return tables, min(current, default=None)


def usage_on(conn) -> dict:
    """Bytes the tables take against their budget, on an open connection
    (the database health check's)."""
    sql = text("SELECT COALESCE(SUM(pg_total_relation_size(to_regclass(t))), 0) FROM unnest(CAST(:tables AS text[])) AS t")
    used = int(conn.execute(sql, {"tables": list(TABLES)}).scalar_one())
    history = BenchmarkTable.name.startswith(HISTORY_PREFIX)
    count = conn.execute(select(func.count()).select_from(BenchmarkTable).where(~history)).scalar_one()
    kept = conn.execute(select(func.count()).select_from(BenchmarkTable).where(history)).scalar_one()
    return {"bytes": used, "budget_bytes": BUDGET_BYTES, "tables": count, "history_tables": kept,
            "warning": used >= BUDGET_WARN_FRACTION * BUDGET_BYTES}


def usage() -> dict:
    with connect() as conn:
        return usage_on(conn)
