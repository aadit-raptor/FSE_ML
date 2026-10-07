"""Stored industry averages and country tax rates (PLAN.md 4.3).

Public data, shared by every account: nothing here records who read it.

**Staying inside the free 0.5 GB.** Forty-one tables -- five data sets for
eight regions, and the country tax rates -- each the few figures a deal
starts from for about 85 industries: a few hundred kilobytes against
``BUDGET_BYTES``, which ``/api/health/database`` reports (``usage_on``).
A refresh replaces a table whole; there is no history to grow.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from benchmarks.damodaran import Table
from db.engine import connect, transaction
from db.models import BenchmarkTable, utc_now

BUDGET_BYTES = 4 * 1024 * 1024
BUDGET_WARN_FRACTION = 0.8
TABLES = ("benchmark_tables",)


def save(tables: list[Table]) -> int:
    """Store each table, replacing what was kept under its name."""
    if not tables:
        return 0
    now = utc_now()
    stmt = insert(BenchmarkTable).values([dict(name=t.name, published=t.published, url=t.url[:300],
                                               rows=dict(t.rows), refreshed_at=now) for t in tables])
    stmt = stmt.on_conflict_do_update(index_elements=["name"], set_={
        c: stmt.excluded[c] for c in ("published", "url", "rows", "refreshed_at")})
    with transaction() as conn:
        conn.execute(stmt)
    return len(tables)


def all_tables() -> tuple[dict[str, Table], Optional[datetime]]:
    """Every stored table by name, and when the oldest was refreshed (a
    refresh reads them all again once that is old enough)."""
    with connect() as conn:
        rows = conn.execute(select(BenchmarkTable)).all()
    tables = {r.name: Table(r.name, r.published, r.url, dict(r.rows)) for r in rows}
    return tables, min((r.refreshed_at for r in rows), default=None)


def usage_on(conn) -> dict:
    """Bytes the tables take against their budget, on an open connection
    (the database health check's)."""
    sql = text("SELECT COALESCE(SUM(pg_total_relation_size(to_regclass(t))), 0) FROM unnest(CAST(:tables AS text[])) AS t")
    used = int(conn.execute(sql, {"tables": list(TABLES)}).scalar_one())
    count = conn.execute(select(func.count()).select_from(BenchmarkTable)).scalar_one()
    return {"bytes": used, "budget_bytes": BUDGET_BYTES, "tables": count,
            "warning": used >= BUDGET_WARN_FRACTION * BUDGET_BYTES}


def usage() -> dict:
    with connect() as conn:
        return usage_on(conn)
