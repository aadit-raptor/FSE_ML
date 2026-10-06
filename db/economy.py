"""Stored economic series and exchange rates (PLAN.md 4.2).

Public data, shared by every account: nothing here records who read it.

**Staying inside the free 0.5 GB.** A series keeps its newest observations
only (economy.model.KEEP_OBSERVATIONS), about 200 series in all, and the
ECB's rates keep ``KEEP_FX_DAYS`` of days, one row a day. Together that is a
few hundred kilobytes against ``BUDGET_BYTES``, which ``/api/health/database``
reports (``usage_on``).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert

from db.engine import connect, transaction
from db.models import EconomicSeries, ExchangeRateDay, utc_now
from economy.model import FxDay, Series

KEEP_FX_DAYS = 400
BUDGET_BYTES = 8 * 1024 * 1024
BUDGET_WARN_FRACTION = 0.8
TABLES = ("economic_series", "exchange_rates")


def save_series(series: list[Series]) -> int:
    """Store each series, replacing what was kept under its key."""
    if not series:
        return 0
    now = utc_now()
    rows = [dict(key=s.key, indicator=s.indicator, area=s.area, source=s.source,
                 source_series=s.source_series[:80], frequency=s.frequency, url=s.url[:300],
                 observations=[[p, v] for p, v in s.observations], refreshed_at=now) for s in series]
    stmt = insert(EconomicSeries).values(rows)
    stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={
        c: stmt.excluded[c] for c in ("indicator", "area", "source", "source_series", "frequency", "url",
                                      "observations", "refreshed_at")})
    with transaction() as conn:
        conn.execute(stmt)
    return len(rows)


def all_series() -> tuple[dict[str, Series], Optional[datetime]]:
    """Every stored series by key, and when the newest was refreshed."""
    with connect() as conn:
        rows = conn.execute(select(EconomicSeries)).all()
    store = {r.key: Series(r.indicator, r.area, r.source, r.source_series, r.frequency, r.url,
                           tuple((p, v) for p, v in r.observations)) for r in rows}
    return store, max((r.refreshed_at for r in rows), default=None)


def save_fx(days: list[FxDay]) -> int:
    if not days:
        return 0
    now = utc_now()
    stmt = insert(ExchangeRateDay).values([dict(rate_date=d.day, rates=d.rates, refreshed_at=now) for d in days])
    stmt = stmt.on_conflict_do_update(index_elements=["rate_date"],
                                      set_={"rates": stmt.excluded.rates, "refreshed_at": now})
    with transaction() as conn:
        conn.execute(stmt)
    return len(days)


def latest_fx_day() -> Optional[date]:
    with connect() as conn:
        return conn.execute(select(func.max(ExchangeRateDay.rate_date))).scalar()


def fx_on(day: Optional[date] = None) -> Optional[FxDay]:
    """The rates of ``day``, or of the last day before it the ECB published
    (weekends and TARGET holidays have none); the latest without a day."""
    query = select(ExchangeRateDay).order_by(ExchangeRateDay.rate_date.desc()).limit(1)
    if day is not None:
        query = query.where(ExchangeRateDay.rate_date <= day)
    with connect() as conn:
        row = conn.execute(query).one_or_none()
    return None if row is None else FxDay(row.rate_date, dict(row.rates))


def prune_fx(today: date) -> int:
    with transaction() as conn:
        return conn.execute(delete(ExchangeRateDay).where(
            ExchangeRateDay.rate_date < today - timedelta(days=KEEP_FX_DAYS))).rowcount


def usage_on(conn) -> dict:
    """Bytes the economic tables take against their budget, on an open
    connection (the database health check's)."""
    sql = text("SELECT COALESCE(SUM(pg_total_relation_size(to_regclass(t))), 0) FROM unnest(CAST(:tables AS text[])) AS t")
    used = int(conn.execute(sql, {"tables": list(TABLES)}).scalar_one())
    series = conn.execute(select(func.count()).select_from(EconomicSeries)).scalar_one()
    days = conn.execute(select(func.count()).select_from(ExchangeRateDay)).scalar_one()
    return {"bytes": used, "budget_bytes": BUDGET_BYTES, "series": series, "fx_days": days,
            "warning": used >= BUDGET_WARN_FRACTION * BUDGET_BYTES}


def usage() -> dict:
    with connect() as conn:
        return usage_on(conn)
