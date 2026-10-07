"""Model validation's storage (PLAN.md 4.6): the deals it may read and the reports.

``contributions`` is the one read of users' deals that is not the owner's
own: it runs inside the nightly ``validation-report`` task, reads only deals
whose owner opted in (``deals.validation_opt_in``) and have actuals, and
hands ``validation/`` their inputs, Settings, versions and actuals -- never
their ids, names or owners -- which the report then turns into counts and
rates per group. Nothing read here is logged or stored again.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import delete, insert, select

from db.engine import connect, transaction
from db.models import Deal, DealVersion, ValidationReport, utc_now
from validation.cases import Contribution, Version


# Deals read a night at most: each runs the deal model twice and a small
# simulation, and the task must finish within the scheduler's request. Past
# this the newest actuals win; phase 12's worker lifts it.
MAX_CONTRIBUTIONS = 400


def contributions(limit: int = MAX_CONTRIBUTIONS) -> list[Contribution]:
    """Opted-in deals with actuals, newest actuals first, with their
    versions (two round trips)."""
    with connect() as conn:
        deals = conn.execute(
            select(Deal.id, Deal.inputs, Deal.settings, Deal.created_at, Deal.actuals,
                   Deal.actuals_first_saved_at)
            .where(Deal.validation_opt_in.is_(True), Deal.actuals.is_not(None))
            .order_by(Deal.actuals_updated_at.desc(), Deal.id).limit(limit)).all()
        versions = conn.execute(
            select(DealVersion.deal_id, DealVersion.created_at, DealVersion.inputs, DealVersion.settings)
            .where(DealVersion.deal_id.in_([d.id for d in deals]))).all() if deals else []
    by_deal: dict = {}
    for v in versions:
        by_deal.setdefault(v.deal_id, []).append(Version(v.created_at, v.inputs, v.settings or {}))
    return [Contribution(inputs=d.inputs, settings=d.settings or {}, created_at=d.created_at,
                         actuals=d.actuals, actuals_first_saved_at=d.actuals_first_saved_at,
                         versions=tuple(by_deal.get(d.id, ())))
            for d in deals]


def save_report(report: dict, *, keep: int, now: Optional[datetime] = None) -> int:
    """Store a report and keep only the newest ``keep``; returns its id."""
    with transaction() as conn:
        report_id = conn.execute(insert(ValidationReport).values(
            generated_at=now or utc_now(), report=report).returning(ValidationReport.id)).scalar_one()
        newest = select(ValidationReport.id).order_by(ValidationReport.generated_at.desc(),
                                                      ValidationReport.id.desc()).limit(keep)
        conn.execute(delete(ValidationReport).where(ValidationReport.id.not_in(newest)))
    return report_id


def latest_report() -> Optional[dict]:
    """The newest report, or None before the first."""
    with connect() as conn:
        row = conn.execute(select(ValidationReport.report).order_by(
            ValidationReport.generated_at.desc(), ValidationReport.id.desc()).limit(1)).first()
    return None if row is None else row.report
