"""Owned deals, their quarterly results and reforecasts (PLAN.md 7.4).

A holding is a saved deal marked as owned. Everything here lives on the deal's own row (``deals.portfolio``,
``quarters``, ``quarters_updated_at``, ``reforecast``) and is reached as the deal is (db/deals.py): its owner and
the editors of a team it is shared with may change it, a viewer may read it, and anyone else gets
``DealNotFound``, matched in the same statement as every read and write. Like plan vs actual's actuals it is not
part of the plan: no version is written and the deal's edit time stays.

**Staying small.** A holding is two values, a year of quarters a few hundred bytes, a reforecast about a
kilobyte; ``MAX_OWNED`` holdings an owner, ``MAX_QUARTERS`` quarters a deal (fifteen years).

Every change writes one audit entry (db/audit.py) in its transaction, and a change that changes nothing none.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional

from pydantic import ValidationError
from sqlalchemy import func, null, or_, select, update

from core.money import DEFAULT_CURRENCY
from core.portfolio import DEFAULT_TOLERANCE, QuartersRefused
from db import audit
from db.deals import (
    EDIT, READ, READ_ROLES, DealNotFound, InvalidDeal, NotPermitted, _access, _allowed, _lock_reachable,
    _owner_id, _reachable, _shared,
)
from db.engine import connect, transaction
from db.models import Deal, utc_now

MAX_OWNED = 50              # holdings an owner may mark
MAX_LISTED = 100            # holdings the dashboard reads (an owner's and those shared with them)
MAX_QUARTERS = 60


class PortfolioFull(Exception):
    pass


@dataclass(frozen=True)
class Holding:
    deal_id: uuid.UUID
    name: str
    inputs: dict
    settings: dict
    portfolio: Optional[dict]
    quarters: Optional[dict]
    quarters_updated_at: Optional[datetime]
    reforecast: Optional[dict]
    access: str


_COLUMNS = (Deal.id, Deal.name, Deal.inputs, Deal.settings, Deal.portfolio, Deal.quarters,
            Deal.quarters_updated_at, Deal.reforecast)


def _holding(row) -> Holding:
    return Holding(deal_id=row.id, name=row.name, inputs=row.inputs, settings=row.settings,
                   portfolio=row.portfolio, quarters=row.quarters, quarters_updated_at=row.quarters_updated_at,
                   reforecast=row.reforecast, access=row.access)


def clean_holding(since: Optional[date], tolerance: Optional[float]) -> dict:
    tolerance = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    if not 0 <= tolerance <= 100:
        raise InvalidDeal("tolerance: between 0 and 100")
    return {"since": since.isoformat() if since else None, "tolerance": tolerance}


def clean_quarters(quarters: Mapping) -> dict:
    """Quarters in ``api.schemas.QuarterlyActuals``' shape, unknown figures left out, oldest first."""
    from api.schemas import QuarterlyActuals

    try:
        cleaned = QuarterlyActuals.model_validate(dict(quarters or {}))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "quarters"
        raise InvalidDeal(f"quarters.{where}: {first.get('msg', 'invalid')}") from None
    out = cleaned.model_dump(exclude_none=True, mode="json")
    out["quarters"] = sorted(out["quarters"], key=lambda q: (q["year"], q["quarter"]))
    return out


def get_holding(subject: str, deal_id: uuid.UUID, need: str = READ) -> Holding:
    """The deal's holding, if the caller may ``need`` (db/deals.py ``READ`` or ``EDIT``) it."""
    with connect() as conn:
        row = conn.execute(select(*_COLUMNS, _access(subject)).where(_reachable(subject, deal_id))).first()
    return _holding(_allowed(row, deal_id, need))


def holdings(subject: str) -> list[Holding]:
    """The deals in the caller's portfolio and their teams' (not archived), by name; at most ``MAX_LISTED``."""
    query = (select(*_COLUMNS, _access(subject))
             .where(or_(Deal.owner_id == _owner_id(subject), _shared(subject, READ_ROLES)),
                    Deal.portfolio.is_not(None), Deal.archived_at.is_(None))
             .order_by(Deal.name, Deal.id).limit(MAX_LISTED))
    with connect() as conn:
        return [_holding(r) for r in conn.execute(query).all()]


def set_holding(subject: str, deal_id: uuid.UUID, since: Optional[date], tolerance: Optional[float],
                *, now: Optional[datetime] = None) -> dict:
    """Mark the deal as owned, or change its holding. ``PortfolioFull`` past ``MAX_OWNED`` of its owner's."""
    holding = clean_holding(since, tolerance)
    with transaction() as conn:
        held = _lock_reachable(conn, subject, deal_id, EDIT, Deal.owner_id, Deal.portfolio)
        if held.portfolio == holding:
            return holding
        if held.portfolio is None:
            owned = conn.execute(select(func.count()).select_from(Deal).where(
                Deal.owner_id == held.owner_id, Deal.portfolio.is_not(None))).scalar_one()
            if owned >= MAX_OWNED:
                raise PortfolioFull(str(MAX_OWNED))
        conn.execute(update(Deal).where(Deal.id == deal_id).values(portfolio=holding))
        audit.record(conn, audit.subject_id(subject),
                     "portfolio_added" if held.portfolio is None else "portfolio_changed", deal_id=deal_id, now=now)
    return holding


def remove_holding(subject: str, deal_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Take the deal out of the portfolio. Its quarters stay, ready if it is marked again."""
    with transaction() as conn:
        held = _lock_reachable(conn, subject, deal_id, EDIT, Deal.portfolio)
        if held.portfolio is None:
            return
        conn.execute(update(Deal).where(Deal.id == deal_id).values(portfolio=null()))  # SQL NULL, not JSON null
        audit.record(conn, audit.subject_id(subject), "portfolio_removed", deal_id=deal_id, now=now)


def save_quarters(subject: str, deal_id: uuid.UUID, quarters: Optional[Mapping],
                  *, now: Optional[datetime] = None) -> tuple[Optional[dict], Optional[datetime]]:
    """Replace the deal's quarters (``None`` clears them). They must be in the deal's currency
    (``QuartersRefused("other_currency")``); another unit is converted when they are used."""
    cleaned = None if quarters is None else clean_quarters(quarters)
    now = now or utc_now()
    with transaction() as conn:
        held = _lock_reachable(conn, subject, deal_id, EDIT, Deal.inputs, Deal.quarters, Deal.quarters_updated_at)
        if cleaned is not None and cleaned["currency"] != held.inputs.get("currency", DEFAULT_CURRENCY):
            raise QuartersRefused("other_currency")
        if held.quarters == cleaned:
            return held.quarters, held.quarters_updated_at
        stamp = None if cleaned is None else now
        conn.execute(update(Deal).where(Deal.id == deal_id).values(
            quarters=null() if cleaned is None else cleaned, quarters_updated_at=stamp))
        audit.record(conn, audit.subject_id(subject), "quarters_cleared" if cleaned is None else "quarters_saved",
                     deal_id=deal_id, now=now)
    return cleaned, stamp


def save_reforecast(subject: str, deal_id: uuid.UUID, result: Mapping[str, Any],
                    *, now: Optional[datetime] = None) -> None:
    """Keep the latest reforecast's answer on the deal (the dashboard shows it)."""
    with transaction() as conn:
        _lock_reachable(conn, subject, deal_id, EDIT, Deal.id)
        conn.execute(update(Deal).where(Deal.id == deal_id).values(reforecast=dict(result)))
        audit.record(conn, audit.subject_id(subject), "reforecast_run", deal_id=deal_id, now=now)


__all__ = ["DealNotFound", "EDIT", "READ", "Holding", "MAX_LISTED", "MAX_OWNED", "NotPermitted", "PortfolioFull", "clean_holding",
           "clean_quarters", "get_holding", "holdings", "remove_holding", "save_quarters", "save_reforecast",
           "set_holding"]
