"""Covered companies and their accepted periods (PLAN.md R2).

**Whose coverage.** As with deals and files, every function takes the
caller's subject, from the verified token, and matches it in the same
statement that reads or writes a row. Someone else's company or period is
indistinguishable from one that doesn't exist: both raise ``NotFound``.

**Room.** A person keeps at most ``MAX_COMPANIES`` companies and
``MAX_PERIODS`` periods, and everyone's coverage at most ``BUDGET_BYTES``
of the free database (reported by ``/api/health/database`` as
``coverage_data``). A stored period is about 20 KB (a filing's three
statements, every figure with its page), so the budget holds a few
thousand. Each limit is checked in the transaction that writes, with the
caller's account row locked, so two requests can't both take the last room.

**Audit history.** Adding or removing a company, and accepting or removing
periods, each write one entry in the same transaction, with no name and no
figure.

**Never log a company's name or a figure**, as with a deal's.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from sqlalchemy import delete, func, insert, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db import audit
from db.engine import connect, transaction
from db.models import CoveredCompany, CoveredPeriod, StoredFile, User, utc_now

MAX_COMPANIES = 100
MAX_PERIODS = 400
BUDGET_BYTES = 64 * 1024 * 1024
BUDGET_WARN_FRACTION = 0.8
TABLES = ("coverage_companies", "coverage_periods", "research_models", "research_model_versions")  # R5's saved models too

# Refusals, as codes the screen words (web/messages/en.json coverage.error_*)
TOO_MANY_COMPANIES = "too_many_companies"
TOO_MANY_PERIODS = "too_many_periods"
COVERAGE_FULL = "coverage_full"


class NoAccount(LookupError):
    """The caller hasn't finished their account, so there is no one to own coverage."""


class NotFound(LookupError):
    """No such company or period for this caller (it may exist and belong to someone else)."""


class Full(ValueError):
    """A limit is reached; ``code`` says which."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CompanyRecord:
    id: uuid.UUID
    name: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PeriodRecord:
    id: uuid.UUID
    company_id: uuid.UUID
    company_name: str
    period_end: date
    months: int
    scope: str
    unit: str
    currency: Optional[str]
    file_id: Optional[uuid.UUID]
    accepted_at: datetime
    figures: Optional[dict] = None


_COMPANY = (CoveredCompany.id, CoveredCompany.name, CoveredCompany.created_at, CoveredCompany.updated_at)
_PERIOD = (CoveredPeriod.id, CoveredPeriod.company_id, CoveredCompany.name.label("company_name"),
           CoveredPeriod.period_end, CoveredPeriod.months, CoveredPeriod.scope, CoveredPeriod.unit,
           CoveredPeriod.currency, CoveredPeriod.file_id, CoveredPeriod.accepted_at)


def _company(row) -> CompanyRecord:
    return CompanyRecord(id=row.id, name=row.name, created_at=row.created_at, updated_at=row.updated_at)


def _period(row, figures: Optional[dict] = None) -> PeriodRecord:
    return PeriodRecord(id=row.id, company_id=row.company_id, company_name=row.company_name,
                        period_end=row.period_end, months=row.months, scope=row.scope, unit=row.unit,
                        currency=row.currency, file_id=row.file_id, accepted_at=row.accepted_at, figures=figures)


def _owner_id(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def _locked_owner(conn, subject: str) -> int:
    owner = conn.execute(select(User.id).where(User.subject == subject).with_for_update()).scalar()
    if owner is None:
        raise NoAccount(subject)
    return owner


def _used(conn) -> int:
    sql = text("SELECT COALESCE(SUM(pg_total_relation_size(to_regclass(t))), 0) "
               "FROM unnest(CAST(:tables AS text[])) AS t")
    return int(conn.execute(sql, {"tables": list(TABLES)}).scalar_one())


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------
def list_companies(subject: str) -> list[tuple[CompanyRecord, list[PeriodRecord]]]:
    """The caller's companies, newest first, each with its periods (newest first, figures left out)."""
    with connect() as conn:
        companies = conn.execute(select(*_COMPANY).where(CoveredCompany.owner_id == _owner_id(subject))
                                 .order_by(CoveredCompany.created_at.desc(), CoveredCompany.id)).all()
        periods = conn.execute(select(*_PERIOD).join(CoveredCompany, CoveredCompany.id == CoveredPeriod.company_id)
                               .where(CoveredPeriod.owner_id == _owner_id(subject))
                               .order_by(CoveredPeriod.period_end.desc(), CoveredPeriod.months,
                                         CoveredPeriod.scope)).all()
    by_company: dict[uuid.UUID, list[PeriodRecord]] = {}
    for row in periods:
        by_company.setdefault(row.company_id, []).append(_period(row))
    return [(_company(c), by_company.get(c.id, [])) for c in companies]


def create_company(subject: str, name: str, *, now: Optional[datetime] = None) -> CompanyRecord:
    now = now or utc_now()
    with transaction() as conn:
        owner = _locked_owner(conn, subject)
        held = conn.execute(select(func.count()).where(CoveredCompany.owner_id == owner)).scalar_one()
        if held >= MAX_COMPANIES:
            raise Full(TOO_MANY_COMPANIES)
        row = conn.execute(insert(CoveredCompany).values(
            id=uuid.uuid4(), owner_id=owner, name=name, created_at=now, updated_at=now,
        ).returning(*_COMPANY)).one()
        audit.record(conn, owner, "coverage_added", now=now)
    return _company(row)


def get_company(subject: str, company_id: uuid.UUID) -> CompanyRecord:
    with connect() as conn:
        row = conn.execute(select(*_COMPANY).where(
            CoveredCompany.id == company_id, CoveredCompany.owner_id == _owner_id(subject))).first()
    if row is None:
        raise NotFound(str(company_id))
    return _company(row)


def delete_company(subject: str, company_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Remove the company and every period it holds."""
    now = now or utc_now()
    with transaction() as conn:
        row = conn.execute(delete(CoveredCompany).where(
            CoveredCompany.id == company_id, CoveredCompany.owner_id == _owner_id(subject),
        ).returning(CoveredCompany.owner_id)).first()
        if row is None:
            raise NotFound(str(company_id))
        audit.record(conn, row.owner_id, "coverage_removed", now=now)


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
def save_periods(subject: str, company_id: uuid.UUID, periods: list[dict], file_id: Optional[uuid.UUID], *,
                 now: Optional[datetime] = None, budget: Optional[int] = None) -> list[PeriodRecord]:
    """Store accepted periods (``research.coverage.accept``'s), each replacing
    the one kept for the same period and scope. One audit entry for the lot."""
    now = now or utc_now()
    with transaction() as conn:
        owner = _locked_owner(conn, subject)
        company = conn.execute(select(CoveredCompany.id).where(
            CoveredCompany.id == company_id, CoveredCompany.owner_id == owner)).first()
        if company is None:
            raise NotFound(str(company_id))
        # the source file only while it is the caller's (a read's file may have gone since)
        if file_id is not None and conn.execute(select(StoredFile.id).where(
                StoredFile.id == file_id, StoredFile.owner_id == owner)).first() is None:
            file_id = None
        held = conn.execute(select(func.count()).where(CoveredPeriod.owner_id == owner)).scalar_one()
        replacing = conn.execute(select(func.count()).where(
            CoveredPeriod.company_id == company_id,
            tuple_(CoveredPeriod.period_end, CoveredPeriod.months, CoveredPeriod.scope).in_(
                [(date.fromisoformat(str(p["period_end"])), p["months"], p["scope"]) for p in periods]),
        )).scalar_one()
        if held - replacing + len(periods) > MAX_PERIODS:
            raise Full(TOO_MANY_PERIODS)
        if _used(conn) >= (BUDGET_BYTES if budget is None else budget):
            raise Full(COVERAGE_FULL)
        ids = []
        for p in periods:
            values = dict(company_id=company_id, owner_id=owner, period_end=date.fromisoformat(str(p["period_end"])),
                          months=p["months"], scope=p["scope"], unit=p["unit"], currency=p.get("currency"),
                          file_id=file_id, figures=p["figures"], accepted_at=now)
            stmt = pg_insert(CoveredPeriod).values(id=uuid.uuid4(), **values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_coverage_periods_company_period_scope",
                set_={k: stmt.excluded[k] for k in ("unit", "currency", "file_id", "figures", "accepted_at")})
            ids.append(conn.execute(stmt.returning(CoveredPeriod.id)).scalar_one())
        conn.execute(update(CoveredCompany).where(CoveredCompany.id == company_id).values(updated_at=now))
        audit.record(conn, owner, "period_accepted", now=now)
        rows = conn.execute(select(*_PERIOD).join(CoveredCompany, CoveredCompany.id == CoveredPeriod.company_id)
                            .where(CoveredPeriod.id.in_(ids))).all()
    order = {i: n for n, i in enumerate(ids)}
    return sorted((_period(r) for r in rows), key=lambda p: order[p.id])


def get_period(subject: str, period_id: uuid.UUID) -> PeriodRecord:
    """One of the caller's periods with its figures."""
    with connect() as conn:
        row = conn.execute(select(*_PERIOD, CoveredPeriod.figures)
                           .join(CoveredCompany, CoveredCompany.id == CoveredPeriod.company_id)
                           .where(CoveredPeriod.id == period_id,
                                  CoveredPeriod.owner_id == _owner_id(subject))).first()
    if row is None:
        raise NotFound(str(period_id))
    return _period(row, row.figures)


def company_figures(subject: str, company_id: uuid.UUID) -> list[dict]:
    """Every period kept for one of the caller's companies, with its figures:
    what a new read of that company is compared with."""
    with connect() as conn:
        rows = conn.execute(select(CoveredPeriod.scope, CoveredPeriod.unit, CoveredPeriod.currency,
                                   CoveredPeriod.figures).where(
            CoveredPeriod.company_id == company_id, CoveredPeriod.owner_id == _owner_id(subject))).all()
    return [{"scope": r.scope, "unit": r.unit, "currency": r.currency, "figures": r.figures} for r in rows]


def delete_period(subject: str, period_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    now = now or utc_now()
    with transaction() as conn:
        row = conn.execute(delete(CoveredPeriod).where(
            CoveredPeriod.id == period_id, CoveredPeriod.owner_id == _owner_id(subject),
        ).returning(CoveredPeriod.owner_id)).first()
        if row is None:
            raise NotFound(str(period_id))
        audit.record(conn, row.owner_id, "period_removed", now=now)


def record_export(subject: str, *, kind: str = "results_workbook", now: Optional[datetime] = None) -> None:
    """A checked workbook downloaded (``results_workbook``) or written into Excel by the add-in
    (``results_excel``): one entry in the caller's history (no name, no figure)."""
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject)).scalar()
        if owner is None:
            raise NoAccount(subject)
        audit.record(conn, owner, "exported", now=now or utc_now(), detail={"export": kind})


def usage_on(conn) -> dict:
    """Bytes coverage takes against its budget, on an open connection (the
    database health check's)."""
    used = _used(conn)
    periods = conn.execute(select(func.count()).select_from(CoveredPeriod)).scalar_one()
    return {"bytes": used, "budget_bytes": BUDGET_BYTES, "periods": int(periods),
            "fraction": round(used / BUDGET_BYTES, 4), "warning": used >= BUDGET_BYTES * BUDGET_WARN_FRACTION}


__all__ = ["BUDGET_BYTES", "CompanyRecord", "Full", "MAX_COMPANIES", "MAX_PERIODS", "NoAccount", "NotFound",
           "PeriodRecord", "company_figures", "create_company", "delete_company", "delete_period",
           "get_company", "get_period", "list_companies", "record_export", "save_periods", "usage_on"]
