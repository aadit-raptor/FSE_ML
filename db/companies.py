"""Stored company figures and the EDINET report index (PLAN.md 4.1).

Company data is public filings, shared by every account: nobody owns a
company row, and nothing here records who looked one up.

**Staying inside the free 0.5 GB.** What is stored is the summary
(companies/items.py), never a filing: a company is one row of names and
identifiers plus a row per fiscal year (``MAX_YEARS`` at most), under 4 KB
a company with five years (``tests/test_company_data.py`` measures it). At most
``MAX_COMPANIES`` are kept -- the scheduled refresh removes the least
recently used beyond that -- and EDINET's index keeps ``KEEP_EDINET_YEARS``
of reports. Together that is under ``BUDGET_BYTES``, which
``/api/health/database`` reports against (``usage``).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import insert

from companies.edinet import IndexedReport
from companies.items import SUMMARY_FIELDS
from companies.model import CompanyData, CompanyRef, FilingLink, Warning, YearFigures
from db.engine import connect, transaction
from db.models import Company, CompanyYear, EdinetReport, SourceCursor, utc_now

MAX_COMPANIES = 8_000
MAX_YEARS = 5
KEEP_EDINET_YEARS = 3
# 64 MB of the free 512 MB: companies, their years and the EDINET index
BUDGET_BYTES = 64 * 1024 * 1024
BUDGET_WARN_FRACTION = 0.8
# A read marks a company used at most this often, so reading isn't writing
TOUCH_EVERY = timedelta(hours=6)
TABLES = ("companies", "company_years", "edinet_reports", "source_cursors")


def save(data: CompanyData) -> datetime:
    """Store a company's figures, replacing what was kept; when it was saved."""
    ref, now = data.ref, utc_now()
    values = dict(
        name=ref.name[:200], local_name=(ref.local_name or None) and ref.local_name[:200],
        country=ref.country, lei=ref.identifiers.get("lei"), identifiers=dict(ref.identifiers),
        currency=data.currency, unit=data.unit, accounting_standard=data.accounting_standard,
        fiscal_year_end_month=data.fiscal_year_end_month,
        warnings=[{"code": w.code, **({"field": w.field} if w.field else {})} for w in data.warnings],
        refreshed_at=now, last_used_at=now)
    stmt = insert(Company).values(source=ref.source, source_id=ref.source_id, **values)
    stmt = stmt.on_conflict_do_update(constraint="uq_companies_source_source_id", set_=values)
    with transaction() as conn:
        company_id = conn.execute(stmt.returning(Company.id)).scalar_one()
        conn.execute(delete(CompanyYear).where(CompanyYear.company_id == company_id))
        years = data.years[-MAX_YEARS:]
        if years:
            conn.execute(insert(CompanyYear), [dict(
                company_id=company_id, fiscal_year=y.fiscal_year, period_end=y.period_end,
                figures={k: v for k, v in y.figures.items() if v is not None},
                filing_url=y.filing.url[:400], filing_form=y.filing.form[:30], filing_id=y.filing.id[:80],
                filed_on=y.filing.filed_on) for y in years])
    return now


def get(source: str, source_id: str, *, touch: bool = True) -> Optional[tuple[CompanyData, datetime]]:
    """A stored company and when it was refreshed, or None."""
    with connect() as conn:
        row = conn.execute(select(Company).where(Company.source == source,
                                                 Company.source_id == source_id)).one_or_none()
        if row is None:
            return None
        years = conn.execute(select(CompanyYear).where(CompanyYear.company_id == row.id)
                             .order_by(CompanyYear.fiscal_year)).all()
    if touch and row.last_used_at < utc_now() - TOUCH_EVERY:
        with transaction() as conn:
            conn.execute(update(Company).where(Company.id == row.id).values(last_used_at=utc_now()))
    data = CompanyData(
        ref=CompanyRef(row.source, row.source_id, row.name, row.country, dict(row.identifiers), row.local_name),
        currency=row.currency, accounting_standard=row.accounting_standard,
        fiscal_year_end_month=row.fiscal_year_end_month,
        years=[YearFigures(y.fiscal_year, y.period_end, {k: y.figures.get(k) for k in SUMMARY_FIELDS},
                           FilingLink(y.filing_url, y.filed_on, y.filing_form, y.filing_id)) for y in years],
        warnings=[Warning(w.get("code", ""), w.get("field", "")) for w in row.warnings], unit=row.unit)
    return data, row.refreshed_at


def stored(keys: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Which of these (source, source_id) pairs are stored."""
    if not keys:
        return set()
    with connect() as conn:
        rows = conn.execute(select(Company.source, Company.source_id).where(
            tuple_(Company.source, Company.source_id).in_(keys))).all()
    return {(r.source, r.source_id) for r in rows}


def stale(limit: int, refreshed_before: datetime) -> list[tuple[str, str, int]]:
    """Companies due a refresh, the longest waiting first, with how many
    years each keeps (a refresh reads as many again)."""
    kept = (select(func.count()).where(CompanyYear.company_id == Company.id)
            .correlate(Company).scalar_subquery())
    with connect() as conn:
        rows = conn.execute(select(Company.source, Company.source_id, kept.label("years"))
                            .where(Company.refreshed_at < refreshed_before)
                            .order_by(Company.refreshed_at).limit(limit)).all()
    return [(r.source, r.source_id, r.years) for r in rows]


def mark_refreshed(source: str, source_id: str) -> None:
    """A refresh that found nothing new (or failed): try again next cycle."""
    with transaction() as conn:
        conn.execute(update(Company).where(Company.source == source, Company.source_id == source_id)
                     .values(refreshed_at=utc_now()))


def evict(max_companies: int = MAX_COMPANIES) -> int:
    """Delete the least recently used companies beyond ``max_companies``."""
    keep = select(Company.id).order_by(Company.last_used_at.desc(), Company.id.desc()).limit(max_companies)
    with transaction() as conn:
        return conn.execute(delete(Company).where(Company.id.not_in(keep))).rowcount


def usage_on(conn) -> dict:
    """Bytes the company tables take (data, indexes and TOAST) against the
    budget, on an open connection (the database health check's)."""
    sql = text("SELECT COALESCE(SUM(pg_total_relation_size(to_regclass(t))), 0) FROM unnest(CAST(:tables AS text[])) AS t")
    used = int(conn.execute(sql, {"tables": list(TABLES)}).scalar_one())
    companies = conn.execute(select(func.count()).select_from(Company)).scalar_one()
    return {"bytes": used, "budget_bytes": BUDGET_BYTES, "companies": companies,
            "max_companies": MAX_COMPANIES, "warning": used >= BUDGET_WARN_FRACTION * BUDGET_BYTES}


def usage() -> dict:
    with connect() as conn:
        return usage_on(conn)


class DatabaseReportIndex:
    """companies.edinet.ReportIndex in the database (``edinet_reports``)."""

    source = "edinet"

    def reports(self, edinet_code: str) -> list[IndexedReport]:
        with connect() as conn:
            rows = conn.execute(select(EdinetReport).where(EdinetReport.edinet_code == edinet_code)
                                .order_by(EdinetReport.period_end.desc())).all()
        return [IndexedReport(r.edinet_code, r.doc_id, r.period_start, r.period_end, r.submitted_at) for r in rows]

    def add(self, reports: list[IndexedReport]) -> int:
        """Keep each company's newest report per period (an amendment
        replaces the original); how many rows were written."""
        newest: dict[tuple[str, date], IndexedReport] = {}
        for r in reports:
            held = newest.get((r.edinet_code, r.period_end))
            if held is None or r.submitted_at > held.submitted_at:
                newest[(r.edinet_code, r.period_end)] = r
        if not newest:
            return 0
        stmt = insert(EdinetReport).values([dict(
            edinet_code=r.edinet_code, period_end=r.period_end, doc_id=r.doc_id, period_start=r.period_start,
            submitted_at=_aware(r.submitted_at)) for r in newest.values()])
        stmt = stmt.on_conflict_do_update(
            index_elements=["edinet_code", "period_end"],
            set_={"doc_id": stmt.excluded.doc_id, "period_start": stmt.excluded.period_start,
                  "submitted_at": stmt.excluded.submitted_at},
            where=EdinetReport.submitted_at < stmt.excluded.submitted_at)
        with transaction() as conn:
            return len(conn.execute(stmt.returning(EdinetReport.doc_id)).all())

    def scanned_through(self) -> Optional[date]:
        with connect() as conn:
            return conn.execute(select(SourceCursor.through).where(SourceCursor.source == self.source)).scalar()

    def set_scanned_through(self, day: date) -> None:
        stmt = insert(SourceCursor).values(source=self.source, through=day, updated_at=utc_now())
        stmt = stmt.on_conflict_do_update(index_elements=["source"],
                                          set_={"through": day, "updated_at": utc_now()})
        with transaction() as conn:
            conn.execute(stmt)

    def prune(self, today: date) -> int:
        from companies.edinet import shift_years
        cutoff = shift_years(today, KEEP_EDINET_YEARS)      # safe on 29 February
        with transaction() as conn:
            return conn.execute(delete(EdinetReport).where(EdinetReport.period_end < cutoff)).rowcount


def _aware(moment: datetime) -> datetime:
    """EDINET's submission times are Japan time without a zone."""
    if moment.tzinfo is not None:
        return moment
    from zoneinfo import ZoneInfo
    return moment.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
