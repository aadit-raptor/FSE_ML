"""The reader as the API runs it: who may read what, and the job (PLAN.md R1).

``start`` (``POST /api/research/reads``) checks, in order: the reader is
configured and open (``gate``), the person confirmed the file is public
when the provider is a free tier, the file is theirs, a PDF and **scanned
clean** (PLAN.md 6.1), and their day's reads and the server's are not used
up. Then it queues job ``research.read`` under the person's own name, so
only they can see it and it counts among their waiting runs.

``read_file`` is the job: the file is fetched from the store to a private
spool file, its pages counted (in a child process) and charged against the
person's **daily page limit**, then read twice and checked
(``read.read_filing``), against what is kept for the company when the read is
for a covered one. The answer is the job's result (kept six hours, like every
job's); nothing is stored until the analyst accepts it (PLAN.md R2,
``api/routers/coverage.py``).

**Nothing from a document reaches a log**: events carry counts, models and
timings only, never a label, a figure or the file's name.
"""
from __future__ import annotations

import datetime
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from api.limits import holding_simulation_slot
from api.observability import deploy_environment, log_event
from research.reader import gate, pages
from research.reader import providers as ai
from research.reader.read import read_filing

KIND = "research.read"

# Refusals, as codes the screen words (web/messages/en.json reader.error_*)
OFF = "reader_off"
NOT_MEASURED = "reader_not_measured"
CONFIRM_PUBLIC = "confirm_public"
NOT_PDF = "not_pdf"
NOT_CLEAN = "not_clean"
NO_FILE = "no_file"
PAGE_LIMIT = "page_limit"
TOO_LONG = "too_long"
STORE_DOWN = "store_unreachable"
NO_COMPANY = "no_company"


def _refuse(status: int, code: str) -> HTTPException:
    return HTTPException(status, code)


def provider():
    found = ai.get_provider()
    if found is None:
        raise _refuse(503, OFF)
    return found


def check_open() -> None:
    if not gate.open_to_people(deploy_environment()):
        raise _refuse(503, NOT_MEASURED)


def clean_pdf(subject: str, file_id: uuid.UUID):
    from db import files as rows

    try:
        record = rows.get_file(subject, file_id)
    except rows.FileNotFound:
        raise _refuse(404, NO_FILE) from None
    if record.kind != "pdf":
        raise _refuse(422, NOT_PDF)
    if record.status != "clean":
        raise _refuse(409, NOT_CLEAN)
    return record


def _kept(subject: str, company_id: Optional[uuid.UUID]) -> list[dict]:
    """The periods kept for one of the caller's covered companies (PLAN.md R2);
    a company that isn't theirs (or no longer exists) has none."""
    if company_id is None:
        return []
    from db import coverage

    return coverage.company_figures(subject, company_id)


def start(subject: str, file_id: uuid.UUID, *, public: bool, quarter_end: Optional[datetime.date],
          company_id: Optional[uuid.UUID] = None):
    """Queue a read for the caller. Answers the job record."""
    from api.limits import READS_ALL, USER_READS, enforce
    from api.routers.jobs import submit_kind

    reader = provider()
    check_open()
    if reader.free_tier and not public:
        raise _refuse(422, CONFIRM_PUBLIC)
    clean_pdf(subject, file_id)
    if company_id is not None:
        from db import coverage
        from db.engine import is_configured

        if not is_configured():
            raise _refuse(404, NO_COMPANY)
        try:
            coverage.get_company(subject, company_id)
        except coverage.NotFound:
            raise _refuse(404, NO_COMPANY) from None
    enforce([(USER_READS, subject), (READS_ALL, "all")])
    payload = {"owner": subject, "file_id": str(file_id),
               "quarter_end": quarter_end.isoformat() if quarter_end else None,
               "company_id": str(company_id) if company_id else None}
    return submit_kind(subject, KIND, payload)


def _charge_pages(subject: str, count: int, now: Optional[float] = None) -> None:
    from api.limits import USER_READ_PAGES, check_for
    from api.usage import counters

    now = time.time() if now is None else now
    check = check_for(USER_READ_PAGES, subject, now)
    if counters().usage(check.key) + count > check.limit:
        raise _refuse(429, PAGE_LIMIT)
    counters().add(check, count)


def refund_pages(rule, subject: str, count: int) -> None:
    """Give back pages charged for a read no model answered: nothing was read, so nothing is owed."""
    from api.limits import check_for
    from api.usage import counters

    if count:
        counters().add(check_for(rule, subject, time.time()), -count)


def pages_left(subject: str, now: Optional[float] = None) -> int:
    from api.limits import USER_READ_PAGES, check_for
    from api.usage import counters

    check = check_for(USER_READ_PAGES, subject, time.time() if now is None else now)
    return max(0, check.limit - counters().usage(check.key))


def tier_of(reader) -> str:
    return "free" if getattr(reader, "free_tier", True) else "billed"


def count_usage(subject: str, kind: str, reader, *, pages: int = 0, answers=()) -> None:
    """What the request used, kept per person and day (PLAN.md R9, ``db.ai_usage``). Never fails the read or
    the writing it counts: a database that can't be reached is logged by kind and the work goes on."""
    from db.engine import is_configured

    if not is_configured() or (not pages and not answers):
        return
    try:
        from db import ai_usage

        ai_usage.record(subject, kind, tier_of(reader), pages=pages, answers=answers)
    except Exception as exc:  # noqa: BLE001 -- counting is never worth a lost read
        log_event("ai_usage_not_recorded", logging.WARNING, kind=kind, error=type(exc).__name__)


def read_file(owner: str, file_id: uuid.UUID, quarter_end: Optional[datetime.date], *,
              today: Optional[datetime.date] = None, company_id: Optional[uuid.UUID] = None) -> dict:
    """The job: fetch, count and charge the pages, read twice, check (against
    the periods kept for ``company_id``, when the read is for one)."""
    from uploads import rules, service as uploads
    from uploads.store import NotStored, StorageError

    reader = provider()
    record = clean_pdf(owner, file_id)
    path = uploads.spool() / f"read-{uuid.uuid4().hex}.pdf"
    t0 = time.perf_counter()
    charged, spent = 0, []
    try:
        try:
            uploads.store().fetch(record.key, path, rules.MAX_FILE_BYTES)
        except NotStored:
            raise _refuse(404, NO_FILE) from None
        except StorageError:
            raise _refuse(503, STORE_DOWN) from None
        try:
            with holding_simulation_slot():
                info = pages.info(path)
        except pages.PagesError:
            raise _refuse(422, "unreadable_pdf") from None
        if info.pages > pages.MAX_PAGES:
            raise _refuse(422, TOO_LONG)
        _charge_pages(owner, info.pages)
        charged = info.pages
        try:
            result = read_filing(path, reader, today=today or datetime.date.today(), quarter=quarter_end,
                                 info=info, kept=_kept(owner, company_id), spent=spent)
        except ai.ProviderError as exc:
            log_event("reader_failed", logging.WARNING, provider=reader.name, code=exc.code,
                      http_status=exc.status)
            status = {ai.SPENT: 429, ai.BUSY: 503, ai.NOT_RECORDED: 404}.get(exc.code, 502)
            raise _refuse(status, exc.code) from None
        except pages.PagesError:
            raise _refuse(422, "unreadable_pdf") from None
    finally:
        uploads.discard(path)
        if charged and not spent and sys.exc_info()[0] is not None:
            from api.limits import USER_READ_PAGES

            refund_pages(USER_READ_PAGES, owner, charged)
            charged = 0
        count_usage(owner, "read", reader, pages=charged, answers=spent)
    summary = result["summary"]
    log_event("reader_read", provider=reader.name, pages=info.pages, tables=summary["tables"],
              figures=summary["figures"], amber=summary["amber"],
              model_a=(result["reads"]["A"] or {}).get("model"), model_b=(result["reads"]["B"] or {}).get("model"),
              duration_ms=round((time.perf_counter() - t0) * 1000))
    result["file_id"] = str(file_id)
    return result
