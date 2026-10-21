"""Results notes (PLAN.md R8): a covered company's note drafted from its saved sector model.

    GET  /api/account/report-details                         the firm name and disclosures every note carries
    PUT  /api/account/report-details                         change them
    GET  /api/research/companies/{id}/report                 the note: what the analyst typed and the draft
    PUT  /api/research/companies/{id}/report                 keep what the analyst typed; the draft again
    POST /api/research/companies/{id}/report/sentences       the AI writes the linking sentences (or the templates)
    GET  /api/research/companies/{id}/report/word            the note as a Word file
    GET  /api/research/companies/{id}/report/pdf             the note as a PDF

**Every figure is read from the saved model each time the note is drawn**
(research/report.py ``build``, which checks every one against its cell and
refuses a draft holding anything else); nothing here stores a figure. The
rating, the target and the argument are only ever what the analyst typed. The
sentences endpoint is the one call that reaches the AI: it sends no company
and no figure (research/writer.py), and is limited a day per person and for
the service (``USER_WRITES``, ``WRITES_ALL``) because it spends the reader's
free allowance. Refusals are 422 with ``{"code", "message"}``; another
account's company is a 404, as a missing one is.
"""
from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import AuthUser, require_user
from api.routers.coverage import store_errors
from api.routers.research_models import refusals
from api.routers.research_saved import _built, saved_errors
from api.schemas import ReportDetailsIn, ReportDetailsOut, ReportFieldsIn, ReportOut
from db import coverage as coverage_rows
from db import research_models as model_rows
from db import research_reports as rows
from db import users
from research import compose, report, report_files, writer
from research.definition import Company
from research.evaluate import evaluate
from research.models import CATALOGUE

router = APIRouter(tags=["research"])

NO_VERSION = "no_version"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@contextmanager
def report_errors():
    with saved_errors():
        try:
            yield
        except report.DraftRefused as exc:
            raise HTTPException(422, {"code": exc.code, "message": exc.where}) from None


def _grouping(subject: str) -> str:
    profile = users.get_profile(subject)
    return report.grouping_for(profile.digit_grouping, profile.locale) if profile else "thousands"


def _earlier(subject: str, company_id: uuid.UUID, number: Optional[int], *,
             strict: bool = True) -> Optional[report.Earlier]:
    """The version the estimates are set against. One that is gone (versions are pruned) refuses a save
    (`strict`); a note that chose it earlier is drawn without it, so it can be opened and another chosen."""
    if number is None:
        return None
    try:
        version = model_rows.get_version(subject, company_id, number)
    except model_rows.VersionNotFound:
        if not strict:
            return None
        raise HTTPException(422, {"code": NO_VERSION, "message": ""}) from None
    model = CATALOGUE.build(compose.read(version.structure, CATALOGUE.specs))
    raw = dict(version.company)
    raw["off"] = tuple(raw.get("off") or ())
    company, _ = compose.fit(model, Company(**raw))
    return report.Earlier(number, model, company, evaluate(model, company, partial=True))


def _comparison(subject: str, company_id: uuid.UUID) -> Optional[dict]:
    """The comparison the latest roll forward stored with its version, as it was kept."""
    return next((v.comparison for v in model_rows.list_versions(subject, company_id) if v.comparison), None)


def _fields(stored: dict) -> report.Fields:
    target = stored.get("target")
    return report.Fields(argument=stored.get("argument") or "", rating=stored.get("rating") or "",
                         target=None if target is None else float(target), note_date=stored.get("note_date") or "",
                         compare_version=stored.get("compare_version"))


def _sources(subject: str, company_id: uuid.UUID, stored: dict,
             saved: Optional[dict]) -> tuple[report.Sources, dict, list[dict]]:
    """Everything a draft reads, from storage; the model's record; and who wrote each linking sentence."""
    record = model_rows.get_model(subject, company_id)
    with refusals():
        model, company = _built(record)
    firm, disclosures = rows.get_details(subject)
    fields = _fields(stored)
    earlier = _earlier(subject, company_id, fields.compare_version, strict=False)
    if earlier is None and fields.compare_version is not None:
        fields = replace(fields, compare_version=None)
    s = report.Sources(model=model, company=company, result=evaluate(model, company, partial=True), fields=fields,
                       details=report.Details(firm, disclosures), comparison=_comparison(subject, company_id),
                       earlier=earlier, grouping=_grouping(subject))
    written = writer.stored(saved, s)
    s = replace(s, sentences={name: text for name, (text, _) in written.items()})
    sources = [{"section": name, "source": source} for name, (_, source) in written.items()]
    return s, {"company_name": record.company_name}, sources


def _out(company_id: uuid.UUID, s: report.Sources, sources: list, saved: Optional[dict], name: str) -> dict:
    draft = report.build(s)
    return {"company_id": str(company_id), "company_name": name,
            "fields": {"argument": s.fields.argument, "rating": s.fields.rating, "target": s.fields.target,
                       "note_date": s.fields.note_date or None, "compare_version": s.fields.compare_version},
            "details": {"firm": s.details.firm, "disclosures": s.details.disclosures},
            "sentences": sources,
            "written_by": (saved or {}).get("model"),
            "blocks": report.to_dict(draft)["blocks"]}


def _draw(subject: str, company_id: uuid.UUID) -> dict:
    """The note as the API answers it."""
    stored, saved = rows.get_report(subject, company_id)
    s, meta, sources = _sources(subject, company_id, stored, saved)
    return _out(company_id, s, sources, saved, meta["company_name"])


# ---------------------------------------------------------------------------------------------- the account's details
@router.get("/account/report-details", response_model=ReportDetailsOut)
def get_details(user: AuthUser = Depends(require_user)):
    with store_errors():
        firm, disclosures = rows.get_details(user.subject)
    return {"firm": firm, "disclosures": disclosures}


@router.put("/account/report-details", response_model=ReportDetailsOut)
def put_details(body: ReportDetailsIn, user: AuthUser = Depends(require_user)):
    with store_errors():
        rows.save_details(user.subject, body.firm, body.disclosures)
        firm, disclosures = rows.get_details(user.subject)
    return {"firm": firm, "disclosures": disclosures}


# ---------------------------------------------------------------------------------------------- the note
@router.get("/research/companies/{company_id}/report", response_model=ReportOut)
def get_report(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with report_errors():
        return _draw(user.subject, company_id)


@router.put("/research/companies/{company_id}/report", response_model=ReportOut)
def put_report(company_id: uuid.UUID, body: ReportFieldsIn, user: AuthUser = Depends(require_user)):
    """Keep what the analyst typed. A version to compare with must be one the saved model has."""
    fields = body.model_dump()
    fields["note_date"] = body.note_date.isoformat() if body.note_date else None
    fields["rating"] = " ".join(body.rating.split())
    with report_errors():
        model_rows.get_model(user.subject, company_id)              # no model, no note
        if body.compare_version is not None:
            _earlier(user.subject, company_id, body.compare_version)
        rows.save_report(user.subject, company_id, fields)
        return _draw(user.subject, company_id)


@router.post("/research/companies/{company_id}/report/sentences", response_model=ReportOut)
def post_sentences(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """The linking sentences written again: by the AI when one is set up (its answer checked, a section that breaks
    a rule written by its template), else by the templates. Kept with the note."""
    from api.limits import USER_WRITES, WRITES_ALL, enforce
    from research.reader.providers import get_provider

    with report_errors():
        stored, _ = rows.get_report(user.subject, company_id)
        s, _, _ = _sources(user.subject, company_id, stored, None)
        provider = get_provider()
        if provider is not None:
            enforce([(USER_WRITES, user.subject), (WRITES_ALL, "all")])
        spent: list = []
        written = writer.write(s, provider, spent)
        if spent:
            from research.reader.service import count_usage

            count_usage(user.subject, "write", provider, answers=spent)
        rows.save_report(user.subject, company_id, sentences=written)      # the fields as they stand now
        return _draw(user.subject, company_id)


def _filename(name: str, extension: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "company").strip("_")[:80] or "company"
    return "%s_results_note.%s" % (base, extension)


def _file(company_id: uuid.UUID, user: AuthUser, kind: str) -> Response:
    with report_errors():
        stored, saved = rows.get_report(user.subject, company_id)
        s, meta, _ = _sources(user.subject, company_id, stored, saved)
        draft, name = report.build(s), meta["company_name"]
        coverage_rows.record_export(user.subject, kind=kind)
    title = "%s, results note" % name
    if kind == "report_word":
        data, media, extension = report_files.word(draft, title), DOCX, "docx"
    else:
        data, media, extension = report_files.pdf(draft, title), "application/pdf", "pdf"
    return Response(data, media_type=media,
                    headers={"Content-Disposition": 'attachment; filename="%s"' % _filename(name, extension)})


@router.get("/research/companies/{company_id}/report/word", response_class=Response, responses={
    200: {"content": {DOCX: {}}, "description": "The note as a Word file, its charts native Word charts"}})
def get_word(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    return _file(company_id, user, "report_word")


@router.get("/research/companies/{company_id}/report/pdf", response_class=Response, responses={
    200: {"content": {"application/pdf": {}}, "description": "The note as a PDF"}})
def get_pdf(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    return _file(company_id, user, "report_pdf")
