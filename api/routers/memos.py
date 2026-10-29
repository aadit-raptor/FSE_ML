"""Investment memos (PLAN.md 6.3): a saved deal version's memo, every figure from its model run.

    GET    /api/deals/{id}/memos                          the deal's memos
    POST   /api/deals/{id}/memos                          draft a version's memo: 202 and the job (``memo.draft``)
    GET    /api/deals/{id}/memos/jobs/{job_id}            the job: stage, progress, and the version once done
    GET    /api/deals/{id}/memos/{version}                the memo, drawn from its run and checked
    DELETE /api/deals/{id}/memos/{version}                delete it
    POST   /api/deals/{id}/memos/{version}/sentences      the AI writes the linking sentences (or the templates)
    GET    /api/deals/{id}/memos/{version}/word           the memo as a Word file
    GET    /api/deals/{id}/memos/{version}/pdf            the memo as a PDF

**Drafting runs the model; drawing never does.** The job runs the version
through the screens' own endpoints and keeps the run (ml/memo.py); every
figure of a drawn memo is then checked against that run, and the memo says
when the engine has moved on since (``current``). The sentences endpoint is
the one call that reaches the AI: it sends no deal, no name and no figure
(ml/memo_writer.py), and spends the same daily allowance as results notes
(``USER_WRITES``, ``WRITES_ALL``). The memo's words are in the account's
language. Refusals are 422 ``{"code", "message"}``; another account's deal is
a 404, as a missing one is.
"""
from __future__ import annotations

import re
import uuid
from contextlib import contextmanager

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.auth import AuthUser, require_user
from api.routers.jobs import job_out, job_queue, submit_kind
from api.schemas import MemoJobOut, MemoList, MemoOut, MemoStart
from core.model_version import ENGINE_VERSION
from db import deals, memos, users
from jobs import config
from jobs.queue import ACTIVE, JobNotFound, JobQueue
from ml import memo, memo_writer
from research import report, report_files

router = APIRouter(prefix="/deals", tags=["memos"])

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
NO_DEAL = "no_deal"
NO_VERSION = "no_version"
NO_MEMO = "no_memo"
FULL = "memos_full"


@contextmanager
def memo_errors():
    try:
        yield
    except deals.DealNotFound:
        raise HTTPException(404, NO_DEAL) from None
    except deals.VersionNotFound:
        raise HTTPException(404, NO_VERSION) from None
    except memos.MemoNotFound:
        raise HTTPException(404, NO_MEMO) from None
    except memos.MemosFull:
        raise HTTPException(409, FULL) from None
    except report.DraftRefused as exc:
        raise HTTPException(422, {"code": exc.code, "message": exc.where}) from None


def _reader(subject: str) -> tuple[str, str, str]:
    """The account's language, digit grouping and locale tag for the memo."""
    profile = users.get_profile(subject)
    if profile is None:
        return memo.DEFAULT_LANGUAGE, "thousands", "en-GB"
    return (memo.language_for(profile.locale), report.grouping_for(profile.digit_grouping, profile.locale),
            profile.locale)


def _sources(subject: str, deal_id: uuid.UUID, version: int) -> tuple[memo.Sources, memos.MemoRecord, str, list]:
    """Everything a memo is drawn from, the stored memo, its locale, and who wrote each linking sentence."""
    record = memos.get_memo(subject, deal_id, version)
    deal = deals.get_deal(subject, deal_id)
    try:
        saved = deals.get_version(subject, deal_id, version)
        label, saved_on = saved.label or "", saved.created_at.date().isoformat()
    except deals.VersionNotFound:       # automatic versions are pruned; the memo keeps its run
        label, saved_on = "", ""
    language, grouping, locale = _reader(subject)
    s = memo.Sources(run=record.run, meta=memo.Meta(deal.name, version, label, saved_on),
                     language=language, grouping=grouping)
    written = memo_writer.stored(record.sentences, s)
    s = memo.with_sentences(s, {name: text for name, (text, _) in written.items()})
    return s, record, locale, [{"section": name, "source": source} for name, (_, source) in written.items()]


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _draw(subject: str, deal_id: uuid.UUID, version: int) -> dict:
    s, record, _, sentences = _sources(subject, deal_id, version)
    draft = memo.build(s)
    return {"deal_id": str(deal_id), "deal_name": s.meta.deal_name, "version": version,
            "drafted_at": _iso(record.updated_at), "engine_version": record.engine_version,
            "current": record.engine_version == ENGINE_VERSION, "language": s.language,
            "sentences": sentences, "written_by": (record.sentences or {}).get("model"),
            "blocks": report.to_dict(draft)["blocks"]}


# ---------------------------------------------------------------------------------------------- the deal's memos
@router.get("/{deal_id}/memos", response_model=MemoList)
def get_memos(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with memo_errors():
        found = memos.list_memos(user.subject, deal_id)
    return {"max_memos": memos.MAX_MEMOS,
            "memos": [{"version": m.version, "drafted_at": _iso(m.updated_at), "engine_version": m.engine_version,
                       "current": m.engine_version == ENGINE_VERSION} for m in found]}


def _job_out(record) -> dict:
    out = job_out(record)
    return {k: out[k] for k in ("id", "status", "progress", "stage", "error", "error_status", "result_expired",
                                "result")}


@router.post("/{deal_id}/memos", response_model=MemoJobOut, status_code=202)
def post_memo(deal_id: uuid.UUID, body: MemoStart, request: Request, response: Response,
              user: AuthUser = Depends(require_user)):
    """Draft (or draft again) a version's memo, as a background job: the version runs through the model."""
    with memo_errors():
        deals.get_version(user.subject, deal_id, body.version)        # the caller's, and there
        if not memos.has_room(user.subject, deal_id, body.version):
            raise memos.MemosFull(str(memos.MAX_MEMOS))
    record = submit_kind(user.subject, memo.KIND, {"owner": user.subject, "deal_id": str(deal_id),
                                                   "version": body.version})
    response.headers["Location"] = f"{request.url.path}/jobs/{record.id}"
    return _job_out(record)


@router.get("/{deal_id}/memos/jobs/{job_id}", response_model=MemoJobOut)
def get_memo_job(deal_id: uuid.UUID, job_id: uuid.UUID, user: AuthUser = Depends(require_user),
                 queue: JobQueue = Depends(job_queue)):
    """A draft's progress. Another account's job, or another kind, is a 404 (a job is the caller's whatever the
    deal in the address: the queue keeps no payload once it has run)."""
    try:
        record = queue.get(user.subject, job_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "no_job") from None
    if record.kind != memo.KIND:
        raise HTTPException(404, "no_job")
    if record.status in ACTIVE:
        config.wake_runner()
    return _job_out(record)


# ---------------------------------------------------------------------------------------------- one memo
@router.get("/{deal_id}/memos/{version}", response_model=MemoOut)
def get_memo(deal_id: uuid.UUID, version: int, user: AuthUser = Depends(require_user)):
    with memo_errors():
        return _draw(user.subject, deal_id, version)


@router.delete("/{deal_id}/memos/{version}", status_code=204, response_class=Response)
def delete_memo(deal_id: uuid.UUID, version: int, user: AuthUser = Depends(require_user)):
    with memo_errors():
        memos.delete_memo(user.subject, deal_id, version)
    return Response(status_code=204)


@router.post("/{deal_id}/memos/{version}/sentences", response_model=MemoOut)
def post_sentences(deal_id: uuid.UUID, version: int, user: AuthUser = Depends(require_user)):
    """The linking sentences written again: by the AI when one is set up and the reader reads English (its answer
    checked, a section that breaks a rule written by its template), else by the templates. Kept with the memo."""
    from api.limits import USER_WRITES, WRITES_ALL, enforce
    from research.reader.providers import get_provider

    with memo_errors():
        s, _, _, _ = _sources(user.subject, deal_id, version)
        provider = get_provider() if s.language == "en" else None
        if provider is not None:
            enforce([(USER_WRITES, user.subject), (WRITES_ALL, "all")])
        spent: list = []
        written = memo_writer.write(s, provider, spent)
        if spent:
            from research.reader.service import count_usage

            count_usage(user.subject, "memo", provider, answers=spent)
        memos.save_sentences(user.subject, deal_id, version, written)
        return _draw(user.subject, deal_id, version)


def _filename(name: str, version: int, extension: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "deal").strip("_")[:80] or "deal"
    return "%s_v%d_memo.%s" % (base, version, extension)


def _lang(locale: str) -> str:
    return locale if re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*", locale or "") else "en-GB"


def _file(deal_id: uuid.UUID, version: int, user: AuthUser, kind: str) -> Response:
    with memo_errors():
        s, _, locale, _ = _sources(user.subject, deal_id, version)
        draft = memo.build(s)
        deals.record_export(user.subject, deal_id, kind)
    title = s.words["doc"]["file_title"].replace("{name}", s.meta.deal_name)
    if kind == "memo_word":
        data, media, extension = report_files.word(draft, title, lang=_lang(locale)), DOCX, "docx"
    else:
        data, media, extension = report_files.pdf(draft, title), "application/pdf", "pdf"
    return Response(data, media_type=media, headers={
        "Content-Disposition": 'attachment; filename="%s"' % _filename(s.meta.deal_name, version, extension)})


@router.get("/{deal_id}/memos/{version}/word", response_class=Response, responses={
    200: {"content": {DOCX: {}}, "description": "The memo as a Word file, its chart a native Word chart"}})
def get_word(deal_id: uuid.UUID, version: int, user: AuthUser = Depends(require_user)):
    return _file(deal_id, version, user, "memo_word")


@router.get("/{deal_id}/memos/{version}/pdf", response_class=Response, responses={
    200: {"content": {"application/pdf": {}}, "description": "The memo as a PDF"}})
def get_pdf(deal_id: uuid.UUID, version: int, user: AuthUser = Depends(require_user)):
    return _file(deal_id, version, user, "memo_pdf")
