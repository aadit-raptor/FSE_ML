"""File uploads (PLAN.md 6.1): PDF, Excel and Word files, private to their owner.

    GET    /api/files                 the caller's files and what they hold against their limits
    POST   /api/files                 {"name", "size"} -> room reserved for the file
    PUT    /api/files/{id}/content    the file's bytes, as the request body
    GET    /api/files/{id}            where the file stands
    POST   /api/files/{id}/link       a signed download link (clean files only)
    GET    /api/files/{id}/content    a clean PDF's bytes, for the review screen's page view (PLAN.md R2)
    DELETE /api/files/{id}

The bytes are streamed to disk as they arrive and counted: they are never
held in memory, and an upload that passes the size it declared is cut off
and refused. A scan is queued; only a file it passes can be downloaded. The
owner is always the signed-in caller, matched in every statement
(``db/files.py``): another account's file answers 404, like one that doesn't
exist.

A refusal says why with a ``code`` (the web has the words for each) beside a
plain sentence. Nothing here logs a file's name, a link or a refusal's text.

``local`` holds the route the local directory store's download links point
at (``uploads/store.py``). It is not in the schema and answers 404 unless
that store is in use, which production refuses.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import contextmanager
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from api.auth import AuthUser, require_user
from api.schemas import FileBegin, FileLink, FileList, FileOut, FileRefusal
from db import files as rows
from db.engine import is_configured as database_configured
from uploads import rules, service
from uploads.store import BadToken, NotStored, StorageError, TooLarge, local_store

router = APIRouter(prefix="/files", tags=["files"])
local = APIRouter(prefix="/files/local", include_in_schema=False)

MESSAGES = {
    "bad_name": "Give the file a name of at most 120 characters.",
    "bad_type": "Only PDF, Excel (.xlsx, .xls) and Word (.docx, .doc) files can be uploaded.",
    "empty": "The file is empty.",
    "too_large": f"The file is too large: the limit is {rules.MAX_FILE_BYTES // rules.MB} MB.",
    "quota_bytes": f"Your uploads would pass {rules.USER_QUOTA_BYTES // rules.MB} MB. Delete a file first.",
    "quota_files": f"You already have {rules.USER_MAX_FILES} files. Delete one first.",
    "storage_full": "The file store is full for now. Try again later.",
    "not_set_up": "File uploads are not set up on this server.",
    "in_flight": f"Wait for your other uploads to finish: at most {rules.MAX_IN_FLIGHT} at a time.",
    "not_expected": "This file's contents have already arrived.",
    "busy": "The server is taking in other uploads. Try again in a moment.",
    "too_slow": "The upload took too long to arrive. Try again on a faster connection.",
    "not_ready": "The file can be downloaded once it has been checked.",
    "storage_unavailable": "The file store can't be reached right now. Try again in a minute.",
}
STATUS = {"too_large": 413, "quota_bytes": 409, "quota_files": 409, "storage_full": 507,
          "not_set_up": 503, "in_flight": 429, "not_expected": 409, "not_ready": 409, "busy": 503, "too_slow": 408,
          "storage_unavailable": 503}
REFUSALS = {code: {"model": FileRefusal} for code in (408, 409, 413, 422, 429, 503, 507)}
# Uploads being streamed to disk right now, across all callers
_arriving = threading.BoundedSemaphore(rules.MAX_ARRIVING)
NO_ACCOUNT = "Finish your account (country, currency, format and time zone) before uploading."


def refusal(code: str, headers: dict | None = None) -> HTTPException:
    return HTTPException(STATUS.get(code, 422), {"code": code, "message": MESSAGES[code]}, headers=headers)


@contextmanager
def errors():
    """Upload errors as HTTP answers."""
    if not database_configured() or not service.enabled():
        raise refusal("not_set_up")
    try:
        yield
    except rules.InvalidUpload as exc:
        raise refusal(exc.code) from None
    except rows.FileNotFound:
        raise HTTPException(404, "File not found") from None
    except rows.NoAccount:
        raise HTTPException(409, NO_ACCOUNT) from None
    except service.NotSetUp:
        raise refusal("not_set_up") from None
    except service.NotExpected:
        raise refusal("not_expected") from None
    except service.NotReady:
        raise refusal("not_ready") from None
    except (StorageError, NotStored):
        raise refusal("storage_unavailable", {"Retry-After": "30"}) from None


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def file_out(record: rows.FileRecord) -> dict:
    return {"id": str(record.id), "name": record.name, "kind": record.kind, "size": record.size,
            "status": record.status, "reason": record.reason, "created_at": _iso(record.created_at),
            "expires_at": _iso(record.expires_at)}


@router.get("", response_model=FileList)
def get_files(user: AuthUser = Depends(require_user)):
    """The caller's files, newest first, and their use of their limits."""
    kinds = list(rules.KINDS)
    if not database_configured() or not service.enabled():
        return {"enabled": False, "files": [], "usage": rules.usage(0, 0), "kinds": kinds}
    records, usage = rows.list_files(user.subject)
    service.wake_scans(user.subject, records)
    return {"enabled": True, "files": [file_out(r) for r in records], "usage": usage, "kinds": kinds}


@router.post("", response_model=FileOut, status_code=201, responses=REFUSALS)
def post_file(body: FileBegin, user: AuthUser = Depends(require_user)):
    """Reserve room for a file. Then send its bytes: ``PUT /api/files/{id}/content``."""
    with errors():
        return file_out(service.begin(user.subject, body.name, body.size))


async def _spool(request: Request, path, limit: int) -> int:
    """Write the request's body to ``path``; how many bytes, or -1 past
    ``limit``. A body that sends nothing for ``ARRIVE_IDLE_S`` times out."""
    total, body = 0, request.stream().__aiter__()
    with open(path, "wb") as out:
        while True:
            try:
                chunk = await asyncio.wait_for(body.__anext__(), rules.ARRIVE_IDLE_S)
            except StopAsyncIteration:
                return total
            total += len(chunk)
            if total > limit:
                return -1
            out.write(chunk)


@router.put("/{file_id}/content", response_model=FileOut, responses=REFUSALS, openapi_extra={
    "requestBody": {"required": True, "content": {"application/octet-stream": {
        "schema": {"type": "string", "format": "binary"}}}}})
async def put_content(file_id: uuid.UUID, request: Request, user: AuthUser = Depends(require_user)):
    """The file's bytes, as the body. No more of them than ``size`` said:
    that is what the quota counted. The answer is the file, now being checked."""
    with errors():
        record = await run_in_threadpool(service.expecting, user.subject, file_id)
        if not _arriving.acquire(blocking=False):
            raise refusal("busy", {"Retry-After": "5"})
        path = None
        try:
            path = service.arriving(file_id)
            async with asyncio.timeout(rules.ARRIVE_TIMEOUT_S):
                arrived = await _spool(request, path, min(record.size, rules.MAX_FILE_BYTES))
            if arrived <= 0:
                reason = "empty" if arrived == 0 else "too_large"
                await run_in_threadpool(service.refuse_arrival, user.subject, file_id, reason)
                raise refusal(reason)
            return file_out(await run_in_threadpool(service.arrived, user.subject, record, path, arrived))
        except TimeoutError:
            raise refusal("too_slow") from None      # the room stays reserved: it can be sent again
        except ClientDisconnect:
            raise HTTPException(400, "The upload was interrupted.") from None
        except OSError:
            raise refusal("storage_unavailable", {"Retry-After": "30"}) from None
        finally:
            if path is not None:
                service.discard(path)
            _arriving.release()


@router.get("/{file_id}", response_model=FileOut, responses=REFUSALS)
def get_file(file_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with errors():
        record = rows.get_file(user.subject, file_id)
        service.wake_scans(user.subject, [record])
        return file_out(record)


@router.post("/{file_id}/link", response_model=FileLink, responses=REFUSALS)
def post_link(file_id: uuid.UUID, response: Response, user: AuthUser = Depends(require_user)):
    """A download link for a clean file. It works for a minute."""
    with errors():
        url = service.link(user.subject, file_id)
    response.headers["Cache-Control"] = "no-store"
    return {"url": url, "expires_in": rules.LINK_S}


@router.get("/{file_id}/content", response_class=FileResponse, responses={
    **REFUSALS, 200: {"content": {"application/pdf": {}}, "description": "The file's bytes"}})
def get_content(file_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """A clean PDF's bytes, to its owner: what the review screen draws the
    filing's pages from in the browser (PLAN.md R2). Sent as an attachment,
    never shown on the API's own address."""
    with errors():
        record = rows.get_file(user.subject, file_id)
        if record.status != "clean":
            raise service.NotReady()
        if record.kind != "pdf":
            raise HTTPException(404, "File not found")
        target = service.spool() / f"view-{uuid.uuid4().hex}"
        try:
            service.store().fetch(record.key, target, rules.MAX_FILE_BYTES)
        except BaseException:
            service.discard(target)
            raise
    return FileResponse(target, media_type="application/pdf", background=BackgroundTask(service.discard, target),
                        headers={"Content-Disposition": "attachment", "Cache-Control": "private, no-store",
                                 "X-Content-Type-Options": "nosniff"})


@router.delete("/{file_id}", status_code=204, responses=REFUSALS)
def delete_file(file_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Delete the file for good: its row now, its bytes with it."""
    if not database_configured():
        raise refusal("not_set_up")
    try:
        service.remove(user.subject, file_id)
    except rows.FileNotFound:
        raise HTTPException(404, "File not found") from None
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# The local directory store's links (local runs and tests only)
# ---------------------------------------------------------------------------
@local.get("/{token}")
def get_local(token: str):
    store = local_store()
    if store is None:
        raise HTTPException(404, "Not found")
    try:
        payload = store.open_token(token, "get")
        target = service.spool() / f"download-{uuid.uuid4()}"
        try:
            store.fetch(payload["key"], target, rules.MAX_FILE_BYTES)
            data = target.read_bytes()
        finally:
            target.unlink(missing_ok=True)
    except (BadToken, StorageError):
        raise HTTPException(403, "This link is not valid, or no longer works.") from None
    except (NotStored, TooLarge):
        raise HTTPException(404, "Not found") from None
    name = quote(str(payload.get("name") or "file"), safe="")
    return Response(data, media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{name}",
        "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})
