"""The rules for uploaded files: every number, and why it is what it is.

**What may be uploaded.** PDF, Excel and Word documents, by their extension
*and* by their contents (``uploads/scan.py`` reads the bytes; a name proves
nothing). Macro-enabled formats (``.xlsm``, ``.docm``) are not accepted at all.

**How big.** ``MAX_FILE_BYTES`` a file. An annual report is typically 5-10 MB;
Supabase's free plan takes 50 MB in one request, and the scanner's own
process holds the file in memory on a server with 512 MB (measured:
``tests/test_uploads.py``). The API never does: an upload is streamed to
disk as it arrives.

**How much.** Supabase's free plan stores 1 GB per project. ``TOTAL_BUDGET_BYTES``
is what all users together may hold (``FSE_UPLOADS_BUDGET_MB`` lowers it where
the project is shared with something else); ``USER_QUOTA_BYTES`` and
``USER_MAX_FILES`` are one person's share. Room reserved for an upload that
has not arrived yet counts, a row counts until its bytes are really gone, and
at most ``MAX_IN_FLIGHT`` of a person's files may be on their way or being
checked at once, so nobody can queue scans for everyone else's to wait behind.

**How long.** Files are working material for reading figures out of documents
(PLAN.md 6.2), not an archive: each is deleted ``RETENTION_DAYS`` after it was
uploaded, and the screen says when. That is what keeps the free 1 GB enough.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

MB = 1024 * 1024

MAX_FILE_BYTES = 20 * MB
USER_QUOTA_BYTES = 100 * MB
USER_MAX_FILES = 40
# The screen warns from this share of either personal limit
NEAR_QUOTA = 0.8
# 1 GB free, less room for files being replaced and for the bucket's own overhead
TOTAL_BUDGET_BYTES = 800 * MB
MIN_BUDGET_MB, MAX_BUDGET_MB = 20, 900

RETENTION_DAYS = 90
# Room reserved for an upload is given back if the bytes have not arrived by then
PENDING_S = 30 * 60
# A person's files that may be arriving or being checked at the same time
MAX_IN_FLIGHT = 3
# Uploads the server takes in at once (each is streamed to disk, so this
# bounds open connections and disk, not memory)
MAX_ARRIVING = 2
# ... and how long one may take to arrive: 20 MB in this time is 67 KB a
# second. A connection that stalls must not hold one of those places for good
ARRIVE_TIMEOUT_S = 300
# A scan normally takes a second; one still unfinished after this long has
# lost its job (three restarts), and the file is refused rather than trusted
SCAN_STALE_S = 15 * 60
# A scan still unfinished this long after it was queued is queued again the
# next time its owner looks: the job was lost (a restart of an in-memory
# queue), failed on a store that could not be reached, or never fitted in a
# full queue. A second job for a file already scanned finds nothing to do
RESCAN_AFTER_S = 60
# A refused file's bytes are deleted at once; its row stays this long so the
# person can read why
REFUSED_KEEP_DAYS = 7
# A download link works this long: enough to start the download, no longer
LINK_S = 60

MAX_NAME_LENGTH = 120


@dataclass(frozen=True)
class Kind:
    id: str
    extension: str
    content_type: str


KINDS: dict[str, Kind] = {k.id: k for k in (
    Kind("pdf", ".pdf", "application/pdf"),
    Kind("xlsx", ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    Kind("docx", ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    Kind("xls", ".xls", "application/vnd.ms-excel"),
    Kind("doc", ".doc", "application/msword"),
)}
_BY_EXTENSION = {k.extension: k for k in KINDS.values()}

STATUSES = ("pending", "scanning", "clean", "refused")
# Why a file was refused. Codes, never sentences: the words are in
# web/messages/en.json ``files.reason_<code>``
REASONS = (
    "wrong_type",       # the contents are not the kind of file the name says
    "empty",
    "too_large",        # more bytes arrived than the limit, or than were declared
    "encrypted",        # password-protected: it can't be checked
    "macros",
    "active_content",   # scripts, launch actions, embedded files or objects, remote templates
    "archive_bomb",     # expands to far more than its size
    "unreadable",       # built in a way the scanner can't follow, so it can't be vouched for
    "scan_failed",      # the scanner itself failed; the file is refused, not trusted
)


class InvalidUpload(ValueError):
    """An upload the rules refuse before a byte is sent. ``code`` is one of
    ``REFUSALS``; the words are the web's."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# Why an upload is refused before it starts
REFUSALS = ("bad_name", "bad_type", "empty", "too_large", "quota_bytes", "quota_files", "storage_full",
            "in_flight")

_CONTROL = re.compile(r"[\x00-\x1f\x7f\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def clean_name(name: object) -> str:
    r"""The file's display name: its last path component, with control and
    text-direction characters removed (a right-to-left override can make
    ``report\u202Efdp.exe`` read as ``reportexe.pdf``) and spaces collapsed."""
    text = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = " ".join(_CONTROL.sub("", text).split())
    if not text or text.startswith(".") or len(text) > MAX_NAME_LENGTH:
        raise InvalidUpload("bad_name")
    return text


def kind_of(name: str) -> Kind:
    """The kind a (cleaned) name says the file is."""
    dot = name.rfind(".")
    kind = _BY_EXTENSION.get(name[dot:].lower()) if dot > 0 else None
    if kind is None:
        raise InvalidUpload("bad_type")
    return kind


def check_size(size: int) -> int:
    if size <= 0:
        raise InvalidUpload("empty")
    if size > MAX_FILE_BYTES:
        raise InvalidUpload("too_large")
    return size


def total_budget_bytes(environ: Optional[dict] = None) -> int:
    raw = (environ if environ is not None else os.environ).get("FSE_UPLOADS_BUDGET_MB", "").strip()
    try:
        megabytes = int(raw) if raw else TOTAL_BUDGET_BYTES // MB
    except ValueError:
        megabytes = TOTAL_BUDGET_BYTES // MB
    return max(MIN_BUDGET_MB, min(megabytes, MAX_BUDGET_MB)) * MB


def check_room(size: int, *, user_bytes: int, user_files: int, total_bytes: int,
               budget: Optional[int] = None, in_flight: int = 0) -> None:
    """Refuse an upload of ``size`` bytes that would pass a limit."""
    if in_flight >= MAX_IN_FLIGHT:
        raise InvalidUpload("in_flight")
    if user_files + 1 > USER_MAX_FILES:
        raise InvalidUpload("quota_files")
    if user_bytes + size > USER_QUOTA_BYTES:
        raise InvalidUpload("quota_bytes")
    if total_bytes + size > (total_budget_bytes() if budget is None else budget):
        raise InvalidUpload("storage_full")


def usage(user_bytes: int, user_files: int) -> dict:
    """What the person holds against their limits, for the screen."""
    share = max(user_bytes / USER_QUOTA_BYTES, user_files / USER_MAX_FILES)
    return {"bytes": user_bytes, "files": user_files, "quota_bytes": USER_QUOTA_BYTES,
            "quota_files": USER_MAX_FILES, "max_file_bytes": MAX_FILE_BYTES,
            "near_limit": share >= NEAR_QUOTA, "full": share >= 1.0,
            "retention_days": RETENTION_DAYS}
