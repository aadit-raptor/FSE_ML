"""Which AI reads the filings: a switch, not a dependency (PLAN.md R1, 12.4).

``AI_PROVIDER`` chooses:

- ``gemini``: Google's Gemini API with ``GEMINI_API_KEY``. Decided by the
  user (2026-10-14): R1 is built on a **free** key (Google project
  ``variater-reader``); billing with a spending cap is switched on before
  the first outside showing after R2. On the free tier Google may use what
  it is sent to improve its products, so ``AI_FREE_TIER`` (on unless set to
  ``0``) makes every read ask the person to confirm the file is public.
- ``replay``: recorded answers (``tests/fixtures/research/reader``), looked
  up by the document's SHA-256. What CI, the browser tests and development
  use: **no AI call, nothing sent anywhere**. Refused in production.
- unset: the reader is off and says so.

Read A and read B go to **two different full models** (R0's decision:
never one read, never a lite model). Each read tries its list of models in
order; a model whose day's allowance is spent (a 429 that names a daily
quota) is skipped for ``SPENT_FOR_S``; "high demand" (a 503) moves on to the
next model, and spends nothing.

**Nothing from a document and no key reaches a message or a log**: errors
carry a code and the HTTP status only. The key goes in a header, never in a
URL.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from research.reader import prompt

BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
# Read A by these, in order; read B by these, never by A's own model (R0's pairs)
MODELS_A = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash")
MODELS_B = ("gemini-3-flash-preview", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash")
INLINE_LIMIT = 13_000_000      # bytes of PDF: base64 adds a third and a request holds about 20 MB
TIMEOUT_S = 600.0
SPENT_FOR_S = 6 * 3600         # Google's day is a rolling 24 hours; look again in six
STATUS_CACHE_S = 300.0
RECORDINGS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "research" / "reader"
# The report writer's recorded answers (PLAN.md R8), by the SHA-256 of the prompt (research/writer.py)
WRITINGS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "research" / "writer"

# Errors, as codes the screen words (web/messages/en.json reader.error_*)
SPENT = "allowance_spent"          # every model's day is used up: try again tomorrow
BUSY = "provider_busy"             # every model said "high demand": try again in minutes
KEY_REFUSED = "key_refused"
UNREADABLE = "unreadable_answer"
NETWORK = "provider_unreachable"
NOT_RECORDED = "not_recorded"
BLOCKED = "answer_blocked"


class ProviderError(RuntimeError):
    def __init__(self, code: str, status: Optional[int] = None):
        self.code, self.status = code, status
        super().__init__(code + (f" (HTTP {status})" if status else ""))


@dataclass(frozen=True)
class Document:
    """What a read is given: the PDF for read A, pictures of pages for B."""
    sha256: str
    pdf: Optional[bytes] = None
    pictures: Sequence[bytes] = ()


@dataclass(frozen=True)
class Answer:
    read: object                    # the JSON the model answered, as parsed
    model: str
    prompt_version: str
    usage: Mapping = field(default_factory=dict)
    seconds: float = 0.0


@dataclass(frozen=True)
class Instructions:
    """Another task's prompt for a read: its version, the words for the whole PDF and for pictures of pages."""
    version: str
    pdf: str
    pictures: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json(text: str) -> object:
    """The model's text as JSON; a fenced block is unwrapped."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        return json.loads(body)
    except ValueError:
        raise ProviderError(UNREADABLE) from None


def free_tier() -> bool:
    return os.environ.get("AI_FREE_TIER", "1").strip() != "0"


def _models(variable: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = os.environ.get(variable, "")
    found = tuple(m.strip() for m in raw.split(",") if m.strip())
    return found or tuple(default)


class Gemini:
    name = "gemini"
    needs_pictures = True
    _spent: dict[str, float] = {}
    _lock = threading.Lock()

    def __init__(self, key: str, *, session=None, high_resolution: Optional[bool] = None):
        import requests

        self._key = key
        self._http = session or requests.Session()
        self.models_a = _models("GEMINI_MODELS_A", MODELS_A)
        self.models_b = _models("GEMINI_MODELS_B", MODELS_B)
        self.high = (os.environ.get("READER_HIGH_RESOLUTION", "0") == "1") if high_resolution is None \
            else high_resolution
        self.free_tier = free_tier()

    # -- the calls -----------------------------------------------------------
    def _post(self, url: str, **kwargs):
        try:
            return self._http.post(url, headers={"x-goog-api-key": self._key, **kwargs.pop("headers", {})},
                                   timeout=kwargs.pop("timeout", TIMEOUT_S), **kwargs)
        except Exception:  # noqa: BLE001 -- the library's text can hold the URL; keep only the fact
            raise ProviderError(NETWORK) from None

    def _generate(self, model: str, parts: list) -> tuple[str, Mapping]:
        config = {"temperature": 0, "responseMimeType": "application/json"}
        if self.high:
            config["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"
        answer = self._post(f"{BASE}/models/{model}:generateContent",
                            json={"contents": [{"parts": parts}], "generationConfig": config})
        try:
            data = answer.json() if answer.content else {}
        except ValueError:
            data = {}
        if answer.status_code == 200:
            candidates = data.get("candidates") or []
            if not candidates:
                raise ProviderError(BLOCKED)
            content = candidates[0].get("content") or {}
            text = "".join(p.get("text", "") for p in content.get("parts") or [] if isinstance(p, dict))
            return text, data.get("usageMetadata") or {}
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        if answer.status_code == 429 and "PerDay" in json.dumps(error):
            raise ProviderError(SPENT, 429)
        if answer.status_code in (429, 500, 503, 504):
            raise ProviderError(BUSY, answer.status_code)
        if answer.status_code in (400, 401, 403) and "API_KEY" in json.dumps(error):
            raise ProviderError(KEY_REFUSED, answer.status_code)
        raise ProviderError(NETWORK, answer.status_code)

    def _upload(self, pdf: bytes) -> tuple[dict, str]:
        """A PDF too big to send inline goes to Google's file store (kept 48
        hours, deleted as soon as the read is done)."""
        start = self._post(UPLOAD, timeout=120, json={"file": {"display_name": "results"}}, headers={
            "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(pdf)),
            "X-Goog-Upload-Header-Content-Type": "application/pdf"})
        target = start.headers.get("x-goog-upload-url") if start.status_code == 200 else None
        if not target or not target.startswith("https://generativelanguage.googleapis.com/"):
            raise ProviderError(NETWORK, start.status_code)
        sent = self._post(target, timeout=900, data=pdf,
                          headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
        if sent.status_code != 200:
            raise ProviderError(NETWORK, sent.status_code)
        info = sent.json().get("file") or {}
        for _ in range(60):
            if info.get("state") != "PROCESSING":
                break
            time.sleep(3)
            try:
                info = self._http.get(f"{BASE}/{info['name']}", headers={"x-goog-api-key": self._key},
                                      timeout=60).json()
            except Exception:  # noqa: BLE001
                raise ProviderError(NETWORK) from None
        if info.get("state") != "ACTIVE":
            raise ProviderError(NETWORK)
        return {"file_data": {"mime_type": "application/pdf", "file_uri": info["uri"]}}, info["name"]

    def _forget(self, name: str) -> None:
        try:
            self._http.delete(f"{BASE}/{name}", headers={"x-goog-api-key": self._key}, timeout=30)
        except Exception:  # noqa: BLE001 -- Google deletes it within 48 hours anyway
            pass

    # -- the reader's side ---------------------------------------------------
    def _usable(self, models: Sequence[str], exclude: frozenset) -> list[str]:
        now = time.time()
        with self._lock:
            return [m for m in models if m not in exclude and self._spent.get(m, 0.0) <= now]

    def read(self, which: str, document: Document, *, exclude: frozenset = frozenset(),
             instructions: Optional["Instructions"] = None) -> Answer:
        """One read; ``instructions`` another task's prompt (PLAN.md 6.2's document reader), else the results
        reader's."""
        models = self.models_a if which == "A" else self.models_b
        uploaded = None
        if which == "A":
            if document.pdf is None:
                raise ValueError("read A needs the PDF")
            if len(document.pdf) <= INLINE_LIMIT:
                parts = [{"inline_data": {"mime_type": "application/pdf",
                                          "data": base64.b64encode(document.pdf).decode()}}]
            else:
                part, uploaded = self._upload(document.pdf)
                parts = [part]
        else:
            parts = [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(p).decode()}}
                     for p in document.pictures]
        version = instructions.version if instructions else prompt.PROMPT_VERSION
        parts.append({"text": (instructions.pictures if which == "B" else instructions.pdf) if instructions
                      else prompt.text(prompt.PROMPT_VERSION, pictures=which == "B")})
        # Every model spent is "try tomorrow"; anything else (busy, a bad
        # answer) is "try again soon", which is what the person is told
        failure = ProviderError(SPENT)
        try:
            for model in self._usable(models, exclude):
                started = time.time()
                try:
                    text, usage = self._generate(model, parts)
                    read = parse_json(text)
                except ProviderError as exc:
                    if exc.code == KEY_REFUSED:
                        raise
                    if exc.code == SPENT:
                        with self._lock:
                            self._spent[model] = time.time() + SPENT_FOR_S
                    else:
                        failure = exc
                    continue
                return Answer(read, model, version, dict(usage), round(time.time() - started, 1))
        finally:
            if uploaded:
                self._forget(uploaded)
        raise failure

    # -- the report writer's side (PLAN.md R8) --------------------------------
    def write(self, text: str) -> Answer:
        """Linking sentences for a note: `text` is the whole prompt (research/writer.py), which holds no company,
        no figure and nothing from a document. One request, by the first model with allowance left."""
        failure = ProviderError(SPENT)
        for model in self._usable(_models("GEMINI_MODELS_WRITE", MODELS_A), frozenset()):
            started = time.time()
            try:
                answer, usage = self._generate(model, [{"text": text}])
                parsed = parse_json(answer)
            except ProviderError as exc:
                if exc.code == KEY_REFUSED:
                    raise
                if exc.code == SPENT:
                    with self._lock:
                        self._spent[model] = time.time() + SPENT_FOR_S
                else:
                    failure = exc
                continue
            return Answer(parsed, model, "", dict(usage), round(time.time() - started, 1))
        raise failure

    def status(self) -> dict:
        """Whether the key works, by listing the models (free: it spends no
        allowance). Never the key itself."""
        try:
            answer = self._http.get(f"{BASE}/models", params={"pageSize": 1000},
                                    headers={"x-goog-api-key": self._key}, timeout=20)
        except Exception:  # noqa: BLE001
            return {"status": "error", "provider": self.name, "key": "unchecked", "reason": NETWORK}
        if answer.status_code != 200:
            refused = answer.status_code in (400, 401, 403)
            return {"status": "error", "provider": self.name, "key": "refused" if refused else "unchecked",
                    "reason": KEY_REFUSED if refused else NETWORK, "http_status": answer.status_code}
        try:
            names = {str(m.get("name", "")).removeprefix("models/") for m in answer.json().get("models") or []}
        except ValueError:
            names = set()
        wanted = list(dict.fromkeys(self.models_a + self.models_b))
        listed = [m for m in wanted if m in names]
        return {"status": "ok" if any(m in names for m in self.models_a) and
                any(m in names for m in self.models_b) else "error",
                "provider": self.name, "key": "accepted", "free_tier": self.free_tier,
                "models_listed": listed, "models_missing": [m for m in wanted if m not in names]}


class Replay:
    """Recorded answers, by the document's SHA-256 (``research/reader/record.py``)."""

    name = "replay"
    needs_pictures = False         # the recorded answers stand for the reads

    def __init__(self, folder: Path = RECORDINGS, default: Optional[str] = None):
        self.folder = folder
        self.default = default if default is not None else os.environ.get("FSE_READER_REPLAY_DEFAULT") or None
        self._index: Optional[dict[str, Path]] = None
        # the recordings stand for the provider, tier and all: switching AI_PROVIDER changes nothing else
        self.free_tier = free_tier()

    def recordings(self) -> dict[str, Path]:
        if self._index is None:
            index = {}
            for path in sorted(self.folder.rglob("*.json")):
                try:
                    found = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                digest = (found.get("document") or {}).get("sha256") if isinstance(found, dict) else None
                if digest:
                    index[digest] = path
            self._index = index
        return self._index

    def recording(self, digest: str) -> dict:
        path = self.recordings().get(digest)
        if path is None and self.default:
            path = next((p for p in self.recordings().values() if p.stem == self.default), None)
        if path is None:
            raise ProviderError(NOT_RECORDED)
        return json.loads(path.read_text(encoding="utf-8"))

    def read(self, which: str, document: Document, *, exclude: frozenset = frozenset(),
             instructions: Optional[Instructions] = None) -> Answer:
        found = self.recording(document.sha256)
        read = (found.get("reads") or {}).get(which)
        if not read:
            raise ProviderError(NOT_RECORDED)
        return Answer(read["read"], read["model"], found.get("prompt", "r0"), read.get("usage") or {},
                      read.get("seconds") or 0.0)

    def write(self, text: str) -> Answer:
        path = WRITINGS / ("%s.json" % sha256(text.encode("utf-8")))
        if not path.is_file():
            raise ProviderError(NOT_RECORDED)
        found = json.loads(path.read_text(encoding="utf-8"))
        return Answer(found["answer"], found["model"], "", found.get("usage") or {}, found.get("seconds") or 0.0)

    def status(self) -> dict:
        return {"status": "ok", "provider": self.name, "key": "none", "free_tier": self.free_tier,
                "recordings": len(self.recordings())}


def get_provider():
    """The configured provider, or None when the reader is off."""
    from api.observability import deploy_environment

    choice = os.environ.get("AI_PROVIDER", "").strip().lower()
    if choice == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        return Gemini(key) if key else None
    if choice == "replay" and deploy_environment() != "production":
        return Replay()
    return None


_status: list = [0.0, None, None]


def status(*, now: Optional[float] = None) -> dict:
    """``/api/health/reader``: which provider, whether its key works (never
    the key). Cached five minutes; never touches the database."""
    now = time.time() if now is None else now
    choice = os.environ.get("AI_PROVIDER", "").strip().lower() or "none"
    key_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    fingerprint = (choice, key_set)
    if _status[1] is not None and _status[2] == fingerprint and now - _status[0] < STATUS_CACHE_S:
        return _status[1]
    provider = get_provider()
    if provider is None:
        answer = {"status": "off", "provider": choice,
                  "key": "missing" if choice == "gemini" and not key_set else "none"}
    else:
        answer = provider.status()
    _status[:] = [now, answer, fingerprint]
    return answer
