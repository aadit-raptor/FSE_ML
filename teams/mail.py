"""Sending an email (PLAN.md 7.2): Resend's API, a folder for local runs, or nothing.

``get_mail`` chooses:

- ``ResendMail`` when ``RESEND_API_KEY`` is set, from ``FSE_MAIL_FROM``
  (default ``Variater <notify@send.variater.com>``, a sender on the domain
  verified with Resend); the key goes in a header only, and every email
  carries an idempotency key, so a retry after a timeout never sends twice;
- ``FolderMail`` when ``FSE_MAIL_DIR`` is set and the environment is not
  production: each email a JSON file there (local runs, the browser tests);
- otherwise ``None``: mentions wait (teams/notices.py) until one is set up.

``send`` raises ``MailError`` with a reason code only: an address, a subject or a
body never reaches a message or a log.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol

RESEND_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "Variater <notify@send.variater.com>"
TIMEOUT_S = 15.0


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    text: str
    # One per batch of notices: the same batch sent twice is one email
    key: str


class MailError(RuntimeError):
    """The email wasn't sent: ``reason`` is ``refused`` (the service said no, try later), ``rejected`` (this
    email can never be sent as it is) or ``unreachable``."""

    def __init__(self, reason: str, status: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status


class Mail(Protocol):
    kind: str

    def send(self, email: Email) -> None: ...


# A POST: (url, JSON body, headers) -> status; tests swap it
Post = Callable[[str, Mapping, Mapping[str, str]], int]


def _requests_post(url: str, body: Mapping, headers: Mapping[str, str]) -> int:
    import requests

    return requests.post(url, json=dict(body), headers=dict(headers), timeout=TIMEOUT_S,
                         allow_redirects=False).status_code


class ResendMail:
    kind = "resend"

    def __init__(self, api_key: str, sender: str = DEFAULT_FROM, post: Post = _requests_post):
        self._key = api_key
        self._sender = sender
        self._post = post

    def send(self, email: Email) -> None:
        headers = {"Authorization": f"Bearer {self._key}", "Idempotency-Key": email.key}
        body = {"from": self._sender, "to": [email.to], "subject": email.subject, "text": email.text}
        try:
            status = self._post(RESEND_URL, body, headers)
        except Exception:
            raise MailError("unreachable") from None
        if 200 <= status < 300:
            return
        # 429: over a limit; 5xx: their outage. Either may pass. Anything else is this email's fault
        raise MailError("refused" if status == 429 or status >= 500 else "rejected", status)


class FolderMail:
    """Each email a JSON file in a folder: what a local run "sends"."""
    kind = "folder"

    def __init__(self, folder: Path):
        self.folder = folder

    def send(self, email: Email) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{email.key}.json"
        path.write_text(json.dumps({"to": email.to, "subject": email.subject, "text": email.text},
                                   ensure_ascii=False, indent=1), encoding="utf-8")


class OutboxMail:
    """Emails kept in memory, for the tests."""
    kind = "outbox"

    def __init__(self):
        self.sent: list[Email] = []
        self.fail: Optional[MailError] = None

    def send(self, email: Email) -> None:
        if self.fail is not None:
            raise self.fail
        if any(e.key == email.key for e in self.sent):
            return
        self.sent.append(email)


_lock = threading.Lock()
_mail: Optional[Mail] = None
_chosen = False


def _environment() -> str:
    from api.auth import deploy_environment
    return deploy_environment()


def get_mail() -> Optional[Mail]:
    global _mail, _chosen
    with _lock:
        if not _chosen:
            key = os.environ.get("RESEND_API_KEY", "").strip()
            folder = os.environ.get("FSE_MAIL_DIR", "").strip()
            if key:
                _mail = ResendMail(key, os.environ.get("FSE_MAIL_FROM", "").strip() or DEFAULT_FROM)
            elif folder and _environment() != "production":
                _mail = FolderMail(Path(folder))
            else:
                _mail = None
            _chosen = True
        return _mail


def use_mail(mail: Optional[Mail], *, choose_again: bool = False) -> None:
    """Use this sender (tests); ``choose_again`` reads the environment on the next call instead."""
    global _mail, _chosen
    with _lock:
        _mail = mail
        _chosen = not choose_again


def batch_key(recipient_id: int, notice_ids: list[int]) -> str:
    """The same notices to the same person give the same key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"variater-mentions:{recipient_id}:" +
                          ",".join(str(n) for n in sorted(notice_ids))))
