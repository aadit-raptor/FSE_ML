"""Sending webhook messages (PLAN.md 7.7): the address checks, the signature and the delivery thread.

**Signed as Standard Webhooks sign** (https://www.standardwebhooks.com), so
the libraries written for it check ours: headers ``webhook-id`` (the
message's id), ``webhook-timestamp`` (Unix seconds) and ``webhook-signature``
(``v1,`` and the base64 HMAC-SHA256, keyed with the secret's bytes, of
``<id>.<timestamp>.<body>``). docs/public-api.md shows the check.

**Only to the public internet.** An address is ``https`` with a host name or a
public address, no user name or password, at most 500 characters
(``check_url``). Each send resolves the host, refuses the lot if any address
is not public (private, loopback, link-local, reserved, multicast: so a
webhook can't reach the API's own network), and then connects to the address
it checked, sending the host name for TLS and in ``Host`` -- so a name that
answers differently the second time (DNS rebinding) changes nothing. No
redirect is followed and at most ``MAX_ANSWER_BYTES`` of the answer are read.

Local runs and the tests may send to ``http://127.0.0.1`` with
``FSE_WEBHOOK_ALLOW_LOCAL=1``, which production and staging ignore.

**The delivery thread** starts when a message becomes due (``kick``), sends
what is due, waits for retries up to ``MAX_WAIT_S`` and stops: nothing runs
while nothing is owed. The nightly job maintenance sends anything still due
(a restart mid-wait) and deletes old messages. Nothing about a message's
address, answer or secret is ever logged: only its status and how many.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import urlsplit

from api.observability import deploy_environment, log_event

URL_MAX = 500
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 10.0
MAX_ANSWER_BYTES = 4096
# The thread waits this long at most for a retry; anything later is left to the next kick or the nightly run
MAX_WAIT_S = 330.0
USER_AGENT = "Variater-Webhooks/1 (+https://variater.com)"


class Refused(ValueError):
    """An address a webhook may not use; ``code`` says why (``invalid_url``, ``not_https``, ``private_address``)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def local_allowed() -> bool:
    if deploy_environment() in ("production", "staging"):
        return False
    return (os.environ.get("FSE_WEBHOOK_ALLOW_LOCAL") or "").strip() in ("1", "true", "True")


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if local_allowed() and ip.is_loopback:
        return True
    return ip.is_global and not ip.is_multicast


def check_url(url: str) -> str:
    """The address as it will be used, or ``Refused``. Host names are resolved when sending, not here."""
    url = (url or "").strip()
    if not url or len(url) > URL_MAX or any(ord(ch) <= 32 or ord(ch) == 127 for ch in url):
        raise Refused("invalid_url")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise Refused("invalid_url") from None
    host = parts.hostname
    if not host or parts.username is not None or parts.password is not None or parts.fragment:
        raise Refused("invalid_url")
    if parts.scheme != "https" and not (parts.scheme == "http" and local_allowed()):
        raise Refused("not_https")
    if port == 0:
        raise Refused("invalid_url")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _public(str(literal)):
        raise Refused("private_address")
    if literal is None and (host == "localhost" or host.endswith((".localhost", ".local", ".internal"))) \
            and not local_allowed():
        raise Refused("private_address")
    return url


def sign(secret: bytes, message_id: str, timestamp: int, body: bytes) -> str:
    signed = f"{message_id}.{timestamp}.".encode("utf-8") + body
    return "v1," + base64.b64encode(hmac.new(secret, signed, hashlib.sha256).digest()).decode("ascii")


def resolve(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to; ``Refused`` if one isn't public or there are none."""
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise Refused("invalid_url") from None
    addresses = list(dict.fromkeys(info[4][0] for info in found))
    if not addresses or not all(_public(a) for a in addresses):
        raise Refused("private_address")
    return addresses


def post(url: str, body: bytes, headers: dict) -> int:
    """POST ``body`` to the checked address, pinned to the address that was checked. The HTTP status; 0 for none."""
    import certifi
    import urllib3

    try:
        url = check_url(url)
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        address = resolve(parts.hostname, port)[0]
    except Refused:
        return 0
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    host_header = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
    timeout = urllib3.Timeout(connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S)
    if parts.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(address, port, server_hostname=parts.hostname,
                                           assert_hostname=parts.hostname, cert_reqs="CERT_REQUIRED",
                                           ca_certs=certifi.where(), timeout=timeout, retries=False, maxsize=1)
    else:
        pool = urllib3.HTTPConnectionPool(address, port, timeout=timeout, retries=False, maxsize=1)
    try:
        answer = pool.urlopen("POST", path, body=body, headers={**headers, "Host": host_header},
                              redirect=False, retries=False, preload_content=False, assert_same_host=False)
        try:
            answer.read(MAX_ANSWER_BYTES)
        finally:
            answer.release_conn()
        return int(answer.status)
    except Exception:  # noqa: BLE001 - any failure to deliver is "no answer", retried later
        return 0
    finally:
        pool.close()


# Tests swap the sender (``use_sender``); everything else goes through ``post``
_sender: list[Callable[[str, bytes, dict], int]] = [post]


def use_sender(sender: Optional[Callable[[str, bytes, dict], int]]) -> None:
    _sender[0] = sender or post


def deliver(message) -> str:
    """Send one claimed message (``db.webhooks.Claimed``) and record the answer. Its new state."""
    from db import webhooks as store

    body = json.dumps(message.payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    stamp = int(time.time())
    headers = {"content-type": "application/json", "user-agent": USER_AGENT,
               "webhook-id": str(message.id), "webhook-timestamp": str(stamp),
               "webhook-signature": sign(store.secret_bytes(message.secret), str(message.id), stamp, body)}
    status = _sender[0](message.url, body, headers)
    state = store.settle(message, status)
    log_event("webhook_sent", status=status, state=state, attempt=message.attempts)
    return state


def deliver_due(*, limit: int = 200) -> dict:
    """Send every message due now (at most ``limit``). How many went, failed for good, or wait for a retry."""
    from db import webhooks as store

    counts = {"delivered": 0, "failed": 0, "due": 0}
    for _ in range(limit):
        message = store.claim()
        if message is None:
            break
        counts[deliver(message)] += 1
    return counts


_thread_lock = threading.Lock()
_thread: list[Optional[threading.Thread]] = [None]
_wake = threading.Event()


def _loop() -> None:
    from db import webhooks as store
    from db.models import utc_now

    while True:
        try:
            deliver_due()
            upcoming: Optional[datetime] = store.next_due()
        except Exception as exc:  # noqa: BLE001 - the nightly run sends what this misses
            log_event("webhook_thread_error", logging.WARNING, error=type(exc).__name__)
            return
        if upcoming is None:
            return
        wait = (upcoming - utc_now()).total_seconds()
        if wait > MAX_WAIT_S:
            return
        _wake.wait(max(0.0, wait))
        _wake.clear()


def _run() -> None:
    try:
        _loop()
    finally:
        with _thread_lock:
            _thread[0] = None


def kick() -> None:
    """Send what is due, in a thread of its own (started if it isn't running)."""
    from db.engine import is_configured
    if not is_configured():
        return
    with _thread_lock:
        if _thread[0] is None:
            _thread[0] = threading.Thread(target=_run, name="fse-webhooks", daemon=True)
            _thread[0].start()
    _wake.set()


def job_finished(job_id, kind: str, status: str, error_status: Optional[int] = None) -> None:
    """The runner's call when a job ends: its waiting messages become due and are sent. Never raises."""
    from db import webhooks as store
    from db.engine import is_configured
    if not is_configured():
        return
    try:
        if store.job_finished(job_id, kind, status, error_status):
            kick()
    except Exception as exc:  # noqa: BLE001 - a job's end never fails because a message couldn't be queued
        log_event("webhook_queue_failed", logging.WARNING, error=type(exc).__name__)


__all__ = ["Refused", "check_url", "deliver", "deliver_due", "job_finished", "kick", "local_allowed", "post",
           "resolve", "sign", "use_sender"]
