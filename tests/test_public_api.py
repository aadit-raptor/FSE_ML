"""The public API (PLAN.md 7.7): API keys, version 1 and webhooks.

Done when: a script with a key runs a deal and receives a webhook, and a
revoked key stops immediately. The first test is exactly that: the documented
example script (docs/examples/variater_api.py, standard library only) talks
to a real server on a local port with a key, and a real web server of the
test's own receives the signed message and checks it with the script's
``verify``. Real sign-in throughout (the development sign-in, as local runs
use it), on a throwaway database.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import select

from api import auth, limits
from api import webhooks as sender
from api.main import app
from db import api_keys, webhooks
from db.engine import connect
from db.models import ApiKey, AuditEvent, Webhook, WebhookDelivery
from jobs import drill
from jobs.runner import Runner
from tests.test_deals import PROFILE

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
DEAL = {"ebitda": 120.0, "entry_mult": 9.0, "exit_mult": 10.0, "hold": 5, "growth": 6.0}


def _example():
    spec = importlib.util.spec_from_file_location("variater_api", ROOT / "docs" / "examples" / "variater_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _example()


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def site(fresh_db, monkeypatch, signed_out):  # noqa: ARG001 -- real sign-in on a database
    for name in ("CLERK_ISSUER", "CLERK_PUBLISHABLE_KEY", "RENDER_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    monkeypatch.setenv("FSE_ENV", "local")
    monkeypatch.setenv("FSE_WEBHOOK_ALLOW_LOCAL", "1")
    for who in ("alice", "bob"):
        assert client.post("/api/account", json=PROFILE, headers=bearer(f"dev:{who}")).status_code == 200
    yield
    sender.use_sender(None)


def new_key(who: str = "alice", name: str = "nightly script") -> str:
    made = client.post("/api/account/api-keys", json={"name": name}, headers=bearer(f"dev:{who}"))
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["token"].startswith("vk_") and body["hint"] == body["token"][-4:] and body["name"] == name
    return body["token"]


class Receiver:
    """A web server on a local port that keeps every message it is sent."""

    def __init__(self, status: int = 200):
        self.messages: list[tuple[dict, bytes]] = []
        self.arrived = threading.Event()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - the standard library's name
                body = self.rfile.read(int(self.headers["Content-Length"]))
                receiver.messages.append(({k: v for k, v in self.headers.items()}, body))
                self.send_response(receiver.status)
                self.end_headers()
                receiver.arrived.set()

            def log_message(self, *args):
                pass

        self.status = status
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/variater"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def wait(self, count: int = 1, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while len(self.messages) < count and time.monotonic() < deadline:
            self.arrived.wait(0.2)
        assert len(self.messages) >= count, f"{len(self.messages)} of {count} messages arrived"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def receiver():
    made = Receiver()
    yield made
    made.close()


@pytest.fixture
def server(site):  # noqa: ARG001
    """The API on a local port, as a script reaches it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not running.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert running.started
    yield f"http://127.0.0.1:{port}"
    running.should_exit = True
    thread.join(5)


def test_a_script_with_a_key_runs_a_deal_and_receives_a_webhook(server, receiver, job_queue):
    key = new_key()
    added = client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice"))
    assert added.status_code == 201, added.text
    secret = added.json()["secret"]
    assert secret.startswith("whsec_")

    # The script runs a deal: the answer is the site's own, to the cent
    answer = example.run_deal(key, DEAL, base=server)
    site_answer = client.post("/api/deal/run", json={"inputs": DEAL}, headers=bearer("dev:alice")).json()
    assert answer["returns"] == site_answer["returns"]
    assert answer["returns"]["irr"] == pytest.approx(site_answer["returns"]["irr"]) and answer["model"]

    # ... queues a simulation; the job runs; the webhook hears of it, signed
    job = example.submit_simulation(key, drill.DRILL_REQUEST, base=server)
    assert job["status"] == "queued" and job["webhooks"] == 1
    assert Runner(lambda: job_queue).run_next() == "succeeded"
    receiver.wait()
    headers, body = receiver.messages[0]
    message = example.verify(secret, headers, body)
    assert message["type"] == "job.finished"
    assert message["data"] == {"job_id": job["id"], "kind": "montecarlo.run", "status": "succeeded",
                               "error_status": None}
    # the message holds no result: the script fetches it with its key
    assert "result" not in json.dumps(message)
    done = example.wait_for(key, job["id"], base=server)
    direct = client.post("/api/montecarlo/run", json=drill.DRILL_REQUEST, headers=bearer("dev:alice")).json()
    assert done["result"]["summary"] == direct["summary"]

    # A message whose body or secret differs fails the script's check
    with pytest.raises(ValueError):
        example.verify(secret, headers, body.replace(b"succeeded", b"failed"))
    with pytest.raises(ValueError):
        example.verify(webhooks.new_secret(), headers, body)
    listed = client.get("/api/account/webhooks", headers=bearer("dev:alice")).json()["webhooks"]
    assert listed[0]["last_state"] == "delivered" and listed[0]["last_status"] == 200


def test_a_revoked_key_stops_immediately(server):
    key = new_key()
    assert example.call("GET", "/api/v1/me", key, base=server)["via"] == "key"
    keys = client.get("/api/account/api-keys", headers=bearer("dev:alice")).json()["keys"]
    assert len(keys) == 1 and keys[0]["last_used_at"] is not None and "token" not in keys[0]
    assert client.delete(f"/api/account/api-keys/{keys[0]['id']}", headers=bearer("dev:alice")).status_code == 204
    # the very next request, on the same server, is refused: nothing cached the key
    with pytest.raises(urllib.error.HTTPError) as refused:
        example.call("GET", "/api/v1/me", key, base=server)
    assert refused.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as again:
        example.run_deal(key, DEAL, base=server)
    assert again.value.code == 401
    assert client.get("/api/account/api-keys", headers=bearer("dev:alice")).json()["keys"] == []


def test_only_the_keys_hash_is_stored_and_making_and_revoking_are_in_the_history(site):  # noqa: ARG001
    key = new_key()
    with connect() as conn:
        stored = conn.execute(select(ApiKey.token_hash, ApiKey.name)).all()
        actions = conn.execute(select(AuditEvent.action, AuditEvent.detail)).all()
    assert stored == [(api_keys.token_hash(key), "nightly script")]
    assert key not in repr(stored)
    assert ("api_key_created", {}) in actions
    key_id = client.get("/api/account/api-keys", headers=bearer("dev:alice")).json()["keys"][0]["id"]
    client.delete(f"/api/account/api-keys/{key_id}", headers=bearer("dev:alice"))
    with connect() as conn:
        assert ("api_key_revoked", {}) in conn.execute(select(AuditEvent.action, AuditEvent.detail)).all()


@pytest.mark.parametrize("method, path", [
    ("get", "/api/deals"), ("post", "/api/deal/run"), ("get", "/api/account"), ("get", "/api/account/api-keys"),
    ("post", "/api/account/api-keys"), ("get", "/api/account/webhooks"), ("post", "/api/jobs"),
    ("get", "/api/files"), ("post", "/api/addin/sessions"), ("get", "/api/research/companies"),
    ("put", "/api/account/settings"),
])
def test_a_key_reaches_the_public_api_and_nothing_else(site, method, path):  # noqa: ARG001
    key = new_key()
    answer = getattr(client, method)(path, headers=bearer(key), **({"json": {}} if method in ("post", "put") else {}))
    assert answer.status_code == 403 and answer.json()["detail"] == "key_scope"


def test_the_key_path_is_version_one_exactly():
    assert auth.KEY_PATH.fullmatch("/api/v1") and auth.KEY_PATH.fullmatch("/api/v1/deals/x/run")
    for path in ("/api/v10/me", "/api/v1x", "/api/deals", "/api/v2/me", "/v1/me", "/api/account/v1"):
        assert not auth.KEY_PATH.fullmatch(path), path


def test_another_accounts_saved_deal_is_not_there_for_a_key(site):  # noqa: ARG001
    made = client.post("/api/deals", json={"name": "Bob's deal", "inputs": DEAL}, headers=bearer("dev:bob"))
    assert made.status_code == 201
    bobs = made.json()["id"]
    alice = new_key("alice")
    assert client.get(f"/api/v1/deals/{bobs}", headers=bearer(alice)).status_code == 404
    assert client.post(f"/api/v1/deals/{bobs}/run", headers=bearer(alice)).status_code == 404
    assert client.get("/api/v1/deals", headers=bearer(alice)).json()["deals"] == []
    # Bob's own key sees it, and running it gives what its screen shows
    bob = new_key("bob")
    assert [d["id"] for d in client.get("/api/v1/deals", headers=bearer(bob)).json()["deals"]] == [bobs]
    ran = client.post(f"/api/v1/deals/{bobs}/run", headers=bearer(bob))
    assert ran.status_code == 200 and ran.headers["Variater-API-Version"] == "1"
    screen = client.post("/api/deal/run", json={"inputs": made.json()["inputs"],
                                                "settings": made.json()["settings"]}, headers=bearer("dev:bob"))
    assert ran.json()["returns"] == screen.json()["returns"]


def test_another_accounts_keys_and_webhooks_cant_be_revoked_or_removed(site, receiver):  # noqa: ARG001
    new_key("alice")
    key_id = client.get("/api/account/api-keys", headers=bearer("dev:alice")).json()["keys"][0]["id"]
    hook = client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice")).json()
    assert client.delete(f"/api/account/api-keys/{key_id}", headers=bearer("dev:bob")).status_code == 404
    assert client.delete(f"/api/account/webhooks/{hook['id']}", headers=bearer("dev:bob")).status_code == 404
    assert client.post(f"/api/account/webhooks/{hook['id']}/ping", headers=bearer("dev:bob")).status_code == 404
    assert client.get("/api/account/webhooks", headers=bearer("dev:bob")).json()["webhooks"] == []
    assert len(client.get("/api/account/api-keys", headers=bearer("dev:alice")).json()["keys"]) == 1


def test_a_job_of_someone_with_no_webhooks_tells_nobody_and_another_persons_webhook_hears_nothing(
        site, receiver, job_queue):  # noqa: ARG001
    client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:bob"))
    key = new_key("alice")
    job = client.post("/api/v1/jobs", json={"kind": "montecarlo.run", "input": drill.DRILL_REQUEST},
                      headers=bearer(key))
    assert job.status_code == 202 and job.json()["webhooks"] == 0
    assert Runner(lambda: job_queue).run_next() == "succeeded"
    time.sleep(0.5)
    assert receiver.messages == []


def test_a_job_queued_on_the_site_tells_no_webhook(site, receiver, job_queue):  # noqa: ARG001
    client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice"))
    queued = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": drill.DRILL_REQUEST},
                         headers=bearer("dev:alice"))
    assert queued.status_code == 202
    Runner(lambda: job_queue).run_next()
    time.sleep(0.5)
    assert receiver.messages == []


def test_a_failing_endpoint_is_tried_again_then_given_up(site, monkeypatch):  # noqa: ARG001
    tried: list[str] = []
    sender.use_sender(lambda url, body, headers: tried.append(headers["webhook-id"]) or 503)
    client.post("/api/account/webhooks", json={"url": "https://hooks.example.com/v"}, headers=bearer("dev:alice"))
    key = new_key()
    job = client.post("/api/v1/jobs", json={"kind": "montecarlo.run", "input": drill.DRILL_REQUEST},
                      headers=bearer(key)).json()
    webhooks.job_finished(__import__("uuid").UUID(job["id"]), "montecarlo.run", "succeeded")
    assert sender.deliver_due() == {"delivered": 0, "failed": 0, "due": 1}
    # not due again until its backoff has passed
    assert sender.deliver_due() == {"delivered": 0, "failed": 0, "due": 0}
    with connect() as conn:
        conn.execute(WebhookDelivery.__table__.update().values(next_at=None).where(WebhookDelivery.state == "due"))
        conn.commit()
    for attempt in range(2, webhooks.MAX_ATTEMPTS + 1):
        with connect() as conn:
            conn.execute(WebhookDelivery.__table__.update().values(next_at=__import__("datetime").datetime(
                2000, 1, 1, tzinfo=__import__("datetime").timezone.utc)).where(WebhookDelivery.state == "due"))
            conn.commit()
        expected = {"delivered": 0, "failed": 1, "due": 0} if attempt == webhooks.MAX_ATTEMPTS \
            else {"delivered": 0, "failed": 0, "due": 1}
        assert sender.deliver_due() == expected
    assert len(tried) == webhooks.MAX_ATTEMPTS and len(set(tried)) == 1   # one message, the same id each time
    listed = client.get("/api/account/webhooks", headers=bearer("dev:alice")).json()["webhooks"]
    assert listed[0]["last_state"] == "failed" and listed[0]["last_status"] == 503


def test_a_ping_is_signed_and_sent(site, receiver):  # noqa: ARG001
    hook = client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice")).json()
    pinged = client.post(f"/api/account/webhooks/{hook['id']}/ping", headers=bearer("dev:alice"))
    assert pinged.status_code == 202
    receiver.wait()
    headers, body = receiver.messages[0]
    message = example.verify(hook["secret"], headers, body)
    assert message["type"] == "ping" and message["data"] == {"webhook_id": hook["id"]}
    assert headers["webhook-id"] == pinged.json()["message_id"]


def test_removing_a_webhook_removes_its_messages(site, receiver):  # noqa: ARG001
    hook = client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice")).json()
    key = new_key()
    client.post("/api/v1/jobs", json={"kind": "montecarlo.run", "input": drill.DRILL_REQUEST}, headers=bearer(key))
    assert client.delete(f"/api/account/webhooks/{hook['id']}", headers=bearer("dev:alice")).status_code == 204
    with connect() as conn:
        assert conn.execute(select(WebhookDelivery.id)).all() == []
        assert conn.execute(select(Webhook.id)).all() == []
        actions = [a for (a,) in conn.execute(select(AuditEvent.action)).all()]
    assert "webhook_added" in actions and "webhook_removed" in actions


def test_a_person_holds_at_most_five_keys_and_three_webhooks(site):  # noqa: ARG001
    for n in range(api_keys.MAX_PER_USER):
        new_key(name=f"key {n}")
    full = client.post("/api/account/api-keys", json={"name": "one more"}, headers=bearer("dev:alice"))
    assert full.status_code == 409 and full.json()["detail"] == {"code": "full_keys"}
    for n in range(webhooks.MAX_PER_USER):
        assert client.post("/api/account/webhooks", json={"url": f"https://hooks.example.com/{n}"},
                           headers=bearer("dev:alice")).status_code == 201
    more = client.post("/api/account/webhooks", json={"url": "https://hooks.example.com/x"},
                       headers=bearer("dev:alice"))
    assert more.status_code == 409 and more.json()["detail"] == {"code": "full_webhooks"}
    assert client.post("/api/account/api-keys", json={"name": "\u0007bell"},
                       headers=bearer("dev:bob")).json()["detail"] == {"code": "invalid_name"}


@pytest.mark.parametrize("url, code", [
    ("http://hooks.example.com/v", "not_https"),
    ("https://10.0.0.5/hook", "private_address"),
    ("https://169.254.169.254/latest/meta-data", "private_address"),
    ("https://[::1]/hook", "private_address"),
    ("https://[::ffff:192.168.1.1]/hook", "private_address"),
    ("https://localhost/hook", "private_address"),
    ("https://metadata.google.internal/hook", "private_address"),
    ("https://user:pass@hooks.example.com/v", "invalid_url"),
    ("https://hooks.example.com/v#x", "invalid_url"),
    ("ftp://hooks.example.com/v", "not_https"),
    ("https:///nohost", "invalid_url"),
    ("https://hooks.example.com/a b", "invalid_url"),
])
def test_a_webhook_must_be_https_on_the_public_internet(site, monkeypatch, url, code):  # noqa: ARG001
    monkeypatch.setenv("FSE_ENV", "production")      # local addresses are allowed only off the deployed copies
    with pytest.raises(sender.Refused) as refused:
        sender.check_url(url)
    assert refused.value.code == code


def test_local_addresses_are_never_allowed_on_production_or_staging(monkeypatch):
    monkeypatch.setenv("FSE_WEBHOOK_ALLOW_LOCAL", "1")
    for environment in ("production", "staging"):
        monkeypatch.setenv("FSE_ENV", environment)
        assert not sender.local_allowed()
        with pytest.raises(sender.Refused):
            sender.check_url("http://127.0.0.1:9000/hook")
    monkeypatch.setenv("FSE_ENV", "local")
    assert sender.check_url("http://127.0.0.1:9000/hook")


def test_a_name_that_resolves_to_a_private_address_is_never_sent_to(monkeypatch):
    monkeypatch.setenv("FSE_ENV", "production")
    answers = {"rebinding.example": ["93.184.215.14", "10.1.2.3"], "inside.example": ["192.168.0.10"]}
    monkeypatch.setattr(sender.socket, "getaddrinfo", lambda host, port, **kw: [
        (None, None, None, "", (a, port)) for a in answers[host]])
    connected: list = []
    import urllib3
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", lambda *a, **k: connected.append(a) or None)
    for host in answers:
        with pytest.raises(sender.Refused):
            sender.resolve(host, 443)
        assert sender.post(f"https://{host}/hook", b"{}", {}) == 0
    assert connected == []


def test_a_send_connects_to_the_address_it_checked(monkeypatch):
    """Pinned: the pool is opened on the checked address with the host name for TLS, so a second lookup
    (DNS rebinding) can't send the request elsewhere."""
    monkeypatch.setenv("FSE_ENV", "production")
    monkeypatch.setattr(sender.socket, "getaddrinfo", lambda host, port, **kw: [
        (None, None, None, "", ("93.184.215.14", port))])
    seen: dict = {}

    class Pool:
        def __init__(self, host, port, **kw):
            seen.update(host=host, port=port, **kw)

        def urlopen(self, method, path, **kw):
            seen.update(method=method, path=path, **kw)

            class Answer:
                status = 204

                def read(self, n):
                    return b""

                def release_conn(self):
                    pass
            return Answer()

        def close(self):
            pass

    import urllib3
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", Pool)
    assert sender.post("https://hooks.example.com:8443/in?x=1", b"{}", {"a": "b"}) == 204
    assert seen["host"] == "93.184.215.14" and seen["port"] == 8443
    assert seen["server_hostname"] == "hooks.example.com" and seen["assert_hostname"] == "hooks.example.com"
    assert seen["cert_reqs"] == "CERT_REQUIRED" and seen["redirect"] is False
    assert seen["headers"]["Host"] == "hooks.example.com:8443" and seen["path"] == "/in?x=1"


def test_the_signature_is_standard_webhooks(monkeypatch):
    # Standard Webhooks' own published example (github.com/standard-webhooks, the spec's test vector)
    secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    body = b'{"test": 2432232314}'
    assert sender.sign(webhooks.secret_bytes(secret), "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, body) \
        == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="


def test_a_keys_requests_have_limits_of_their_own(site, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(limits, "KEY_REQUESTS", limits.Rule("key_requests_per_minute", "requests with this API key",
                                                            "key", 3, 60))
    first, second = new_key(name="one"), new_key(name="two")
    for _ in range(3):
        assert client.get("/api/v1/me", headers=bearer(first)).status_code == 200
    refused = client.get("/api/v1/me", headers=bearer(first))
    assert refused.status_code == 429 and refused.json()["limit"]["rule"] == "key_requests_per_minute"
    assert "this API key" in refused.json()["detail"]
    # another key of the same person is counted apart, and the site's own sign-in is not held to it
    assert client.get("/api/v1/me", headers=bearer(second)).status_code == 200
    assert client.get("/api/v1/me", headers=bearer("dev:alice")).status_code == 200


def test_a_keys_model_runs_count_against_the_key_and_the_person(site, monkeypatch):  # noqa: ARG001
    monkeypatch.setattr(limits, "KEY_RUNS_DAY", limits.Rule("key_runs_per_day", "model runs with this API key",
                                                            "key", 1, 86_400, shared=True))
    key = new_key()
    body = {"kind": "montecarlo.run", "input": drill.DRILL_REQUEST}
    assert client.post("/api/v1/jobs", json=body, headers=bearer(key)).status_code == 202
    refused = client.post("/api/v1/jobs", json=body, headers=bearer(key))
    assert refused.status_code == 429 and refused.json()["limit"]["rule"] == "key_runs_per_day"
    assert limits.is_run_request("POST", "/api/v1/jobs") and limits.is_run_request("POST", "/api/v1/montecarlo/run")
    assert "/api/v1/montecarlo/run" in limits.SIMULATION_PATHS


def test_the_market_defaults_are_the_sites_own(site):  # noqa: ARG001
    key = new_key()
    for path in ("/defaults/industries", "/defaults/starting?country=DE&currency=EUR",
                 "/defaults/risk?country=DE&currency=EUR"):
        public = client.get("/api/v1" + path, headers=bearer(key))
        internal = client.get("/api/benchmarks" + path.removeprefix("/defaults"), headers=bearer("dev:alice"))
        assert public.status_code == internal.status_code == 200, (path, public.text)
        assert public.json() == internal.json()
    assert client.get("/api/v1/defaults/starting?country=germany&currency=EUR", headers=bearer(key)).status_code \
        == 422


def test_the_simulation_answers_what_the_site_answers(site):  # noqa: ARG001
    key = new_key()
    public = client.post("/api/v1/montecarlo/run", json=drill.DRILL_REQUEST, headers=bearer(key))
    internal = client.post("/api/montecarlo/run", json=drill.DRILL_REQUEST, headers=bearer("dev:alice"))
    assert public.status_code == 200 and public.json()["summary"] == internal.json()["summary"]
    assert public.json()["model"] == internal.json()["model"]


def test_a_cancelled_waiting_job_tells_its_webhooks(site, receiver):  # noqa: ARG001
    hook = client.post("/api/account/webhooks", json={"url": receiver.url}, headers=bearer("dev:alice")).json()
    key = new_key()
    job = client.post("/api/v1/jobs", json={"kind": "montecarlo.run", "input": drill.DRILL_REQUEST},
                      headers=bearer(key)).json()
    cancelled = client.post(f"/api/v1/jobs/{job['id']}/cancel", headers=bearer(key))
    assert cancelled.json()["status"] == "cancelled"
    receiver.wait()
    message = example.verify(hook["secret"], *receiver.messages[0])
    assert message["data"]["status"] == "cancelled" and message["data"]["job_id"] == job["id"]


def test_an_add_in_pass_cant_manage_keys_and_a_key_cant_be_used_as_a_pass(site):  # noqa: ARG001
    made = client.post("/api/addin/sessions", headers=bearer("dev:alice")).json()["token"]
    assert client.get("/api/account/api-keys", headers=bearer(made)).status_code == 403
    assert client.get("/api/v1/me", headers=bearer(made)).status_code == 403
    assert client.get("/api/v1/me", headers=bearer("vk_" + "x" * 43)).status_code == 401
    assert client.get("/api/v1/me", headers=bearer("vk_short")).status_code == 401
    auth.reset_addin_cache()


def test_the_nightly_maintenance_sends_what_is_due_and_deletes_old_messages(site, monkeypatch):  # noqa: ARG001
    from datetime import timedelta

    from db.models import utc_now
    from jobs import scheduled
    sent: list = []
    sender.use_sender(lambda url, body, headers: sent.append(url) or 200)
    client.post("/api/account/webhooks", json={"url": "https://hooks.example.com/n"}, headers=bearer("dev:alice"))
    hook = client.get("/api/account/webhooks", headers=bearer("dev:alice")).json()["webhooks"][0]
    webhooks.ping("dev:alice", __import__("uuid").UUID(hook["id"]))
    with connect() as conn:
        conn.execute(WebhookDelivery.__table__.insert().values(
            webhook_id=__import__("uuid").UUID(hook["id"]), event="ping", state="delivered",
            created_at=utc_now() - timedelta(days=webhooks.KEEP_DAYS + 1)))
        conn.commit()
    summary = scheduled.job_maintenance()
    assert summary["webhooks_delivered"] == 1 and summary["webhook_messages_deleted"] == 1
    assert sent == ["https://hooks.example.com/n"]
