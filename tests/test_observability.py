"""Monitoring (PLAN.md 1.2): request IDs, JSON logs, model-run timings and
Sentry events that carry the request ID but no deal contents."""
import io
import json
import logging
from datetime import datetime, timezone

import pytest
import sentry_sdk
from fastapi.testclient import TestClient
from sentry_sdk.transport import Transport

import api.main as api_main
from api import observability
from api.main import app

client = TestClient(app)

# A deal body with values that must never reach Sentry or the logs
SECRET_EBITDA = 987.654321
DEAL_BODY = {"inputs": {"ebitda": SECRET_EBITDA}}


class CaptureTransport(Transport):
    def __init__(self, options=None):
        super().__init__(options)
        self.events = []

    def capture_envelope(self, envelope):
        for item in envelope.items:
            if item.type == "event":
                self.events.append(item.payload.json)


@pytest.fixture
def sentry_events():
    transport = CaptureTransport()
    assert observability.init_sentry("https://key@o0.ingest.sentry.io/1", environment="test",
                                     release="abc123", transport=transport)
    yield transport.events
    sentry_sdk.init(dsn=None)


@pytest.fixture
def json_logs():
    """Lines written by the fse logger, parsed."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(observability.JsonFormatter())
    observability.logger.addHandler(handler)
    try:
        yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    finally:
        observability.logger.removeHandler(handler)


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setenv("FSE_ENV", "staging")
    monkeypatch.setattr(api_main, "_last_test_error", [0.0])


def test_request_id_is_kept_or_replaced():
    r = client.get("/api/settings/defaults", headers={"X-Request-ID": "web-1234abcd"})
    assert r.headers["X-Request-ID"] == "web-1234abcd"
    for bad in ("short", "has spaces in it!", "x" * 65):
        r = client.get("/api/settings/defaults", headers={"X-Request-ID": bad})
        assert r.headers["X-Request-ID"] != bad and len(r.headers["X-Request-ID"]) == 32
    a = client.get("/api/settings/defaults").headers["X-Request-ID"]
    b = client.get("/api/settings/defaults").headers["X-Request-ID"]
    assert a != b


def test_request_log_line_and_model_timing(json_logs):
    r = client.post("/api/deal/run", json={}, headers={"X-Request-ID": "deal-run-0001"})
    assert r.status_code == 200 and r.json()["returns"]["irr"] == pytest.approx(0.2116, abs=5e-5)
    logs = [e for e in json_logs() if e.get("request_id") == "deal-run-0001"]
    model = next(e for e in logs if e["event"] == "model_run")
    request = next(e for e in logs if e["event"] == "request")
    assert model["model"] == "deal.run" and model["ok"] is True and model["duration_ms"] > 0
    assert request["route"] == "/api/deal/run" and request["method"] == "POST" and request["status"] == 200
    assert request["model_ms"] == pytest.approx(model["duration_ms"])
    assert request["duration_ms"] >= request["model_ms"]
    assert request["ts"].endswith("Z")
    datetime.fromisoformat(request["ts"].replace("Z", "+00:00"))
    assert r.headers["Server-Timing"].startswith("model;dur=")
    # Bodies never logged
    assert str(SECRET_EBITDA) not in json.dumps(logs)


def test_logs_use_route_templates_not_values(json_logs):
    # An invalid ticker answers 422 without calling SEC EDGAR
    assert client.get("/api/edgar/SECRET!CO?q=private").status_code == 422
    text = json.dumps(json_logs())
    assert "SECRETCO" not in text and "private" not in text
    assert any(e.get("route") == "/api/edgar/{ticker}" for e in json_logs())


def test_health_checks_are_not_logged_but_report_utc_time(json_logs):
    body = client.get("/api/health").json()
    assert not [e for e in json_logs() if e["event"] == "request"]
    t = datetime.fromisoformat(body["time"].replace("Z", "+00:00"))
    assert body["time"].endswith("Z") and abs((datetime.now(timezone.utc) - t).total_seconds()) < 60


def test_test_error_reaches_sentry_with_its_request_id(staging, sentry_events, json_logs):
    r = client.get("/api/debug/error", headers={"X-Request-ID": "drill-error-0001"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error", "request_id": "drill-error-0001"}
    assert r.headers["X-Request-ID"] == "drill-error-0001"
    [event] = sentry_events
    assert event["tags"]["request_id"] == "drill-error-0001"
    assert event["tags"]["route"] == "/api/debug/error"
    assert event["environment"] == "test" and event["release"] == "abc123"
    assert event["exception"]["values"][-1]["type"] == "RuntimeError"
    [log] = [e for e in json_logs() if e["event"] == "unhandled_error"]
    assert log["request_id"] == "drill-error-0001" and log["error"] == "RuntimeError"
    # Throttled, so a public URL can't burn the Sentry allowance
    assert client.get("/api/debug/error").status_code == 429


def test_test_error_is_off_in_production(monkeypatch, sentry_events):
    monkeypatch.setenv("FSE_ENV", "production")
    monkeypatch.setattr(api_main, "_last_test_error", [0.0])
    assert client.get("/api/debug/error").status_code == 404
    assert sentry_events == []


def test_crash_event_carries_no_deal_contents(sentry_events, monkeypatch):
    import api.routers.deal as deal_router

    def broken(inputs, cfg):
        local_copy = inputs  # noqa: F841 - a local variable Sentry must not send
        raise ZeroDivisionError("model broke")
    monkeypatch.setattr(deal_router, "run_deal", broken)
    r = client.post("/api/deal/run?deal=private", json=DEAL_BODY,
                    headers={"X-Request-ID": "crash-0001", "Cookie": "session=secret",
                             "Authorization": "Bearer secret"})
    assert r.status_code == 500 and r.json()["request_id"] == "crash-0001"
    [event] = sentry_events
    assert event["tags"]["request_id"] == "crash-0001"
    assert event["tags"]["route"] == "/api/deal/run"
    # Search everything except the source lines around each frame (this file's
    # own code, which spells out the values it checks for)
    for exc in event["exception"]["values"]:
        for frame in exc["stacktrace"]["frames"]:
            for key in ("pre_context", "context_line", "post_context"):
                frame.pop(key, None)
    text = json.dumps(event)
    for leaked in (str(SECRET_EBITDA), "private", "secret", "local_copy", "testclient"):
        assert leaked not in text, leaked
    assert "user" not in event and set(event.get("request", {})) <= {"method", "url"}


def test_scrub_event_removes_sensitive_fields():
    event = {
        "request": {"method": "POST", "url": "https://x/api/deal/run?name=Acme", "data": {"ebitda": 1},
                    "headers": {"Cookie": "a"}, "cookies": {"a": "b"}, "query_string": "name=Acme",
                    "env": {"REMOTE_ADDR": "1.2.3.4"}},
        "user": {"ip_address": "1.2.3.4"}, "extra": {"sys.argv": []}, "breadcrumbs": {"values": [1]},
        "server_name": "host",
        "exception": {"values": [{"stacktrace": {"frames": [{"function": "f", "vars": {"deal": 1}}]}}]},
    }
    out = observability.scrub_event(event)
    assert out["request"] == {"method": "POST", "url": "https://x/api/deal/run"}
    assert not {"user", "extra", "breadcrumbs", "server_name"} & set(out)
    assert out["exception"]["values"][0]["stacktrace"]["frames"][0] == {"function": "f"}


def test_sentry_stays_off_without_dsn():
    assert observability.init_sentry(None, environment="x", release=None) is False
    assert observability.init_sentry("", environment="x", release=None) is False
