"""Better Stack monitor plan (ops/betterstack.py): fits the free limits,
ignores Render wake-ups, and syncs idempotently."""
import copy

import pytest

from ops import betterstack as bs


class FakeBetterStack:
    """In-memory Better Stack API recording every write."""

    def __init__(self):
        self.monitors, self.pages, self.resources, self.writes, self.incidents = {}, {}, {}, [], []
        self.next_id = 100

    def _id(self):
        self.next_id += 1
        return str(self.next_id)

    def __call__(self, method, url, body=None):
        path = url.removeprefix(bs.API)
        if method != "GET":
            self.writes.append((method, path, copy.deepcopy(body)))
        if path == "/v2/monitors" and method == "GET":
            return {"data": [{"id": i, "attributes": a} for i, a in self.monitors.items()],
                    "pagination": {"next": None}}
        if path == "/v2/monitors" and method == "POST":
            i = self._id()
            self.monitors[i] = {**body, "status": "pending"}
            return {"data": {"id": i}}
        if path.startswith("/v2/monitors/") and method == "PATCH":
            self.monitors[path.rsplit("/", 1)[1]].update(body)
            return {"data": {}}
        if path == "/v2/status-pages" and method == "GET":
            return {"data": [{"id": i, "attributes": a} for i, a in self.pages.items()]}
        if path == "/v2/status-pages" and method == "POST":
            i = self._id()
            self.pages[i] = dict(body)
            return {"data": {"id": i, "attributes": body}}
        if path.endswith("/resources"):
            page = path.split("/")[3]
            if method == "GET":
                return {"data": [{"id": n, "attributes": r} for n, r in enumerate(self.resources.get(page, []))]}
            self.resources.setdefault(page, []).append(dict(body))
            return {"data": {}}
        if path == "/v3/incidents" and method == "POST":
            self.incidents.append(body)
            return {"data": {"id": "inc-1"}}
        raise AssertionError(f"unexpected {method} {path}")


def test_monitor_plan_fits_free_limits():
    assert bs.check_plan() == []
    render = [m for m in bs.MONITORS if m["_host"] == "render-free"]
    assert render, "the production API must be monitored"
    hours = sum(bs.render_awake_hours(m["check_frequency"]) for m in render)
    assert hours + bs.RENDER_HOURS_RESERVED <= bs.RENDER_FREE_HOURS
    # Every monitor alerts by email
    assert all(m["email"] for m in bs.MONITORS)


def test_web_monitor_checks_a_route_that_works_signed_out():
    """Since sign-in (PLAN.md 1.4) signed-out pages answer 404 or redirect, so
    the web monitor must use the health route the proxy leaves open, and look
    for text that route really returns (web/e2e/auth.spec.ts requests it)."""
    from pathlib import Path

    web = Path(__file__).resolve().parents[1] / "web"
    [monitor] = [m for m in bs.MONITORS if m["_host"] == "vercel"]
    assert monitor["url"] == "https://fse-ml.vercel.app/healthz"
    route = (web / "src/app/healthz/route.ts").read_text(encoding="utf-8")
    assert monitor["required_keyword"] == '"service":"FSE/ML web"'
    assert 'service: "FSE/ML web"' in route
    proxy = (web / "src/proxy.ts").read_text(encoding="utf-8")
    assert "(?!api|healthz|" in proxy, "the proxy must not send /healthz to sign-in"
    assert monitor["required_keyword"] in (web / "e2e/auth.spec.ts").read_text(encoding="utf-8")


def test_awake_hours_model():
    assert bs.render_awake_hours(180) == bs.MONTH_HOURS          # never gets to sleep
    assert bs.render_awake_hours(960) == bs.MONTH_HOURS
    assert bs.render_awake_hours(1800) == pytest.approx(744 * 960 / 1800)


@pytest.mark.parametrize("change, message", [
    ({"check_frequency": 300}, "exceeds the free"),              # keeps Render awake all month
    ({"confirmation_period": 60, "request_timeout": 30}, "normal wake-up"),
    ({"check_frequency": 120}, "every 3 minutes"),
])
def test_plan_check_catches_bad_settings(change, message):
    monitors = copy.deepcopy(bs.MONITORS)
    target = next(m for m in monitors if m["_host"] == "render-free")
    target.update(change)
    assert any(message in p for p in bs.check_plan(monitors))


def test_sync_creates_then_is_idempotent():
    fake = FakeBetterStack()
    ids = bs.sync(fake, log=lambda _: None)
    assert set(ids) == {m["pronounceable_name"] for m in bs.MONITORS}
    api = next(a for a in fake.monitors.values() if "onrender.com" in a["url"])
    assert api["check_frequency"] == 1800 and api["required_keyword"] == '"status":"ok"'
    assert not any(k.startswith("_") for a in fake.monitors.values() for k in a)
    [page] = fake.pages.values()
    assert page["subdomain"] == bs.STATUS_PAGE["subdomain"]
    [resources] = fake.resources.values()
    assert sorted(r["public_name"] for r in resources) == ["Model API", "Website"]

    writes = len(fake.writes)
    bs.sync(fake, log=lambda _: None)
    assert len(fake.writes) == writes, fake.writes[writes:]


def test_sync_updates_only_drifted_fields():
    fake = FakeBetterStack()
    bs.sync(fake, log=lambda _: None)
    mid, api = next((i, a) for i, a in fake.monitors.items() if "onrender.com" in a["url"])
    api["check_frequency"] = 180                                   # someone changed it by hand
    fake.writes.clear()
    bs.sync(fake, log=lambda _: None)
    assert fake.writes == [("PATCH", f"/v2/monitors/{mid}", {"check_frequency": 1800})]


def test_dry_run_writes_nothing():
    fake = FakeBetterStack()
    bs.sync(fake, dry_run=True, log=lambda _: None)
    assert fake.writes == []


def test_incident_alerts_by_email():
    fake = FakeBetterStack()
    assert bs.raise_incident(fake, "Staging check failed", "run 1") == "inc-1"
    [body] = fake.incidents
    assert body["email"] is True and body["summary"] == "run 1" and body["requester_email"]


def test_cli_needs_token(monkeypatch, capsys):
    monkeypatch.delenv("BETTERSTACK_API_TOKEN", raising=False)
    assert bs.main(["report"]) == 1
    assert "BETTERSTACK_API_TOKEN" in capsys.readouterr().out
    assert bs.main(["check"]) == 0
