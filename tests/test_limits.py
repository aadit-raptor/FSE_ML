"""Usage limits and abuse protection (PLAN.md 1.6): api/limits.py, api/usage.py, db/usage.py.

Limits are made small here with monkeypatch, so a test reaches them in a few
requests; the conftest fixture gives every test fresh counters in memory.
"""
import os
import threading
import time
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api import limits, usage
from api.auth import AuthUser, require_user
from api.main import app
from api.routers import montecarlo
from api.schemas import BacktestRequest, ForecastRunRequest, MonteCarloRequest
from db import engine as db_engine
from db.models import UsageCounter, utc_now

client = TestClient(app)


def small(rule: limits.Rule, limit: int) -> limits.Rule:
    return limits.Rule(rule.name, rule.what, rule.per, limit, rule.window_s, rule.shared)


def as_user(subject: str):
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)


# ---------------------------------------------------------------------------
# Per user
# ---------------------------------------------------------------------------
def test_user_gets_429_after_the_request_limit(monkeypatch):
    monkeypatch.setattr(limits, "USER_REQUESTS", small(limits.USER_REQUESTS, 3))
    as_user("user_a")
    for _ in range(3):
        assert client.get("/api/settings/defaults").status_code == 200
    resp = client.get("/api/settings/defaults")
    assert resp.status_code == 429
    body = resp.json()
    assert body["detail"].startswith("Too many requests: your account can make 3 a minute. Try again in ")
    assert body["limit"]["rule"] == "user_requests_per_minute"
    assert body["limit"]["limit"] == 3
    retry = int(resp.headers["Retry-After"])
    assert 1 <= retry <= 60 and body["limit"]["retry_after_s"] == retry
    # The request ID and CORS still come back on a refusal, so the browser can read it
    assert resp.headers["X-Request-ID"]
    # Another account is not affected
    as_user("user_b")
    assert client.get("/api/settings/defaults").status_code == 200


def test_runs_have_their_own_limit(monkeypatch):
    monkeypatch.setattr(limits, "USER_RUNS", small(limits.USER_RUNS, 2))
    as_user("runner")
    body = {"history": {}, "assumptions": {}, "simulate": False}
    # Refused by validation (422) but still a run request, and still counted
    for _ in range(2):
        assert client.post("/api/forecasting/run", json=body).status_code == 422
    resp = client.post("/api/forecasting/run", json=body)
    assert resp.status_code == 429
    assert "Too many model runs" in resp.json()["detail"]
    assert client.get("/api/edgar/AAPL").status_code == 429       # a data lookup is a run too
    # Ordinary requests still work
    assert client.get("/api/settings/defaults").status_code == 200


def test_daily_run_limit_says_hours():
    rule = limits.USER_RUNS_DAY
    now = 1_700_000_000.0
    check = limits.check_for(rule, "someone", now)
    body, headers = limits.refusal(rule, check, now)
    assert "1,000 a day" in body["detail"]
    assert body["detail"].endswith("hours.")
    assert int(headers["Retry-After"]) == int(check.window_end - now)
    assert check.window_end % 86_400 == 0          # windows end at UTC midnight


def test_multiplier_raises_limits_but_not_in_production(monkeypatch):
    monkeypatch.setenv("FSE_LIMITS_MULTIPLIER", "10")
    monkeypatch.setenv("FSE_ENV", "staging")
    assert limits.effective_limit(limits.USER_RUNS) == 300
    monkeypatch.setenv("FSE_ENV", "production")
    assert limits.effective_limit(limits.USER_RUNS) == 30


# ---------------------------------------------------------------------------
# Per address
# ---------------------------------------------------------------------------
def test_refused_sign_ins_block_the_address(monkeypatch, signed_out):  # noqa: ARG001
    monkeypatch.setattr(limits, "ADDRESS_REFUSED_SIGN_INS", small(limits.ADDRESS_REFUSED_SIGN_INS, 3))
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    for _ in range(3):
        assert client.post("/api/deal/run", json={}, headers={"Authorization": "Bearer guess"}).status_code == 401
    resp = client.post("/api/deal/run", json={}, headers={"Authorization": "Bearer dev:real"})
    assert resp.status_code == 429
    assert resp.json()["detail"].startswith("Too many refused sign-ins: this network address can make 3 a minute.")
    # Health checks are never limited
    assert client.get("/api/health").status_code == 200


def test_signed_in_requests_do_not_count_as_refused(monkeypatch, signed_out):  # noqa: ARG001
    monkeypatch.setattr(limits, "ADDRESS_REFUSED_SIGN_INS", small(limits.ADDRESS_REFUSED_SIGN_INS, 2))
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    for _ in range(5):
        assert client.get("/api/settings/defaults", headers={"Authorization": "Bearer dev:ok"}).status_code == 200


def test_address_request_limit(monkeypatch):
    monkeypatch.setattr(limits, "ADDRESS_REQUESTS", small(limits.ADDRESS_REQUESTS, 4))
    codes = [client.get("/api/settings/defaults").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 200, 429]
    assert client.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# Simulation size, request size, timeout, one run at a time
# ---------------------------------------------------------------------------
def test_oversized_simulation_is_refused_clearly():
    resp = client.post("/api/montecarlo/run", json={"mc": {"n": 1_000_000}})
    assert resp.status_code == 422
    [error] = resp.json()["detail"]
    assert error["loc"][-1] == "n"
    assert error["msg"] == ("This server runs at most 100,000 paths in one simulation; this one asks "
                            "for 1,000,000. Use 100,000 or fewer.")
    backtest = client.post("/api/backtesting/run", json={"n": 150_000})
    assert any("at most 100,000 paths" in e["msg"] for e in backtest.json()["detail"])


def test_path_caps_sit_exactly_at_the_limits():
    assert MonteCarloRequest(mc={"n": limits.MAX_SIMULATION_PATHS}).mc.n == 100_000
    with pytest.raises(ValueError):
        MonteCarloRequest(mc={"n": limits.MAX_SIMULATION_PATHS + 1})
    with pytest.raises(ValueError):
        BacktestRequest.model_validate({"entry": {}, "actual": {}, "actual_exit": {}, "n": 100_001})
    ok = ForecastRunRequest(history={}, assumptions={}, n_sim=200_000)
    assert ok.n_sim == 200_000
    with pytest.raises(ValueError):
        ForecastRunRequest(history={}, assumptions={}, n_sim=200_001)
    schema = client.get("/api/openapi.json").json()["components"]["schemas"]
    assert schema["MCInputsIn"]["properties"]["n"]["maximum"] == 100_000


def test_capped_simulation_runs():
    """The largest allowed run still answers (at the default hold)."""
    resp = client.post("/api/montecarlo/run", json={"mc": {"n": limits.MAX_SIMULATION_PATHS}, "seed": 7})
    assert resp.status_code == 200
    assert resp.json()["n"] == 100_000


def test_request_body_over_the_limit_gets_413(monkeypatch):
    monkeypatch.setattr(limits, "MAX_BODY_BYTES", 2_000)
    big = {"filename": "x", "sheets": [{"name": "s", "columns": ["a"], "rows": [[1.0]] * 1000}]}
    resp = client.post("/api/export/workbook", json=big)
    assert resp.status_code == 413
    assert resp.json()["detail"].startswith("The request is too large: the limit is 2 KB (this one is ")

    # Without a Content-Length (chunked), the body is measured as it arrives
    def chunks():
        yield b'{"filename": "x", "sheets": ['
        for _ in range(200):
            yield b'{"name": "s", "columns": ["a"], "rows": [[1]]},'
        yield b'{"name": "s", "columns": ["a"], "rows": [[1]]}]}'
    resp = client.post("/api/export/workbook", content=chunks(), headers={"Content-Type": "application/json"})
    assert resp.status_code == 413

    small_body = {"filename": "x", "sheets": [{"name": "s", "columns": ["a"], "rows": [[1.0]]}]}
    assert client.post("/api/export/workbook", json=small_body).status_code == 200


def test_slow_simulation_gets_504_and_keeps_its_slot(monkeypatch):
    monkeypatch.setattr(limits, "RUN_TIMEOUT_S", 0.3)
    finished = threading.Event()
    real = montecarlo.run_vectorized_simulation_full

    def slow(*args, **kwargs):
        time.sleep(1.0)
        try:
            return real(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(montecarlo, "run_vectorized_simulation_full", slow)
    t0 = time.perf_counter()
    resp = client.post("/api/montecarlo/run", json={"mc": {"n": 1000}, "seed": 1})
    assert resp.status_code == 504
    assert resp.json()["detail"].startswith("The run took longer than 0.3 seconds")
    assert time.perf_counter() - t0 < 0.9
    # The run carries on in the background, holding the one simulation slot
    assert not limits._slots.acquire(blocking=False)
    assert finished.wait(10)
    for _ in range(50):
        if limits._slots.acquire(blocking=False):
            limits._slots.release()
            break
        time.sleep(0.05)
    else:
        pytest.fail("the slot was never released")


def test_busy_server_refuses_a_second_simulation(monkeypatch):
    monkeypatch.setattr(limits, "SLOT_WAIT_S", 0.1)
    assert limits._slots.acquire(blocking=False)
    try:
        resp = client.post("/api/montecarlo/run", json={"mc": {"n": 1000}, "seed": 1})
    finally:
        limits._slots.release()
    assert resp.status_code == 503
    assert resp.json()["detail"] == "The server is busy with other simulations. Try again in a few seconds."
    assert resp.headers["Retry-After"] == "10"
    assert client.post("/api/montecarlo/run", json={"mc": {"n": 1000}, "seed": 1}).status_code == 200


# ---------------------------------------------------------------------------
# Counters and their sync
# ---------------------------------------------------------------------------
class FakeStore:
    """A shared store two API processes can both use."""

    def __init__(self, name="upstash", fail=False):
        self.name, self.fail, self.data, self.calls = name, fail, {}, []

    def add(self, batch):
        self.calls.append(list(batch))
        if self.fail:
            raise usage.StoreError("down")
        for key, delta, _ in batch:
            self.data[key] = self.data.get(key, 0) + delta
        return {key: self.data[key] for key, _, _ in batch}


def daily(identity="someone", now=1_700_000_000.0):
    return limits.check_for(small(limits.USER_RUNS_DAY, 5), identity, now)


def test_shared_counts_hold_across_processes_after_a_sync():
    store = FakeStore()
    clock = lambda: 1_700_000_000.0  # noqa: E731
    first = usage.UsageCounters([store], clock=clock, background=False)
    second = usage.UsageCounters([store], clock=clock, background=False)
    for _ in range(3):
        assert first.consume([daily()]) is None
    assert first.flush() == "upstash"
    # Only pending increments travel, in one batch; the key carries no user id
    assert store.calls == [[(daily().key, 3, store.calls[0][0][2])]]
    assert "someone" not in daily().key
    assert second.consume([daily()]) is None and second.consume([daily()]) is None
    second.flush()
    assert second.consume([daily()]) is not None          # 3 + 2 = 5: at the limit
    # And a flush with nothing new sends nothing
    calls = len(store.calls)
    assert first.flush() is None and len(store.calls) == calls


def test_per_minute_counts_never_leave_the_process():
    store = FakeStore()
    counters = usage.UsageCounters([store], background=False)
    counters.consume([limits.check_for(limits.USER_REQUESTS, "u", time.time())])
    assert counters.flush() is None and store.calls == []


def test_sync_falls_back_to_the_next_store_and_keeps_counts_when_all_fail():
    down, db = FakeStore("upstash", fail=True), FakeStore("database")
    counters = usage.UsageCounters([down, db], clock=lambda: 1_700_000_000.0, background=False)
    counters.consume([daily()])
    assert counters.flush() == "database"
    assert db.data[daily().key] == 1

    db.fail = True
    counters2 = usage.UsageCounters([FakeStore(fail=True), db], clock=lambda: 1_700_000_000.0,
                                    background=False)
    counters2.consume([daily()])
    assert counters2.flush() is None
    assert counters2.usage(daily().key) == 1                # still enforced from memory
    db.fail = False
    counters2._failed_until.clear()
    assert counters2.flush() == "database" and db.data[daily().key] == 2


class FakeUpstash:
    """Upstash's REST API, running the script's logic in Python."""

    def __init__(self):
        self.data, self.commands = {}, []

    def post(self, url, json, timeout, headers):  # noqa: A002, ARG002
        assert headers["Authorization"] == "Bearer secret-token"
        self.commands.append(json)

        class Resp:
            status_code = 200

            def __init__(self, result):
                self._result = result

            def json(self):
                return {"result": self._result}
        if json[0] == "PING":
            return Resp("PONG")
        assert json[0] == "EVAL" and json[1] == usage.REDIS_SCRIPT
        n = int(json[2])
        keys, args = json[3:3 + n], json[3 + n:]
        out = []
        for i, key in enumerate(keys[:-1]):
            self.data[key] = self.data.get(key, 0) + int(args[2 * i])
            out.append(self.data[key])
        self.data[keys[-1]] = self.data.get(keys[-1], 0) + int(args[-1])
        out.append(self.data[keys[-1]])
        return Resp(out)


def test_upstash_budget_guard_switches_to_the_database():
    http = FakeUpstash()
    now = 1_700_000_000.0
    redis = usage.UpstashStore("https://example.upstash.io", "secret-token", prefix="fse:staging:",
                               daily_budget=12, session=http, clock=lambda: now)
    db = FakeStore("database")
    counters = usage.UsageCounters([redis, db], clock=lambda: now, background=False)

    counters.consume([daily("a"), daily("b")])
    assert counters.flush() == "upstash"
    # One call for both keys, prefixed per environment, costing 1 + 2x2 + 2 = 7
    assert len(http.commands) == 1
    assert http.data[f"fse:staging:{daily('a').key}"] == 1
    assert http.data["fse:staging:redis-commands:2023-11-14"] == 7
    assert redis.commands_today == 7 and not redis.over_budget()

    counters.consume([daily("a")])
    assert counters.flush() == "upstash"                   # 7 + 5 = 12: budget reached
    assert redis.over_budget()
    counters.consume([daily("a")])
    assert counters.flush() == "database"
    assert len(http.commands) == 2                          # no more Redis calls today
    assert "upstash" not in counters._failed_until          # skipped quietly, not a failure
    assert counters.usage(daily("a").key) == 3


def test_monthly_redis_estimate_holds_at_target_traffic():
    report = usage.budget_report()
    assert report["estimated_monthly_commands"] < 0.5 * usage.REDIS_MONTHLY_COMMANDS
    # The daily budgets cap Redis use whatever the traffic
    assert report["budget_ceiling_monthly_commands"] < usage.REDIS_MONTHLY_COMMANDS
    # And the estimate counts the way the guard does
    production = usage.TARGET_TRAFFIC[0]
    syncs = -(-production.active_minutes_per_user * 60 // usage.FLUSH_INTERVAL_S)
    assert usage.monthly_redis_commands(production) == 31 * (
        2 * production.daily_active_users * syncs
        + 3 * -(-production.busy_hours_per_day * 3600 // usage.FLUSH_INTERVAL_S))


def test_health_limits_is_public_and_reports_the_store(monkeypatch, signed_out):  # noqa: ARG001
    usage.set_counters(usage.UsageCounters([], background=False))
    resp = client.get("/api/health/limits")
    assert resp.status_code == 200
    assert resp.json()["store"] == "memory"
    assert resp.json()["upstash"]["configured"] is False

    http = FakeUpstash()
    redis = usage.UpstashStore("https://example.upstash.io", "secret-token", prefix="fse:t:",
                               daily_budget=100, session=http)
    usage.set_counters(usage.UsageCounters([redis], background=False))
    body = client.get("/api/health/limits").json()
    assert body["store"] == "upstash" and body["upstash"]["reachable"] is True
    client.get("/api/health/limits")
    assert http.commands == [["PING"]]                      # reused, and only PING


def test_configured_stores_follow_the_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    assert usage.configured_stores("production") == []
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://x.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    [redis, db] = usage.configured_stores("production")
    assert redis.prefix == "fse:production:" and redis.daily_budget == 8_000
    assert db.name == "database"


# ---------------------------------------------------------------------------
# Real stores
# ---------------------------------------------------------------------------
def test_database_counters_add_up_and_expire(fresh_db):  # noqa: ARG001
    from db.usage import add_counts
    now = utc_now()
    assert add_counts([("k1", 2, 3600), ("k2", 1, 60)], now=now) == {"k1": 2, "k2": 1}
    assert add_counts([("k1", 3, 3600)], now=now) == {"k1": 5}
    # A later batch deletes counters whose window is over
    assert add_counts([("k3", 1, 60)], now=now + timedelta(seconds=120)) == {"k3": 1}
    with db_engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(UsageCounter)).scalar() == 2
    # And through the counters, as the fallback store
    counters = usage.UsageCounters([usage.DatabaseStore()], background=False)
    check = limits.check_for(limits.USER_RUNS_DAY, "db-user", time.time())
    counters.consume([check])
    assert counters.flush() == "database"


@pytest.mark.skipif(not (os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get("UPSTASH_REDIS_REST_TOKEN")),
                    reason="UPSTASH_REDIS_REST_URL/TOKEN not set (CI sets them)")
def test_real_upstash_round_trip():
    """The script against real Upstash: totals add up across two processes."""
    run = uuid.uuid4().hex[:8]
    make = lambda: usage.UpstashStore(  # noqa: E731
        os.environ["UPSTASH_REDIS_REST_URL"], os.environ["UPSTASH_REDIS_REST_TOKEN"],
        prefix=f"fse:ci:{run}:", daily_budget=usage.OTHER_DAILY_COMMAND_BUDGET)
    first, second = make(), make()
    assert first.ping()
    assert first.add([("runs", 2, 120)]) == {"runs": 2}
    assert second.add([("runs", 3, 120), ("other", 1, 120)]) == {"runs": 5, "other": 1}
    assert second.commands_today and second.commands_today >= usage.redis_commands_for_sync(2)
