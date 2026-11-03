"""Result caching (PLAN.md 8.1): api/result_cache.py, db/result_cache.py.

The conftest turns the cache off for every other test; the ``fresh_cache``
fixture here turns on a new one in memory, with the stores a test gives it.
"""
import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from api import result_cache as rc
from api import usage
from api.auth import AuthUser, require_user
from api.main import app
from api.schemas import MonteCarloRequest
from core import model_version
from jobs.runner import Runner

client = TestClient(app)
MC = {"mc": {"n": 4000}, "seed": 11}
SCENARIOS = {"mc": {"n": 2000}, "seed": 5}
NOW = 1_700_000_000.0           # 2023-11-14


@pytest.fixture
def fresh_cache():
    cache = rc.ResultCache([])
    rc.set_cache(cache)
    return cache


def as_user(subject: str) -> None:
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)


def run(body=MC, path="/api/montecarlo/run"):
    t0 = time.perf_counter()
    resp = client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json(), (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path, body", [("/api/montecarlo/run", MC), ("/api/montecarlo/scenarios", SCENARIOS)])
def test_a_repeated_seeded_run_answers_identically_in_under_100_ms(fresh_cache, path, body):
    first, _ = run(body, path)
    assert fresh_cache.misses == 1 and fresh_cache.hits == 0
    second, ms = run(body, path)
    assert fresh_cache.hits == 1
    assert second == first                       # elapsed_ms too: the very answer
    assert ms < 100, f"cached answer took {ms:.1f} ms"


def test_the_cached_answer_is_what_the_simulation_gives(fresh_cache):
    cached, _ = run()
    rc.set_cache(rc.ResultCache([], enabled=False))
    fresh, _ = run()
    strip = lambda a: {k: v for k, v in a.items() if k != "elapsed_ms"}  # noqa: E731
    assert strip(cached) == strip(fresh)


def test_an_unseeded_run_is_never_cached(fresh_cache):
    body = {"mc": {"n": 2000}}
    run(body)
    run(body)
    assert fresh_cache.hits == 0 and fresh_cache.misses == 0 and fresh_cache.memory._size == 0


def test_a_new_engine_version_misses(fresh_cache, monkeypatch):
    run()
    monkeypatch.setattr(model_version, "ENGINE_VERSION", "99.0.0")
    answer, _ = run()
    assert fresh_cache.hits == 0 and fresh_cache.misses == 2
    assert answer["model"]["engine_version"] == "99.0.0"
    run()
    assert fresh_cache.hits == 1                 # and the new version's answer is kept in turn


def test_new_data_a_new_release_or_other_settings_miss(fresh_cache, monkeypatch):
    run()
    monkeypatch.setattr(model_version, "data_sets", lambda: {"tax_presets": "2099-01"})
    run()
    monkeypatch.setattr(model_version, "commit", lambda: "f" * 40)
    run()
    run({**MC, "settings": {"mc_clip_irr": 0.9}})
    run({**MC, "seed": 12})
    assert fresh_cache.hits == 0 and fresh_cache.misses == 5


def test_another_account_never_gets_the_answer(fresh_cache):
    as_user("user_a")
    run()
    as_user("user_b")
    run()
    assert fresh_cache.hits == 0 and fresh_cache.misses == 2
    request = MonteCarloRequest.model_validate(MC)
    a = fresh_cache.entry("montecarlo.run", "user_a", request)
    b = fresh_cache.entry("montecarlo.run", "user_b", request)
    assert a.key != b.key
    with pytest.raises(rc.CacheError):           # b's secret can't open a's value
        b.open(fresh_cache.memory.get(a.key))


def test_settings_the_run_refuses_are_not_cached(fresh_cache):
    request = MonteCarloRequest.model_validate({**MC, "settings": {"mc_corr_growth_exit": 5.0}})
    assert fresh_cache.entry("montecarlo.run", "user_a", request) is None


def test_system_callers_and_python_callers_skip_the_cache(fresh_cache):
    request = MonteCarloRequest.model_validate(MC)
    assert fresh_cache.entry("montecarlo.run", "system:drill", request) is None
    from api.routers import montecarlo
    montecarlo.post_run(request)                 # called from Python: no caller, no cache
    assert fresh_cache.hits == 0 and fresh_cache.misses == 0


def test_nothing_readable_is_stored(fresh_cache):
    body = {**MC, "deal": {"ebitda": 123.456789}}
    run(body)
    [(key, (_, blob))] = fresh_cache.memory._items.items()
    assert len(key) == 64 and int(key, 16) >= 0
    assert b"123.456789" not in blob and b"irr" not in blob
    assert "test" not in key


def test_a_job_is_answered_from_the_cache_without_the_slot(fresh_cache, job_queue):
    from api import limits
    runner = Runner(lambda: job_queue)
    first = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": MC}).json()
    assert runner.run_next() == "succeeded"
    direct, _ = run()                            # the endpoint shares the job's entry
    assert fresh_cache.hits == 1
    second = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": MC}).json()
    with limits.holding_simulation_slot(timeout=0) as got:   # the slot is busy elsewhere
        assert got
        assert runner.run_next() == "succeeded"
    assert fresh_cache.hits == 2
    one = client.get(f"/api/jobs/{first['id']}").json()["result"]
    two = client.get(f"/api/jobs/{second['id']}").json()["result"]
    assert one == two == direct


# ---------------------------------------------------------------------------
# The memory layer
# ---------------------------------------------------------------------------
def test_the_memory_layer_keeps_the_newest_within_its_budget():
    clock = [NOW]
    memory = rc.MemoryStore(budget_bytes=10, clock=lambda: clock[0])
    memory.put("a", b"1234", 60)
    memory.put("b", b"1234", 60)
    memory.get("a")                              # a is now the newer
    memory.put("c", b"1234", 60)
    assert memory.get("b") is None and memory.get("a") and memory.get("c")
    clock[0] += 61
    assert memory.get("a") is None and memory._size == 4


# ---------------------------------------------------------------------------
# Upstash and its budget
# ---------------------------------------------------------------------------
class FakeUpstash:
    """Upstash's REST API: GET, SET with EX, and pipelines of GETs."""

    def __init__(self, fail=False):
        self.data, self.calls, self.fail = {}, [], fail

    def _one(self, command):
        if command[0] == "GET":
            return self.data.get(command[1])
        assert command[0] == "SET" and command[3] == "EX" and int(command[4]) == rc.TTL_S
        self.data[command[1]] = command[2]
        return "OK"

    def post(self, url, json, timeout, headers):  # noqa: A002, ARG002
        assert headers["Authorization"] == "Bearer secret-token"
        self.calls.append((url, json))
        if self.fail:
            raise rc.requests.ConnectionError()

        class Resp:
            status_code = 200

            def __init__(self, body):
                self._body = body

            def json(self):
                return self._body
        if url.endswith("/pipeline"):
            return Resp([{"result": self._one(c)} for c in json])
        return Resp({"result": self._one(json)})


def upstash(http, counters, *, commands=100, size=10_000_000):
    return rc.UpstashCache("https://example.upstash.io", "secret-token", prefix="fse:staging:",
                           daily_commands=commands, daily_bytes=size, counters=counters,
                           session=http, clock=lambda: NOW)


def test_upstash_holds_an_opaque_entry_and_counts_its_spend():
    http = FakeUpstash()
    counters = usage.UsageCounters([], clock=lambda: NOW, background=False)
    redis = upstash(http, counters)
    cache = rc.ResultCache([redis], memory=rc.MemoryStore(budget_bytes=0))
    request = MonteCarloRequest.model_validate(MC)
    entry = cache.entry("montecarlo.run", "user_a", request)
    cache.put(entry, {"answer": [1.5, 2.5]})
    assert cache.get(entry) == {"answer": [1.5, 2.5]}
    stored_key = f"fse:staging:result:{entry.key}"
    assert list(http.data) == [stored_key]
    blob = base64.b64decode(http.data[stored_key])
    assert b"answer" not in blob
    # Today's counts read once (two commands), then one SET and one GET
    assert [c[1][0] if c[0].endswith("/pipeline") else c[1][0] for c in http.calls] == [["GET", "fse:staging:cache-commands:2023-11-14"], "SET", "GET"]
    spent = redis.spent()
    assert spent["commands"] == 4
    assert spent["bytes"] == 2 * len(http.data[stored_key])


def test_a_restart_starts_from_the_days_shared_count():
    http = FakeUpstash()
    http.data["fse:staging:cache-commands:2023-11-14"] = "95"
    counters = usage.UsageCounters([], clock=lambda: NOW, background=False)
    redis = upstash(http, counters, commands=100)
    cache = rc.ResultCache([redis], memory=rc.MemoryStore(budget_bytes=0))
    entry = cache.entry("montecarlo.run", "user_a", MonteCarloRequest.model_validate(MC))
    cache.get(entry)
    assert redis.spent()["commands"] == 98       # 95 + the read + the GET
    cache.put(entry, {"x": 1})
    cache.put(entry, {"x": 1})                   # 99, then 100: the budget
    assert redis.over_budget()
    calls = len(http.calls)
    cache.get(entry)
    assert len(http.calls) == calls              # no more Upstash today


def test_the_budget_switches_to_the_database_and_holds():
    http = FakeUpstash()
    counters = usage.UsageCounters([], clock=lambda: NOW, background=False)
    redis = upstash(http, counters, commands=6)

    class Table:
        name = "database"

        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def put(self, key, value, ttl):  # noqa: ARG002
            self.data[key] = value

    table = Table()
    cache = rc.ResultCache([redis, table], memory=rc.MemoryStore(budget_bytes=0))
    request = MonteCarloRequest.model_validate(MC)
    for seed in range(20):
        entry = cache.entry("montecarlo.run", "user_a", request.model_copy(update={"seed": seed}))
        assert cache.get(entry) is None
        cache.put(entry, {"seed": seed})
        assert cache.get(entry) == {"seed": seed}
    assert redis.spent()["commands"] <= 6
    assert len(table.data) >= 17
    # The spend reaches the shared counters, as the usage counters' own do
    assert counters.usage("cache-commands:2023-11-14") == redis.spent()["commands"]


def test_a_failing_upstash_never_fails_a_run(fresh_cache):
    http = FakeUpstash(fail=True)
    counters = usage.UsageCounters([], clock=lambda: NOW, background=False)
    cache = rc.ResultCache([upstash(http, counters)])
    rc.set_cache(cache)
    first, _ = run()
    second, _ = run()
    assert first == second and cache.hits == 1   # from this process's memory
    assert len(http.calls) == 1                  # tried once, then left alone for a while


def test_the_daily_ceilings_stay_under_the_free_plan():
    report = usage.budget_report()
    assert report["budget_ceiling_monthly_commands"] < usage.REDIS_MONTHLY_COMMANDS
    assert report["cache_ceiling_monthly_commands"] > 0
    daily_bytes = sum(rc.DAILY_BYTE_BUDGET.values()) + rc.OTHER_DAILY_BYTE_BUDGET
    assert 31 * daily_bytes < 10_000_000_000      # Upstash free: 10 GB of traffic a month
    assert daily_bytes < 256_000_000              # and 256 MB stored: a day's writes at most
    assert rc.TTL_S <= 24 * 3600


def test_the_public_health_check_says_where_and_nothing_about_use(fresh_cache):
    run()
    run()
    body = client.get("/api/health/limits").json()["result_cache"]
    assert body == {"enabled": True, "store": "memory"}
    assert fresh_cache.spent_today() is None     # no Upstash here


def test_the_hashes_are_keyed_with_the_servers_secret(monkeypatch):
    request = MonteCarloRequest.model_validate(MC)
    stamp = model_version.stamp({"a": 1})
    monkeypatch.setenv("FSE_CACHE_KEY", "one")
    one = rc.entry_for("montecarlo.run", "user_a", request, stamp)
    monkeypatch.setenv("FSE_CACHE_KEY", "two")
    two = rc.entry_for("montecarlo.run", "user_a", request, stamp)
    assert one.key != two.key                    # a copy of a store can't be tested by guessing
    with pytest.raises(rc.CacheError):
        two.open(one.seal({"a": 1}))
    monkeypatch.delenv("FSE_CACHE_KEY")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t1")
    three = rc.entry_for("montecarlo.run", "user_a", request, stamp)
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t2")
    assert rc.entry_for("montecarlo.run", "user_a", request, stamp).key != three.key


def test_without_a_known_commit_an_entry_lasts_one_process(monkeypatch):
    request = MonteCarloRequest.model_validate(MC)
    stamp = {**model_version.stamp({"a": 1}), "commit": None}
    here = rc.entry_for("montecarlo.run", "user_a", request, stamp).key
    monkeypatch.setattr(rc, "_PROCESS_BUILD", "another process")
    assert rc.entry_for("montecarlo.run", "user_a", request, stamp).key != here
    released = {**stamp, "commit": "a" * 40}
    assert rc.entry_for("montecarlo.run", "user_a", request, released).key == \
        rc.entry_for("montecarlo.run", "user_a", request, released).key


def test_an_answer_without_its_explanations_is_not_kept(fresh_cache, monkeypatch):
    from api import limits
    monkeypatch.setattr(limits, "EXPLAIN_RUN_LIMIT_S", 0.0)   # too slow to explain
    first, _ = run()
    assert first["explanations"] is None
    monkeypatch.setattr(limits, "EXPLAIN_RUN_LIMIT_S", 20.0)
    second, _ = run()
    assert second["explanations"] is not None and fresh_cache.hits == 0
    third, _ = run()
    assert third == second and fresh_cache.hits == 1


# ---------------------------------------------------------------------------
# The database fallback
# ---------------------------------------------------------------------------
def test_the_table_keeps_entries_expires_them_and_holds_its_budget(fresh_db):  # noqa: ARG001
    from datetime import timedelta

    from db import result_cache as table
    from db.models import utc_now

    now = utc_now()
    table.put("a" * 64, b"x" * 100, 60, budget_bytes=250, now=now)
    assert table.get("a" * 64, now=now) == b"x" * 100
    assert table.get("a" * 64, now=now + timedelta(seconds=61)) is None
    table.put("b" * 64, b"y" * 100, 60, budget_bytes=250, now=now + timedelta(seconds=1))
    table.put("c" * 64, b"z" * 100, 60, budget_bytes=250, now=now + timedelta(seconds=2))
    assert table.get("a" * 64, now=now) is None          # the oldest, over the budget
    assert table.usage() == {"rows": 2, "bytes": 200}
    table.put("d" * 64, b"w", 60, budget_bytes=250, now=now + timedelta(seconds=120))
    assert table.usage() == {"rows": 1, "bytes": 1}      # the expired ones went with it


def test_a_repeated_run_is_answered_from_the_table(fresh_db):  # noqa: ARG001
    cache = rc.ResultCache([rc.DatabaseCache()], memory=rc.MemoryStore(budget_bytes=0))
    rc.set_cache(cache)
    first, _ = run()
    second, _ = run()
    assert first == second and cache.hits == 1 and cache.writes == 1
    from db import result_cache as table
    assert table.usage()["rows"] == 1


def test_the_material_never_holds_more_than_it_needs():
    request = MonteCarloRequest.model_validate(MC)
    entry = rc.entry_for("montecarlo.run", "user_a", request, model_version.stamp({"a": 1}))
    other = rc.entry_for("montecarlo.run", "user_a", request,
                         {**model_version.stamp({"a": 1}), "data_sets": {"x": "y"}})
    assert entry.key == other.key                # data_sets is covered by its fingerprint
    assert json.dumps(entry.seal({"a": 1}).hex()) != entry.key
