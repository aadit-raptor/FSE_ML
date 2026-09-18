"""Background jobs and scheduled jobs (PLAN.md 1.9).

Real runs, real queues: every job here runs the actual model through the
actual runner, and results are compared with what the direct endpoint answers
for the same seed. The queue-level tests run against **both** queue
implementations -- in memory, and the ``jobs`` table in a real Postgres
(``fresh_db``) -- through the same endpoints, which is the "swapping the queue
needs no endpoint changes" test.

Scheduled endpoints are called with real RS256 tokens shaped like GitHub
Actions' OIDC tokens, signed by a key this test owns; only the fetch of
GitHub's key set is replaced.
"""
import base64
import json
import threading
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import api.github_oidc as oidc
from api.auth import AuthUser, require_user
from api.limits import JOBS_PATH, is_run_request
from api.main import app
from db import engine as db_engine
from db.models import Job, ScheduledRun
from jobs import config, drill
from jobs import queue as q
from jobs.database import DatabaseQueue
from jobs.memory import MemoryQueue
from jobs.runner import Runner

real_wake_runner = config.wake_runner   # before conftest switches it off

client = TestClient(app)

MC = {"mc": {"n": 4000}, "seed": 11, "histogram_bins": 20, "scatter_points": 50}
SCENARIOS = {"mc": {"n": 3000}, "seed": 5}


def forecast_request() -> dict:
    d = client.get("/api/forecasting/defaults").json()
    grid = {k: [v] * 3 for k, v in d["seeded_assumptions"].items()}
    return {"history": d["history"], "assumptions": grid, "simulate": True, "n_sim": 2000}


def backtest_request() -> dict:
    deal = client.get("/api/backtesting/deals").json()[0]
    hold = int(deal["entry"]["holding_period"])
    return {"entry": deal["entry"], "actual": {k: v[:hold] for k, v in deal["actual"].items()},
            "actual_exit": deal["actual_exit"], "n": 2000, "histogram_bins": 20}


def without_timing(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "elapsed_ms"}


def ok(resp, code=200):
    assert resp.status_code == code, resp.text
    return resp.json()


def as_user(subject: str) -> None:
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)


# ---------------------------------------------------------------------------
# One fixture, two queues: every test below using ``queue`` runs on both
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "database"])
def queue(request, job_queue):
    if request.param == "memory":
        return job_queue
    request.getfixturevalue("fresh_db")
    db_queue = DatabaseQueue()
    config.use_queue(db_queue)
    return db_queue


def runner_for(queue, **kw) -> Runner:
    return Runner(lambda: queue, **kw)


def submit(kind: str, body: dict, code=202) -> dict:
    return ok(client.post("/api/jobs", json={"kind": kind, "input": body}), code)


def status(job_id: str) -> dict:
    return ok(client.get(f"/api/jobs/{job_id}"))


# ---------------------------------------------------------------------------
# Same code, same results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, path, body", [
    ("montecarlo.run", "/api/montecarlo/run", MC),
    ("montecarlo.scenarios", "/api/montecarlo/scenarios", SCENARIOS),
    ("backtesting.run", "/api/backtesting/run", None),
    ("forecasting.run", "/api/forecasting/run", None),
])
def test_a_job_gives_exactly_what_the_endpoint_gives(queue, kind, path, body):
    body = body or (backtest_request() if kind == "backtesting.run" else forecast_request())
    direct = ok(client.post(path, json=body))
    job = submit(kind, body)
    assert job["status"] == "queued" and job["ahead"] == 0 and job["result"] is None
    assert runner_for(queue).run_next() == "succeeded"
    done = status(job["id"])
    assert done["status"] == "succeeded" and done["progress"] == 1.0 and done["attempts"] == 1
    assert without_timing(done["result"]) == without_timing(direct)


def test_seeded_monte_carlo_is_unchanged_as_a_job(queue):
    """The pinned drill answer is what the direct endpoint gives today, and
    what a job gives: the check staging runs is anchored to the real code."""
    direct = ok(client.post("/api/montecarlo/run", json=drill.DRILL_REQUEST))
    assert drill.summary_mismatches(direct["summary"]) == []
    job = submit("montecarlo.run", drill.DRILL_REQUEST)
    runner_for(queue).run_next()
    assert status(job["id"])["result"]["summary"] == direct["summary"]


def test_the_drill_check_notices_a_different_answer():
    changed = dict(drill.EXPECTED_SUMMARY, mean_irr=drill.EXPECTED_SUMMARY["mean_irr"] * (1 + 1e-6))
    assert drill.summary_mismatches(changed) == ["mean_irr"]
    assert drill.summary_mismatches({}) == list(drill.EXPECTED_SUMMARY)


def test_progress_follows_the_runs_stages(queue):
    seen = []
    real_report = queue.report

    def spy(job_id, worker, *, progress=None, stage=None):
        seen.append((progress, stage))
        return real_report(job_id, worker, progress=progress, stage=stage)

    queue.report = spy
    submit("montecarlo.run", MC)
    runner_for(queue).run_next()
    assert (None, "Simulating paths") in seen
    assert (0.8, "Summarising results") in seen


# ---------------------------------------------------------------------------
# Refused up front, like the endpoints
# ---------------------------------------------------------------------------
def test_bad_input_is_refused_at_submission(queue):
    too_big = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": {"mc": {"n": 5_000_000}}})
    assert too_big.status_code == 422
    bad_corr = {"settings": {"corr_g_em": 0.99, "corr_g_ir": 0.99, "corr_em_ir": -0.99}}
    r = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": bad_corr})
    assert r.status_code == 422 and "semi-definite" in r.text
    assert client.post("/api/jobs", json={"kind": "no.such", "input": {}}).status_code == 422
    assert queue.counts()["queued"] == 0


def test_a_run_that_fails_says_why(queue):
    body = forecast_request()
    body["history"] = {"h_rev": [100.0, 110.0], "h_cogs": [60.0]}
    job = submit("forecasting.run", body)
    assert runner_for(queue).run_next() == "failed"
    done = status(job["id"])
    assert done["error_status"] == 422 and "same number of years" in done["error"]


def test_one_account_can_only_queue_so_many(queue):
    for _ in range(q.MAX_ACTIVE_PER_OWNER):
        submit("montecarlo.run", MC)
    r = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": MC})
    assert r.status_code == 429 and "runs waiting or running" in r.json()["detail"]
    as_user("user_someone_else")
    submit("montecarlo.run", MC)    # a different account still can


def test_a_full_queue_refuses_new_jobs(queue, monkeypatch):
    monkeypatch.setattr(q, "MAX_QUEUED", 2)
    for name in ("user_a", "user_b"):
        as_user(name)
        submit("montecarlo.run", MC)
    as_user("user_c")
    r = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": MC})
    assert r.status_code == 503 and r.headers["Retry-After"]


def test_submitting_counts_as_a_run_and_polling_does_not():
    assert is_run_request("POST", JOBS_PATH)
    assert not is_run_request("GET", JOBS_PATH)
    assert not is_run_request("GET", f"{JOBS_PATH}/{uuid.uuid4()}")


# ---------------------------------------------------------------------------
# Whose job
# ---------------------------------------------------------------------------
def test_nobody_else_can_see_or_cancel_a_job(queue):
    as_user("user_owner")
    job = submit("montecarlo.run", MC)
    as_user("user_intruder")
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 404
    assert ok(client.get("/api/jobs"))["jobs"] == []
    as_user("user_owner")
    assert [j["id"] for j in ok(client.get("/api/jobs"))["jobs"]] == [job["id"]]


def test_the_list_leaves_results_out(queue):
    job = submit("montecarlo.run", MC)
    runner_for(queue).run_next()
    listed = ok(client.get("/api/jobs"))["jobs"]
    assert listed[0]["id"] == job["id"] and listed[0]["result"] is None
    assert status(job["id"])["result"] is not None


# ---------------------------------------------------------------------------
# Queue order, cancelling
# ---------------------------------------------------------------------------
def test_jobs_run_in_order_and_know_their_place(queue):
    ids = [submit("montecarlo.run", {**MC, "seed": s})["id"] for s in (1, 2, 3)]
    assert [status(i)["ahead"] for i in ids] == [0, 1, 2]
    runner = runner_for(queue)
    runner.run_next()
    assert status(ids[0])["status"] == "succeeded"
    assert [status(i)["ahead"] for i in ids[1:]] == [0, 1]


def test_cancelling_a_waiting_job_stops_it(queue):
    job = submit("montecarlo.run", MC)
    cancelled = ok(client.post(f"/api/jobs/{job['id']}/cancel"))
    assert cancelled["status"] == "cancelled"
    assert runner_for(queue).run_next() is None


def test_cancelling_a_running_job_drops_its_result(queue):
    job = submit("montecarlo.run", MC)
    runner = runner_for(queue)
    claimed = queue.claim(runner.worker)
    assert ok(client.post(f"/api/jobs/{job['id']}/cancel"))["cancel_requested"] is True
    assert runner.execute(queue, claimed) == "cancelled"
    done = status(job["id"])
    assert done["status"] == "cancelled" and done["result"] is None


# ---------------------------------------------------------------------------
# Surviving a restart
# ---------------------------------------------------------------------------
def test_a_job_whose_server_restarted_is_resumed_with_the_same_result(queue):
    direct = ok(client.post("/api/montecarlo/run", json=MC))
    job = submit("montecarlo.run", MC)
    lost = runner_for(queue)
    assert queue.claim(lost.worker) is not None       # ...and then the server went away
    queue.age_heartbeat(uuid.UUID(job["id"]), q.LEASE_S + 5)

    after_restart = runner_for(queue)
    after_restart.maintain(queue)
    assert status(job["id"])["status"] == "queued"
    assert after_restart.run_next() == "succeeded"
    done = status(job["id"])
    assert done["attempts"] == 2
    assert without_timing(done["result"]) == without_timing(direct)
    # The lost runner can't overwrite it if it wakes up late
    assert queue.finish(uuid.UUID(job["id"]), lost.worker, result={"stale": True}) is None


def test_a_job_that_keeps_losing_its_server_fails_with_a_message(queue):
    job = submit("montecarlo.run", MC)
    for _ in range(q.MAX_ATTEMPTS):
        assert queue.claim("gone") is not None
        queue.age_heartbeat(uuid.UUID(job["id"]), q.LEASE_S + 5)
        queue.recover()
    done = status(job["id"])
    assert done["status"] == "failed" and done["error_status"] == 503
    assert f"restarted {q.MAX_ATTEMPTS} times" in done["error"]


def test_a_live_heartbeat_is_left_alone(queue):
    submit("montecarlo.run", MC)
    queue.claim("alive")
    assert queue.recover() == {"requeued": 0, "failed": 0, "cancelled": 0}
    assert queue.counts()["running"] == 1


# ---------------------------------------------------------------------------
# Staying small
# ---------------------------------------------------------------------------
def test_results_and_old_jobs_are_pruned(queue):
    ids = []
    for s in range(q.KEEP_RESULTS_PER_OWNER + 1):
        ids.append(submit("montecarlo.run", {**MC, "seed": s, "scatter_points": 0})["id"])
        runner_for(queue).run_next()
    queue.prune()
    oldest = status(ids[0])
    assert oldest["result"] is None and oldest["result_expired"] is True
    assert status(ids[-1])["result"] is not None

    queue.age_finished(uuid.UUID(ids[-1]), q.RESULT_KEEP_S + 5)
    queue.prune()
    assert status(ids[-1])["result_expired"] is True

    queue.age_finished(uuid.UUID(ids[-1]), q.KEEP_JOBS_S + 5)
    assert queue.prune()["jobs_deleted"] == 1
    assert client.get(f"/api/jobs/{ids[-1]}").status_code == 404


def test_an_oversized_result_is_refused(queue, monkeypatch):
    monkeypatch.setattr(q, "MAX_RESULT_BYTES", 1000)
    job = submit("montecarlo.run", MC)
    assert runner_for(queue).run_next() == "failed"
    assert "too large" in status(job["id"])["error"]


def test_inputs_are_cleared_when_a_job_ends(queue):
    if not isinstance(queue, DatabaseQueue):
        pytest.skip("reads the table")
    job = submit("montecarlo.run", MC)
    runner_for(queue).run_next()
    with db_engine.connect() as conn:
        payload = conn.execute(select(Job.payload).where(Job.id == uuid.UUID(job["id"]))).scalar()
        nulls = conn.execute(select(func.count()).where(Job.payload.is_(None))).scalar()
    assert payload is None and nulls == 1


def test_database_claims_never_hand_one_job_to_two_runners(fresh_db):  # noqa: ARG001
    queue = DatabaseQueue()
    for s in range(12):
        queue.submit("owner", "montecarlo.run", {**MC, "seed": s}, max_active_per_owner=None)
    taken, lock = [], threading.Lock()

    def take(worker):
        while (job := queue.claim(worker)) is not None:
            with lock:
                taken.append(job.id)

    threads = [threading.Thread(target=take, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 12 and len(set(taken)) == 12


# ---------------------------------------------------------------------------
# The runner thread: many at once, and the API keeps answering
# ---------------------------------------------------------------------------
def test_ten_simultaneous_simulations_finish_while_health_answers(queue, monkeypatch):
    # The real wake-up: the API starts its own runner thread
    monkeypatch.setattr(config, "wake_runner", real_wake_runner)
    body = {"mc": {"n": 20000}, "seed": 3, "scatter_points": 100}
    direct = ok(client.post("/api/montecarlo/run", json=body))
    ids = []
    for i in range(10):
        as_user(f"user_{i}")
        ids.append(submit("montecarlo.run", body)["id"])
    worst, deadline = 0.0, time.monotonic() + 120
    try:
        while True:
            t0 = time.perf_counter()
            assert client.get("/api/health").status_code == 200
            worst = max(worst, time.perf_counter() - t0)
            states = [queue.get(f"user_{i}", uuid.UUID(j)).status for i, j in enumerate(ids)]
            if all(s == "succeeded" for s in states):
                break
            assert not any(s == "failed" for s in states), states
            assert time.monotonic() < deadline, states
            time.sleep(0.05)
        assert config.runner_alive()
        for i, job_id in enumerate(ids):
            as_user(f"user_{i}")
            assert without_timing(status(job_id)["result"]) == without_timing(direct)
        assert worst < 2.0, f"health took {worst:.2f}s during the runs"
    finally:
        config.use_queue(None)


def test_the_runner_stops_when_idle(queue):
    runner = runner_for(queue, poll_s=0.01, idle_exit_s=0.05, maintenance_s=0.01)
    runner.wake()
    deadline = time.monotonic() + 5
    while runner.alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not runner.alive()


def test_the_queue_is_configuration(monkeypatch):
    monkeypatch.delenv("FSE_JOB_QUEUE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert config.queue_mode() == "memory"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
    assert config.queue_mode() == "database"
    assert isinstance(config.build_queue("database"), DatabaseQueue)
    assert isinstance(config.build_queue("memory"), MemoryQueue)
    monkeypatch.setenv("FSE_JOB_QUEUE", "carrier-pigeon")
    with pytest.raises(ValueError):
        config.queue_mode()
    monkeypatch.setenv("FSE_JOB_RUNNER", "external")
    assert config.runner_mode() == "external"


def test_a_third_queue_needs_no_endpoint_changes(job_queue):  # noqa: ARG001
    """Any object with the queue's methods works behind the same endpoints."""
    calls = []

    class Recording(MemoryQueue):
        name = "recording"

        def submit(self, *a, **kw):
            calls.append("submit")
            return super().submit(*a, **kw)

    config.use_queue(Recording())
    job = submit("montecarlo.run", MC)
    runner_for(config.get_queue()).run_next()
    assert status(job["id"])["status"] == "succeeded" and calls == ["submit"]


# ---------------------------------------------------------------------------
# Scheduled endpoints: GitHub Actions OIDC tokens
# ---------------------------------------------------------------------------
REPO, REPO_ID = oidc.DEFAULT_REPOSITORY, oidc.DEFAULT_REPOSITORY_ID


@pytest.fixture(scope="module")
def github_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def github(monkeypatch, github_key):
    numbers = github_key.public_key().public_numbers()

    def b64(v: int) -> str:
        return base64.urlsafe_b64encode(v.to_bytes((v.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()

    served = {"keys": [{"kty": "RSA", "alg": "RS256", "use": "sig", "kid": "gh1",
                        "n": b64(numbers.n), "e": b64(numbers.e)}]}
    oidc.reset_key_cache()
    monkeypatch.setattr("jwt.PyJWKClient.fetch_data", lambda self: json.loads(json.dumps(served)))
    yield
    oidc.reset_key_cache()


def workflow_token(key, *, environment="local", workflow="scheduled.yml", ref="refs/heads/main",
                   repository=REPO, repository_id=REPO_ID, lifetime=300, issuer=oidc.GITHUB_ISSUER,
                   **extra) -> str:
    now = int(time.time())
    claims = {"iss": issuer, "aud": oidc.audience(environment), "iat": now, "nbf": now,
              "exp": now + lifetime, "sub": f"repo:{repository}:ref:{ref}",
              "repository": repository, "repository_id": repository_id,
              "job_workflow_ref": f"{repository}/.github/workflows/{workflow}@{ref}",
              "workflow_ref": f"{repository}/.github/workflows/{workflow}@{ref}",
              "ref": ref, "event_name": "schedule", "run_id": "987654321", "sha": "abc123", **extra}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "gh1"})


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_scheduled_endpoints_need_a_workflow_token(github, github_key):  # noqa: ARG001
    assert client.post("/api/scheduled/tasks/job-maintenance").status_code == 401
    # A user's session token is not a workflow token
    assert client.post("/api/scheduled/tasks/job-maintenance",
                       headers=bearer("dev:someone")).status_code == 401


@pytest.mark.parametrize("change, why", [
    ({"environment": "production"}, "minted for another environment"),
    ({"repository": "someone/FSE_ML"}, "another repository with the same workflow"),
    ({"repository_id": "1"}, "a repository that took this one's name"),
    ({"workflow": "tests.yml"}, "a workflow that isn't the scheduler"),
    ({"ref": "refs/heads/feature"}, "the scheduler, but from an unprotected branch"),
    ({"lifetime": -120}, "expired"),
    ({"issuer": "https://example.com"}, "not issued by GitHub"),
])
def test_workflow_tokens_are_checked(github, github_key, change, why):  # noqa: ARG001
    r = client.post("/api/scheduled/tasks/job-maintenance",
                    headers=bearer(workflow_token(github_key, **change)))
    assert r.status_code == 401, why


def test_a_token_signed_by_another_key_is_refused(github):  # noqa: ARG001
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r = client.post("/api/scheduled/tasks/job-maintenance", headers=bearer(workflow_token(other)))
    assert r.status_code == 401


def test_production_accepts_only_the_scheduler_on_main(monkeypatch, github, github_key):  # noqa: ARG001
    monkeypatch.setenv("FSE_ENV", "production")
    ok_token = workflow_token(github_key, environment="production")
    staging_workflow = workflow_token(github_key, environment="production", workflow="staging.yml",
                                      ref="refs/heads/staging")
    assert oidc.verify_workflow_token(ok_token).workflow == "scheduled.yml"
    with pytest.raises(Exception) as exc:
        oidc.verify_workflow_token(staging_workflow)
    assert getattr(exc.value, "status_code", None) == 401


def test_a_scheduled_task_runs_and_records_its_run(fresh_db, github, github_key):  # noqa: ARG001
    config.use_queue(DatabaseQueue())
    job = submit("montecarlo.run", MC)
    config.get_queue().claim("gone")
    config.get_queue().age_heartbeat(uuid.UUID(job["id"]), q.LEASE_S + 5)

    body = ok(client.post("/api/scheduled/tasks/job-maintenance",
                          headers=bearer(workflow_token(github_key))))
    assert body["status"] == "succeeded" and body["summary"]["requeued"] == 1
    assert body["github_run_id"] == 987654321
    with db_engine.connect() as conn:
        row = conn.execute(select(ScheduledRun)).one()
    assert (row.task, row.status, row.trigger, row.workflow, row.github_run_id) == \
        ("job-maintenance", "succeeded", "schedule", "scheduled.yml", 987654321)

    health = ok(client.get("/api/health/jobs"))
    assert health["queue"] == "database" and health["counts"]["queued"] == 1
    assert health["scheduled"]["job-maintenance"]["status"] == "succeeded"


def test_a_workflow_reports_a_script_it_ran(fresh_db, github, github_key):  # noqa: ARG001
    config.use_queue(DatabaseQueue())
    token = bearer(workflow_token(github_key))
    run = {"task": "supabase-keepalive", "status": "succeeded", "started_at": "2026-09-18T02:00:00Z",
           "summary": {"objects": 3}}
    assert client.post("/api/scheduled/runs", json=run, headers=token).status_code == 201
    unknown = {**run, "task": "made-up"}
    assert client.post("/api/scheduled/runs", json=unknown, headers=token).status_code == 422
    runs = ok(client.get("/api/scheduled/runs", headers=token))["tasks"]
    assert runs["supabase-keepalive"]["summary"] == {"objects": 3}


def test_unknown_tasks_are_refused(github, github_key):  # noqa: ARG001
    r = client.post("/api/scheduled/tasks/nope", headers=bearer(workflow_token(github_key)))
    assert r.status_code == 404 and "job-maintenance" in r.json()["detail"]


def test_the_drill_runs_many_identical_jobs(job_queue, github, github_key):  # noqa: ARG001
    token = bearer(workflow_token(github_key, workflow="staging.yml", ref="refs/heads/staging"))
    started = ok(client.post("/api/scheduled/drill", json={"count": 3}, headers=token), 202)
    assert len(started["jobs"]) == 3 and started["expected_summary"] == drill.EXPECTED_SUMMARY
    runner = runner_for(job_queue)
    while runner.run_next():
        pass
    for job_id in started["jobs"]:
        body = ok(client.get(f"/api/scheduled/drill/{job_id}", headers=token))
        assert body["status"] == "succeeded" and body["mismatched"] == []
        # Not ==: Linux numpy differs from Windows in the last bits (1e-9, like the golden file)
        assert body["summary"]["mean_irr"] == pytest.approx(drill.EXPECTED_SUMMARY["mean_irr"], rel=1e-9)
    # A user can't see the drill's jobs
    assert client.get(f"/api/jobs/{started['jobs'][0]}").status_code == 404


def test_the_drill_never_runs_in_production(monkeypatch, github, github_key):  # noqa: ARG001
    monkeypatch.setenv("FSE_ENV", "production")
    token = bearer(workflow_token(github_key, environment="production"))
    assert client.post("/api/scheduled/drill", headers=token).status_code == 404


def test_a_result_that_isnt_json_fails_the_job_cleanly(queue):
    """Not stuck as running until the lease runs out and retried three times."""
    from jobs.kinds import KINDS, JobKind

    real = KINDS["montecarlo.run"]
    broken = JobKind(real.name, real.title, real.request, real.response,
                     lambda req: {**real.run(req), "elapsed_ms": float("nan")}, real.stages)
    job = submit("montecarlo.run", MC)
    assert runner_for(queue, kinds={"montecarlo.run": broken}).run_next() == "failed"
    done = status(job["id"])
    assert done["attempts"] == 1 and done["error_status"] == 500 and "unexpectedly" in done["error"]


# ---------------------------------------------------------------------------
# The scheduler's script (ops/scheduled.py), driven against this app
# ---------------------------------------------------------------------------
class InProcessApi:
    """ops.scheduled.Api, but its HTTP goes to this app instead of the network."""

    def __new__(cls, key, environment="local"):
        from ops import scheduled as ops_scheduled

        class _Api(ops_scheduled.Api):
            def call(self, method, path, body=None, *, auth=True, timeout=120):
                headers = {"Authorization": f"Bearer {self.token()}"} if auth else {}
                r = client.request(method, path, json=body, headers=headers)
                return r.status_code, r.json()

        return _Api("http://testserver", environment,
                    token_source=lambda aud: workflow_token(key, workflow="staging.yml",
                                                            ref="refs/heads/staging"))


def test_the_drill_script_passes_on_a_healthy_server(fresh_db, github, github_key, monkeypatch):  # noqa: ARG001
    from ops import scheduled as ops_scheduled

    monkeypatch.setattr(config, "wake_runner", real_wake_runner)
    config.use_queue(DatabaseQueue())
    try:
        outcome = ops_scheduled.drill(InProcessApi(github_key), count=10, log=lambda *_: None,
                                      sleep=lambda _: time.sleep(0.2), timeout_s=300)
    finally:
        config.use_queue(None)
    assert outcome.problems() == []
    assert outcome.summary()["succeeded"] == 10 and outcome.health_checks > 0
    with db_engine.connect() as conn:
        run = conn.execute(select(ScheduledRun).where(ScheduledRun.task == "job-drill")).one()
    assert run.status == "succeeded" and run.workflow == "staging.yml" and run.summary["jobs"] == 10


def test_the_drill_script_fails_when_it_should():
    from ops.scheduled import DrillOutcome

    good = {"status": "succeeded", "mismatched": [], "attempts": 1}
    assert DrillOutcome(jobs={"a": good}, health_checks=3).problems() == []
    assert DrillOutcome(jobs={"a": {"status": "failed", "error": "boom"}}, health_checks=3).problems()
    assert DrillOutcome(jobs={"a": {**good, "mismatched": ["mean_irr"]}}, health_checks=3).problems()
    assert DrillOutcome(jobs={"a": good}, health_checks=3, health_failures=1).problems()
    assert DrillOutcome(jobs={"a": good}, health_checks=3, slowest_health_s=30).problems()
    assert DrillOutcome(jobs={"a": good}).problems()


def test_scheduled_task_script(fresh_db, github, github_key):  # noqa: ARG001
    from ops import scheduled as ops_scheduled

    config.use_queue(DatabaseQueue())
    api = InProcessApi(github_key)
    api.wake = lambda: {"status": "ok"}
    body = ops_scheduled.run_task(api, "job-maintenance", log=lambda *_: None)
    assert body["status"] == "succeeded"
    with pytest.raises(ops_scheduled.Failed):
        ops_scheduled.run_task(api, "no-such-task", log=lambda *_: None)


def test_a_dedicated_worker_runs_what_the_api_queued(fresh_db, monkeypatch):  # noqa: ARG001
    """Phase 12's switch: the API only queues (FSE_JOB_RUNNER=external) and a
    separate worker process runs the jobs. Same endpoints, same result."""
    from jobs import worker

    monkeypatch.setenv("FSE_JOB_RUNNER", "external")
    monkeypatch.setattr(config, "wake_runner", real_wake_runner)
    config.use_queue(DatabaseQueue())
    job = submit("montecarlo.run", MC)
    assert not config.runner_alive()           # the API didn't start one
    assert worker.main([], once=True) == 0
    assert without_timing(status(job["id"])["result"]) == \
        without_timing(ok(client.post("/api/montecarlo/run", json=MC)))
