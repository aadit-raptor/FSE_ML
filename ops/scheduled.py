"""The scheduler's side of scheduled jobs (PLAN.md 1.9), run by GitHub Actions.

    python -m ops.scheduled task job-maintenance --api URL --environment production
    python -m ops.scheduled keepalive --api URL --environment production
    python -m ops.scheduled drill --api URL --environment staging

Each call proves which workflow it is with the OIDC token GitHub mints for the
run (the workflow needs ``permissions: id-token: write``); the API checks it
(api/github_oidc.py). No secret is stored anywhere for this.

- ``task`` runs a task in the API (``/api/scheduled/tasks/{name}``), which
  records the run.
- ``keepalive`` lists the Supabase Storage bucket, so the free project never
  counts as inactive (it pauses after a week without activity), then reports
  the run to the API.
- ``drill`` is the "done when" check on staging: it queues ten identical
  seeded simulations at once, polls ``/api/health`` the whole time, and
  passes only if every job succeeded with the pinned seeded summary and the
  health check kept answering. The outcome is recorded as the ``job-drill``
  run.

Nothing printed here contains a token or deal contents. Standard library only
(plus ops.backup_store for the keep-alive).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

AUDIENCE_PREFIX = "fse-scheduler:"     # api/github_oidc.py
TOKEN_REUSE_S = 120                    # GitHub's tokens live a few minutes
# The drill fails if a health check takes longer than this or fails at all.
# Render's free instance has a fraction of a CPU, shared with the runs.
HEALTH_LIMIT_S = 10.0
DRILL_TIMEOUT_S = 20 * 60
POLL_S = 2.0


class Failed(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def github_oidc_token(audience: str) -> str:
    """A fresh OIDC token for this workflow run, for ``audience``."""
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not bearer:
        raise Failed("no GitHub OIDC token: run this in GitHub Actions with "
                     "'permissions: id-token: write'")
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}audience={urllib.parse.quote(audience)}",
                                 headers={"Authorization": f"bearer {bearer}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["value"]


class Api:
    """HTTP calls to the API, with a workflow token when asked."""

    def __init__(self, base: str, environment: str,
                 token_source: Callable[[str], str] = github_oidc_token):
        self.base = base.rstrip("/")
        self.audience = AUDIENCE_PREFIX + environment
        self._token_source = token_source
        self._token: tuple[float, str] = (0.0, "")

    def token(self) -> str:
        now = time.monotonic()
        if not self._token[1] or now - self._token[0] > TOKEN_REUSE_S:
            self._token = (now, self._token_source(self.audience))
        return self._token[1]

    def call(self, method: str, path: str, body: Optional[dict] = None, *, auth: bool = True,
             timeout: float = 120) -> tuple[int, dict]:
        headers = {"Accept": "application/json", "X-Request-ID": f"scheduler-{os.environ.get('GITHUB_RUN_ID', 'local')}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self.token()}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except ValueError:
                return e.code, {"detail": e.reason}

    def wake(self, attempts: int = 4) -> dict:
        """Wait for a sleeping free service (about a minute) and return its health."""
        last = "no answer"
        for i in range(attempts):
            try:
                code, body = self.call("GET", "/api/health", auth=False)
                if code == 200:
                    return body
                last = f"HTTP {code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = type(exc).__name__
            time.sleep(min(30, 5 * (i + 1)))
        raise Failed(f"the API at {self.base} did not answer /api/health ({last})")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def run_task(api: Api, name: str, log=print) -> dict:
    api.wake()
    code, body = api.call("POST", f"/api/scheduled/tasks/{urllib.parse.quote(name)}")
    if code != 200:
        raise Failed(f"task {name}: HTTP {code}: {body.get('detail')}")
    log(f"task {name}: {body['status']} {json.dumps(body.get('summary', {}), sort_keys=True)}")
    return body


def report(api: Api, task: str, status: str, started_at: str, summary: dict,
           error: Optional[str] = None) -> None:
    code, body = api.call("POST", "/api/scheduled/runs", {
        "task": task, "status": status, "started_at": started_at, "summary": summary,
        **({"error": error[:500]} if error else {})})
    if code != 201:
        raise Failed(f"recording {task}: HTTP {code}: {body.get('detail')}")


def keepalive(api: Api, log=print) -> dict:
    from ops.backup_store import store_from_env

    started = utc_now_iso()
    try:
        objects = store_from_env().list("")
        summary, status, error = {"objects": len(objects)}, "succeeded", None
    except Exception as exc:  # noqa: BLE001 - recorded, then the workflow fails
        summary, status, error = {}, "failed", f"{type(exc).__name__}: listing the bucket failed"
    api.wake()
    report(api, "supabase-keepalive", status, started, summary, error)
    log(f"supabase-keepalive: {status} {summary}")
    if status == "failed":
        raise Failed(error)
    return summary


@dataclass
class DrillOutcome:
    jobs: dict[str, dict] = field(default_factory=dict)
    health_checks: int = 0
    health_failures: int = 0
    slowest_health_s: float = 0.0
    seconds: float = 0.0

    def problems(self) -> list[str]:
        found = []
        not_done = [j for j in self.jobs.values() if j.get("status") != "succeeded"]
        if not_done:
            found.append(f"{len(not_done)} of {len(self.jobs)} jobs did not succeed: "
                         + "; ".join(f"{j.get('status')} ({j.get('error') or j.get('stage')})" for j in not_done[:3]))
        wrong = [j for j in self.jobs.values() if j.get("mismatched")]
        if wrong:
            found.append(f"{len(wrong)} results differ from the pinned seeded summary: {wrong[0]['mismatched']}")
        if self.health_failures:
            found.append(f"/api/health failed {self.health_failures} of {self.health_checks} times during the runs")
        if self.slowest_health_s > HEALTH_LIMIT_S:
            found.append(f"/api/health took {self.slowest_health_s:.1f}s during the runs (limit {HEALTH_LIMIT_S:g}s)")
        if self.health_checks == 0:
            found.append("the health check was never polled")
        return found

    def summary(self) -> dict:
        return {"jobs": len(self.jobs),
                "succeeded": sum(1 for j in self.jobs.values() if j.get("status") == "succeeded"),
                "resumed": sum(1 for j in self.jobs.values() if (j.get("attempts") or 0) > 1),
                "health_checks": self.health_checks, "health_failures": self.health_failures,
                "slowest_health_ms": round(self.slowest_health_s * 1000), "seconds": round(self.seconds)}


def drill(api: Api, count: int = 10, *, log=print, sleep=time.sleep, timeout_s: float = DRILL_TIMEOUT_S) -> DrillOutcome:
    api.wake()
    started_iso, t0 = utc_now_iso(), time.monotonic()
    code, body = api.call("POST", "/api/scheduled/drill", {"count": count})
    if code != 202:
        raise Failed(f"starting the drill: HTTP {code}: {body.get('detail')}")
    outcome = DrillOutcome(jobs={job_id: {"status": "queued"} for job_id in body["jobs"]})
    log(f"drill: {len(outcome.jobs)} jobs queued at once ({json.dumps(body['request'])})")
    while True:
        h0 = time.monotonic()
        try:
            hcode, _ = api.call("GET", "/api/health", auth=False, timeout=30)
            ok = hcode == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            ok = False
        outcome.health_checks += 1
        outcome.health_failures += 0 if ok else 1
        outcome.slowest_health_s = max(outcome.slowest_health_s, time.monotonic() - h0)
        for job_id, job in outcome.jobs.items():
            if job.get("status") in ("succeeded", "failed", "cancelled"):
                continue
            jcode, jbody = api.call("GET", f"/api/scheduled/drill/{job_id}")
            if jcode == 200:
                outcome.jobs[job_id] = jbody
        states = [j.get("status") for j in outcome.jobs.values()]
        done = sum(s in ("succeeded", "failed", "cancelled") for s in states)
        log(f"  {done}/{len(states)} finished, health {'ok' if ok else 'FAILED'} "
            f"({(time.monotonic() - h0) * 1000:.0f} ms)")
        if done == len(states) or time.monotonic() - t0 > timeout_s:
            break
        sleep(POLL_S)
    outcome.seconds = time.monotonic() - t0
    problems = outcome.problems()
    report(api, "job-drill", "failed" if problems else "succeeded", started_iso, outcome.summary(),
           "; ".join(problems) or None)
    log(f"drill: {json.dumps(outcome.summary(), sort_keys=True)}")
    return outcome


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("task", "keepalive", "drill"):
        p = sub.add_parser(name)
        p.add_argument("--api", required=True, help="the API's base URL")
        p.add_argument("--environment", required=True, choices=["production", "staging"])
        if name == "task":
            p.add_argument("name")
        if name == "drill":
            p.add_argument("--count", type=int, default=10)
    args = parser.parse_args(argv)
    api = Api(args.api, args.environment)
    try:
        if args.command == "task":
            run_task(api, args.name)
        elif args.command == "keepalive":
            keepalive(api)
        else:
            problems = drill(api, args.count).problems()
            if problems:
                raise Failed("; ".join(problems))
    except Failed as exc:
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
