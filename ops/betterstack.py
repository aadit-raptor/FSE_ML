"""Uptime monitors, status page and alerts on Better Stack (PLAN.md 1.2).

Run by GitHub Actions with the secret BETTERSTACK_API_TOKEN (a Better Stack
Uptime API token). Standard library only.

    python ops/betterstack.py sync [--dry-run]     # create or update monitors and the status page
    python ops/betterstack.py report               # print monitors and the status page URL
    python ops/betterstack.py incident --name N --summary S   # raise an alert now

Free-plan design:
- Render's free API sleeps after 15 minutes idle and takes about a minute to
  wake, and all free services share 750 instance hours a month. Checking the
  API every 3 minutes would keep it awake all month (744 h) and leave nothing
  for staging, so the API is checked every 30 minutes, which lets it sleep
  between checks. A check that lands on a sleeping API waits for it to wake;
  ``confirmation_period`` keeps rechecking for 4 minutes before alerting, so
  an expected wake-up never alerts but an API that stays down does.
- The web app on Vercel doesn't sleep and costs no Render hours: checked
  every 3 minutes.
- Staging has no scheduled check (it would keep staging awake). Instead
  .github/workflows/staging.yml checks every staging deploy and raises an
  incident here the moment a check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

API = "https://uptime.betterstack.com/api"

# Render free plan (render.com/docs/free)
RENDER_FREE_HOURS = 750
RENDER_SLEEP_AFTER_S = 15 * 60
RENDER_WAKE_S = 60
MONTH_HOURS = 31 * 24
# Hours kept free for staging deploy checks, previews and real use of production
RENDER_HOURS_RESERVED = 250
# Longest wake-up a monitor must ride out without alerting
WAKE_TOLERANCE_S = 180

STATUS_PAGE = {
    "company_name": "FSE/ML",
    "company_url": "https://fse-ml.vercel.app",
    "subdomain": os.environ.get("BETTERSTACK_STATUS_SUBDOMAIN", "fse-ml"),
    "timezone": "UTC",
}

MONITORS = [
    {
        "pronounceable_name": "FSE/ML web (production)",
        # Signed-out pages answer 404 or redirect to sign-in (PLAN.md 1.4), so
        # the monitor checks the app's public health route instead
        # (web/src/app/healthz/route.ts; tests/test_betterstack.py pins both)
        "url": "https://fse-ml.vercel.app/healthz",
        "monitor_type": "keyword",
        "required_keyword": '"service":"FSE/ML web"',
        "check_frequency": 180,
        "request_timeout": 30,
        "confirmation_period": 120,
        "recovery_period": 180,
        "email": True,
        "push": True,
        "follow_redirects": True,
        "verify_ssl": True,
        "_host": "vercel",
        "_public_name": "Website",
    },
    {
        "pronounceable_name": "FSE/ML API (production)",
        "url": "https://fse-api.onrender.com/api/health",
        "monitor_type": "keyword",
        "required_keyword": '"status":"ok"',
        "check_frequency": 1800,
        "request_timeout": 60,
        "confirmation_period": 240,
        "recovery_period": 180,
        "email": True,
        "push": True,
        "follow_redirects": True,
        "verify_ssl": True,
        "_host": "render-free",
        "_public_name": "Model API",
    },
]


def monitor_body(m: dict) -> dict:
    """The fields sent to Better Stack (keys starting with _ are ours)."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


def render_awake_hours(check_frequency_s: int) -> float:
    """Hours a month a free Render service stays awake from one uptime check.

    Each check wakes the service (if asleep) and keeps it up for the sleep
    timeout; checks closer together than that keep it awake all month.
    """
    cycle = RENDER_WAKE_S + RENDER_SLEEP_AFTER_S
    if check_frequency_s <= cycle:
        return float(MONTH_HOURS)
    return MONTH_HOURS * cycle / check_frequency_s


def check_plan(monitors: list[dict] = MONITORS) -> list[str]:
    """Problems with the monitor plan against free limits; empty when fine."""
    problems = []
    render = [m for m in monitors if m["_host"] == "render-free"]
    hours = sum(render_awake_hours(m["check_frequency"]) for m in render)
    if hours + RENDER_HOURS_RESERVED > RENDER_FREE_HOURS:
        problems.append(f"uptime checks keep Render awake {hours:.0f} h/month; with "
                        f"{RENDER_HOURS_RESERVED} h reserved that exceeds the free {RENDER_FREE_HOURS} h")
    for m in render:
        if m["request_timeout"] + m["confirmation_period"] < WAKE_TOLERANCE_S:
            problems.append(f"{m['pronounceable_name']}: would alert on a normal wake-up")
    for m in monitors:
        if m["check_frequency"] < 180:
            problems.append(f"{m['pronounceable_name']}: Better Stack free checks at most every 3 minutes")
    if len(monitors) > 10:
        problems.append("Better Stack free allows 10 monitors")
    return problems


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
Requester = Callable[[str, str, Optional[dict]], dict]


class BetterStackError(RuntimeError):
    pass


def http_requester(token: str) -> Requester:
    def request(method: str, url: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise BetterStackError(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')[:500]}") from None
        return json.loads(raw) if raw else {}
    return request


def list_all(request: Requester, url: str) -> list[dict]:
    items = []
    while url:
        page = request("GET", url, None)
        items.extend(page.get("data") or [])
        url = (page.get("pagination") or {}).get("next")
    return items


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def sync(request: Requester, dry_run: bool = False, log=print) -> dict:
    """Make Better Stack match MONITORS and STATUS_PAGE. Returns monitor ids by name."""
    problems = check_plan()
    if problems:
        raise BetterStackError("; ".join(problems))

    existing = {m["attributes"]["pronounceable_name"]: m for m in list_all(request, f"{API}/v2/monitors")}
    ids = {}
    for m in MONITORS:
        body = monitor_body(m)
        name = body["pronounceable_name"]
        current = existing.get(name)
        if current is None:
            log(f"create monitor: {name}")
            ids[name] = None if dry_run else request("POST", f"{API}/v2/monitors", body)["data"]["id"]
            continue
        ids[name] = current["id"]
        changed = {k: v for k, v in body.items() if current["attributes"].get(k) != v}
        if changed:
            log(f"update monitor: {name}: {sorted(changed)}")
            if not dry_run:
                request("PATCH", f"{API}/v2/monitors/{current['id']}", changed)
        else:
            log(f"monitor up to date: {name}")

    pages = list_all(request, f"{API}/v2/status-pages")
    page = next((p for p in pages if p["attributes"].get("subdomain") == STATUS_PAGE["subdomain"]), None)
    if page is None:
        log(f"create status page: {STATUS_PAGE['subdomain']}")
        if dry_run:
            return ids
        page = request("POST", f"{API}/v2/status-pages", STATUS_PAGE)["data"]
    resources = list_all(request, f"{API}/v2/status-pages/{page['id']}/resources")
    shown = {str(r["attributes"].get("resource_id")) for r in resources
             if r["attributes"].get("resource_type") == "Monitor"}
    for m in MONITORS:
        mid = ids.get(m["pronounceable_name"])
        if mid is not None and str(mid) not in shown:
            log(f"add to status page: {m['_public_name']}")
            if not dry_run:
                request("POST", f"{API}/v2/status-pages/{page['id']}/resources",
                        {"resource_type": "Monitor", "resource_id": int(mid),
                         "public_name": m["_public_name"], "widget_type": "history"})
    return ids


def report(request: Requester, log=print) -> None:
    for m in list_all(request, f"{API}/v2/monitors"):
        a = m["attributes"]
        log(f"monitor {a['pronounceable_name']}: {a.get('status')} every {a.get('check_frequency')}s, "
            f"last checked {a.get('last_checked_at')}")
    for p in list_all(request, f"{API}/v2/status-pages"):
        a = p["attributes"]
        log(f"status page https://{a.get('subdomain')}.betteruptime.com ({a.get('aggregate_state')})")


def raise_incident(request: Requester, name: str, summary: str) -> str:
    body = {
        "requester_email": os.environ.get("BETTERSTACK_REQUESTER_EMAIL",
                                          "github-actions@users.noreply.github.com"),
        "name": name,
        "summary": summary,
        "email": True,
        "push": True,
        "call": False,
        "sms": False,
    }
    return request("POST", f"{API}/v3/incidents", body)["data"]["id"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("report")
    sub.add_parser("check")
    i = sub.add_parser("incident")
    i.add_argument("--name", required=True)
    i.add_argument("--summary", required=True)
    args = parser.parse_args(argv)

    if args.command == "check":
        problems = check_plan()
        for p in problems:
            print(f"::error::{p}")
        return 1 if problems else 0

    token = os.environ.get("BETTERSTACK_API_TOKEN")
    if not token:
        print("::error::Set the GitHub Actions secret BETTERSTACK_API_TOKEN (Better Stack > Uptime > API tokens)")
        return 1
    request = http_requester(token)
    try:
        if args.command == "sync":
            sync(request, dry_run=args.dry_run, log=lambda msg: print(f"::notice::{msg}"))
            report(request, log=lambda msg: print(f"::notice::{msg}"))
        elif args.command == "report":
            report(request, log=lambda msg: print(f"::notice::{msg}"))
        else:
            incident = raise_incident(request, args.name, args.summary)
            print(f"::notice::Better Stack incident {incident} raised: {args.name}")
    except BetterStackError as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
