"""The production site's own domain (PLAN.md 0.2).

Production is https://variater.com (web) and https://api.variater.com (API).
The old hosts still answer, since Vercel and Render keep them, but the web one
only redirects to the domain (web/next.config.ts), and nothing the project
runs or checks may point at either any more.
"""
import re
import subprocess
from pathlib import Path

import pytest

from api.security import DEFAULT_ORIGINS, PRODUCTION_WEB_ORIGIN, cors_origins
from ops import betterstack as bs

ROOT = Path(__file__).resolve().parents[1]
WEB = "https://variater.com"
API = "https://api.variater.com"
OLD_HOSTS = ("fse-ml.vercel.app", "fse-api.onrender.com")

# Files that may still name an old host, and why
ALLOWED = {
    "web/next.config.ts": "the redirect from the old address",
    "web/e2e/live.spec.ts": "checks the old address redirects",
    "web/e2e/domain.spec.ts": "checks the redirect rule by Host header",
    "tests/test_domain.py": "this file",
    "tests/test_security.py": "checks the old origin is refused",
    "CLAUDE.md": "history",
    "PLAN.md": "history",
    "DEPLOY.md": "history and the switch-over steps",
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return [f for f in out.stdout.split("\0") if f and not f.endswith(("package-lock.json", ".png", ".ico", ".pkl", ".pt"))]


def test_nothing_the_project_runs_names_the_old_hosts():
    stale = []
    for name in tracked_files():
        if name in ALLOWED:
            continue
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for host in OLD_HOSTS:
            if host in text:
                stale.append(f"{name}: {host}")
    assert stale == []


def test_production_cors_origin_is_the_domain():
    assert PRODUCTION_WEB_ORIGIN == WEB
    assert DEFAULT_ORIGINS["production"] == (WEB,)


def test_production_cors_refuses_the_old_address(monkeypatch):
    monkeypatch.delenv("FSE_CORS_ORIGINS", raising=False)
    assert "https://fse-ml.vercel.app" not in cors_origins("production")
    assert cors_origins("production") == [WEB]


def test_monitors_and_status_page_use_the_domain():
    urls = {m["_host"]: m["url"] for m in bs.MONITORS}
    assert urls == {"vercel": f"{WEB}/healthz", "render-free": f"{API}/api/health"}
    assert bs.STATUS_PAGE["company_url"] == WEB
    assert bs.STATUS_PAGE["company_name"] == "Variater"


@pytest.mark.parametrize("workflow", ["live.yml", "scheduled.yml"])
def test_production_workflows_call_the_domain(workflow):
    text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    assert API in text


def test_live_checks_default_to_the_domain():
    config = (ROOT / "web/playwright.config.ts").read_text(encoding="utf-8")
    assert re.search(r'E2E_BASE_URL \?\? "https://variater\.com"', config)
    live = (ROOT / ".github/workflows/live.yml").read_text(encoding="utf-8")
    assert f"--web {WEB} --api {API}" in live


def test_old_address_and_www_redirect_to_the_domain():
    """web/next.config.ts sends every path on the old Vercel address and on
    www to the same path on the domain. Checked for real by
    web/e2e/domain.spec.ts (locally) and live.spec.ts (production)."""
    config = (ROOT / "web/next.config.ts").read_text(encoding="utf-8")
    for host in ("fse-ml.vercel.app", "www.variater.com"):
        escaped = host.replace(".", "\\\\.")
        assert f'"{escaped}"' in config, escaped
    assert 'destination: "https://variater.com/:path*"' in config


def test_the_brand_is_variater():
    import json

    messages = json.loads((ROOT / "web/messages/en.json").read_text(encoding="utf-8"))
    assert messages["app"]["brand"] == "Variater"
    visible = json.dumps(messages) + (ROOT / "web/src/app/global-error.tsx").read_text(encoding="utf-8") \
        + (ROOT / "README.md").read_text(encoding="utf-8")
    assert "FSE/ML" not in visible
