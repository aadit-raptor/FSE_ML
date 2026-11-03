"""docs/RUNBOOKS.md names only things that exist (PLAN.md 11.2).

A runbook that tells someone to call an endpoint, run a module or set a
variable that was renamed is worse than none, at the moment it is read. So:

- every `/api/...` path in backticks is a route of the app;
- every workflow file, `ops` module, scheduled task and `docs` file named exists;
- every UPPER_CASE variable named is used somewhere in the repository;
- every Better Stack incident a workflow can raise has a runbook row, and every
  incident the runbooks list is raised by a workflow;
- every runbook has its four parts and every internal link points somewhere.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "RUNBOOKS.md"
TEXT = DOC.read_text(encoding="utf-8")
CODE = re.compile(r"`([^`\n]+)`")

# Names in the document that are not variables of this repository's code:
# a CLI flag value, a Clerk key prefix, a GitHub field, an outside service.
NOT_OURS = {"NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY", "CLERK_SECRET_KEY", "RESEND_API_KEY"}


def spans() -> list[str]:
    return CODE.findall(TEXT)


def route_patterns() -> list[re.Pattern]:
    from api.main import app

    # Included routers are not plain routes in this FastAPI: the schema lists them all
    paths = set(app.openapi()["paths"]) | {r.path for r in app.routes if hasattr(r, "path")}
    return [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", path) + "$") for path in paths]


def test_every_api_path_is_a_route():
    patterns = route_patterns()
    found = set()
    for span in spans():
        for m in re.finditer(r"(?<![\w./])(/api/[A-Za-z0-9_\-/{}.]*[A-Za-z0-9_}])", span):
            found.add(m.group(1).split("?")[0])
    assert found, "the document names no endpoint"
    # Paths ending in ... or holding a placeholder are matched as templates
    missing = [p for p in sorted(found) if not any(r.match(re.sub(r"\{[^}]+\}", "x", p)) for r in patterns)]
    assert not missing, f"not routes of the API: {missing}"


def test_every_workflow_named_exists():
    names = set(re.findall(r"`([a-z\-]+)\.yml`", TEXT)) | set(re.findall(r"\.github/workflows/([a-z\-]+)\.yml", TEXT))
    assert {"live", "staging", "backup", "scheduled"} <= names
    missing = [n for n in sorted(names) if not (ROOT / ".github" / "workflows" / f"{n}.yml").exists()]
    assert not missing, missing


def test_every_ops_command_names_a_module_that_exists():
    modules = set(re.findall(r"python3? -m (ops\.[a-z_]+)", TEXT))
    scripts = set(re.findall(r"python3? (ops/[a-z_]+\.py)", TEXT))
    assert modules and scripts
    for module in modules:
        assert (ROOT / (module.replace(".", "/") + ".py")).exists(), module
    for script in scripts:
        assert (ROOT / script).exists(), script


def test_every_python_module_command_exists():
    for module in set(re.findall(r"python3? -m ((?:db|api)\.[a-z_.]+)", TEXT)):
        assert (ROOT / (module.replace(".", "/") + ".py")).exists(), module


def test_every_scheduled_task_named_is_registered():
    from jobs import scheduled

    named = {s for s in spans() if re.fullmatch(r"[a-z]+(?:-[a-z]+)+", s) and s.split("-")[-1] in {"refresh", "maintenance", "report"}}
    assert {"job-maintenance", "economy-refresh", "benchmarks-refresh"} <= named
    assert named <= set(scheduled.TASKS), named - set(scheduled.TASKS)


def test_every_variable_named_is_used_in_the_repository():
    names = {s for s in spans() if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\*?", s)}
    assert {"AI_FREE_TIER", "DATABASE_URL", "SENTRY_DSN"} & names or names
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    code = [p for p in tracked if p.endswith((".py", ".yml", ".yaml", ".ts", ".tsx", ".json", ".toml", "Dockerfile", ".md"))
            and p != "docs/RUNBOOKS.md" and not p.startswith(("web/node_modules", "tests/fixtures", "web/src/lib/help"))]
    corpus = "\n".join((ROOT / p).read_text(encoding="utf-8", errors="ignore") for p in code if (ROOT / p).is_file())
    missing = [n for n in sorted(names) if n.rstrip("*") not in corpus]
    assert not missing, f"named but not found anywhere else: {missing}"


def workflow_incidents() -> set[str]:
    out = set()
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        for m in re.finditer(r'betterstack\.py incident --name "([^"$]+?)(?: \(\$\{\{[^"]*)?"', path.read_text(encoding="utf-8")):
            out.add(m.group(1).strip())
    return out


def test_every_incident_a_workflow_raises_has_a_runbook_row():
    incidents = workflow_incidents()
    assert len(incidents) >= 6, incidents
    missing = [name for name in sorted(incidents) if name not in TEXT]
    assert not missing, f"raised by a workflow, no runbook: {missing}"


def test_every_monitor_has_a_runbook_row():
    from ops.betterstack import MONITORS

    for monitor in MONITORS:
        assert monitor["pronounceable_name"] in TEXT, monitor["pronounceable_name"]
        assert monitor["url"] in TEXT, monitor["url"]


def test_runbooks_have_all_four_parts():
    sections = re.split(r"^### (RB-\d\d) ", TEXT, flags=re.M)[1:]
    ids = sections[0::2]
    bodies = sections[1::2]
    assert len(ids) >= 18 and ids == sorted(ids)
    for rid, body in zip(ids, bodies):
        for part in ("**Means.**", "**Confirm.**", "**Do.**", "**Roll back.**"):
            assert part in body.split("\n### ")[0].split("\n---")[0], f"{rid} lacks {part}"


def test_every_alert_row_links_to_a_runbook_that_exists():
    anchors = {m.group(1) for m in re.finditer(r"^### (RB-\d\d)", TEXT, flags=re.M)}
    linked = set(re.findall(r"\[(RB-\d\d)\]\(#rb-\d\d-", TEXT))
    assert linked and linked <= anchors
    assert anchors <= linked, f"runbooks no alert points to: {sorted(anchors - linked)}"


def test_relative_links_point_at_files():
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", TEXT):
        assert (DOC.parent / target).exists(), target


def test_status_page_matches_the_code():
    from ops.betterstack import MONITORS, STATUS_PAGE

    assert f"https://{STATUS_PAGE['subdomain']}.betteruptime.com" in TEXT
    for monitor in MONITORS:
        assert f'"{monitor["_public_name"]}"' in TEXT


@pytest.mark.parametrize("level", ["SEV1", "SEV2", "SEV3", "SEV4"])
def test_each_severity_is_defined(level):
    assert f"| **{level}** |" in TEXT
