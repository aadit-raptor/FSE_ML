"""The development cycle's CI gates (PLAN.md 0.3, docs/WORKFLOW.md):
PR titles start with an ECC type, Python coverage never drops below the
recorded floor, the floor only rises, and changed lines are 80% covered."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from ops import coverage_gate, pr_title

ROOT = Path(__file__).resolve().parent.parent


# --- PR titles ---------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "feat: saved deals",
    "fix(api): retry gateway errors",
    "docs: merge the ECC development cycle",
    "refactor: split the money keys",
    "test: pin the heatmap cells",
    "perf: vectorise the backtest",
    "ci: coverage gates",
    "ci!: drop the old workflow",
    "chore: bring staging level with main",
    "chore(deps): bump the python group with 3 updates",
])
def test_a_title_with_an_ecc_type_passes(title):
    assert pr_title.problem(title) is None


@pytest.mark.parametrize("title", [
    "Currency and money units everywhere (PLAN.md 2.2)",
    "Bump the python group with 3 updates",
    "Feat: capitalised type",
    "feature: not an ECC type",
    "build: not in this project's list",
    "feat:no space after the colon",
    "feat : space before the colon",
    "feat: ",
    "feat():empty scope",
    "WIP feat: prefixed",
    "",
])
def test_a_title_without_one_fails_and_says_which_types_work(title):
    message = pr_title.problem(title)
    assert message is not None
    assert "feat:" in message and "ci:" in message


def test_the_title_check_exits_1_on_a_bad_title(monkeypatch, capsys):
    monkeypatch.setenv("PR_TITLE", "Add coverage")
    assert pr_title.main() == 1
    assert "::error" in capsys.readouterr().out
    monkeypatch.setenv("PR_TITLE", "ci: add coverage")
    assert pr_title.main() == 0


def test_the_types_are_the_ones_docs_workflow_lists():
    step7 = (ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8").split("### 7. Ship")[1]
    commits = step7.split("**Commits**")[1].split("\n- ")[0]
    listed = re.findall(r"`([a-z]+):`", commits)
    assert listed and sorted(listed) == sorted(pr_title.TYPES)


def test_dependabot_titles_pass_the_check():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    for update in config["updates"]:
        message = update.get("commit-message") or {}
        # Dependabot writes "<prefix>(deps): Bump ..." with include: scope
        title = f"{message.get('prefix', '')}{'(deps)' if message.get('include') == 'scope' else ''}: Bump x"
        assert pr_title.problem(title) is None, update["package-ecosystem"]


# --- Coverage floor ----------------------------------------------------------

FLOORS = "# a comment\ncore 71.2\n\nml 80.4  # trailing comment\n"


def test_floors_parse_per_job():
    assert coverage_gate.parse_floors(FLOORS) == {"core": 71.2, "ml": 80.4}


@pytest.mark.parametrize("text", ["core", "core seventy", "core 71 extra", "core 101", "core 1\ncore 2"])
def test_a_malformed_floor_file_is_refused(text):
    with pytest.raises(ValueError, match="floor file line"):
        coverage_gate.parse_floors(text)


def test_coverage_at_or_above_the_floor_passes():
    floors = {"core": 71.2, "ml": 80.4}
    assert coverage_gate.problems("core", 71.2, floors, floors) == []
    assert coverage_gate.problems("ml", 90.0, floors, None) == []


def test_coverage_below_the_floor_fails():
    found = coverage_gate.problems("core", 71.19, {"core": 71.2}, None)
    assert len(found) == 1 and "71.19" in found[0] and "71.2" in found[0]


def test_lowering_the_floor_fails_even_when_coverage_meets_it():
    found = coverage_gate.problems("core", 75.0, {"core": 70.0}, {"core": 71.2})
    assert len(found) == 1 and "lower" in found[0]


def test_removing_a_job_from_the_floor_file_fails():
    # Checked from the other job's run too: core's gate notices ml's line went
    found = coverage_gate.problems("core", 75.0, {"core": 71.2}, {"core": 71.2, "ml": 80.4})
    assert len(found) == 1 and "ml" in found[0] and "nothing" in found[0]


def test_raising_the_floor_passes():
    assert coverage_gate.problems("core", 75.0, {"core": 74.0}, {"core": 71.2}) == []


def test_a_job_without_a_floor_fails():
    assert coverage_gate.problems("ml", 90.0, {"core": 71.2}, None)


def test_a_hint_suggests_raising_the_floor_when_coverage_is_well_above_it():
    assert coverage_gate.raise_hint("core", 72.66, {"core": 71.2}) is not None
    assert "72.6" in coverage_gate.raise_hint("core", 72.66, {"core": 71.2})
    assert coverage_gate.raise_hint("core", 71.5, {"core": 71.2}) is None


def _run_gate(tmp_path, percent, floor_text, base_text=None, job="core"):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"totals": {"percent_covered": percent}}), encoding="utf-8")
    floor = tmp_path / "floor"
    floor.write_text(floor_text, encoding="utf-8")
    args = ["--job", job, "--coverage", str(report), "--floor", str(floor)]
    if base_text is not None:
        base = tmp_path / "base"
        base.write_text(base_text, encoding="utf-8")
        args += ["--base-floor", str(base)]
    return coverage_gate.main(args)


def test_the_gate_reads_a_real_coverage_report(tmp_path, capsys):
    assert _run_gate(tmp_path, 71.3, "core 71.2\n") == 0
    assert "71.3" in capsys.readouterr().out
    assert _run_gate(tmp_path, 71.1, "core 71.2\n") == 1
    assert _run_gate(tmp_path, 75.0, "core 70\n", base_text="core 71.2\n") == 1


def test_a_missing_base_floor_file_skips_the_rise_check(tmp_path):
    # The first PR that adds the file has nothing to compare against
    assert _run_gate(tmp_path, 71.3, "core 71.2\n", base_text="") == 0


def test_the_repository_floor_file_covers_both_python_jobs():
    floors = coverage_gate.parse_floors((ROOT / ".coverage-floor").read_text(encoding="utf-8"))
    assert set(floors) == {"core", "ml"}
    assert all(50 <= value <= 100 for value in floors.values())


# --- The workflows keep the gates -------------------------------------------

def _workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _run_lines(job):
    return "\n".join(step.get("run", "") for step in job["steps"])


@pytest.mark.parametrize("job", ["core", "ml"])
def test_both_python_jobs_measure_coverage_against_the_floor(job):
    runs = _run_lines(_workflow("tests.yml")["jobs"][job])
    assert "--cov" in runs
    assert f"ops.coverage_gate --job {job}" in runs
    assert "GITHUB_STEP_SUMMARY" in runs


def test_changed_lines_must_be_80_percent_covered():
    job = _workflow("tests.yml")["jobs"]["ml"]
    assert "diff-cover" in _run_lines(job) and "--fail-under=80" in _run_lines(job)
    checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["fetch-depth"] == 0  # diff-cover needs the merge base


def test_the_title_check_runs_on_every_title_change_with_a_read_only_token():
    flow = _workflow("pr.yml")
    on = flow.get("on", flow.get(True))  # YAML reads a bare `on` as True
    assert set(on["pull_request"]["types"]) >= {"opened", "edited", "reopened", "synchronize"}
    assert flow["permissions"] == {"contents": "read"}
    runs = _run_lines(flow["jobs"]["title"])
    assert "ops.pr_title" in runs and "${{" not in runs  # the title reaches it via env, never the script
