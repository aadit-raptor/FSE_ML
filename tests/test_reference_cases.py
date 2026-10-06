"""The engine against hand-checked reference cases (PLAN.md 3.4).

``tests/reference/cases.py`` holds 26 small deals solved by hand from
``docs/methodology.md`` and ``tests/reference/reference_cases.xlsx`` the same
working as a spreadsheet. Each deal goes through the API exactly as the deal
screen sends it, and every line the working marks with an engine figure must
match to 0.01: money in the deal's own unit, IRR and rates in percentage
points, MOIC in turns. See ``tests/reference/README.md``.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.reference.cases import CASES, REQUIRED_COVERAGE
from tests.reference.workbook import evaluate, is_current

TOLERANCE = 0.01
SCALE = {"money": 1.0, "pct": 100.0, "x": 1.0}


def run(case) -> dict:
    response = TestClient(app).post("/api/deal/run", json={"inputs": case.inputs, "settings": case.settings})
    assert response.status_code == 200, response.text
    return response.json()


def lookup(answer, path: str):
    """A figure in the API's answer by dotted path: keys, list indexes, and a
    risk warning by its id (whose figures are then read by name)."""
    node = answer
    for part in path.split("."):
        if isinstance(node, list) and not part.isdigit():
            node = next(w for w in node if w["id"] == part)["figures"]
        else:
            node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_engine_matches_the_hand_working(case):
    answer = run(case)
    values = evaluate(case)
    misses = []
    for line in case.lines:
        if not line.check:
            continue
        got = lookup(answer, line.check)
        want = values[line.key]
        if got is None or abs(got - want) * SCALE[line.kind] > TOLERANCE:
            misses.append(f"{line.key} ({line.check}): hand {want:.6f}, engine {got}")
    assert not misses, f"{case.id} differs from its hand working:\n" + "\n".join(misses)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_working_comes_to_the_answers_typed_by_hand(case):
    """The headline figures were worked out on paper before the formulas
    were written; a slip in either shows up here."""
    values = evaluate(case)
    for key, typed in case.answers.items():
        assert values[key] == pytest.approx(typed, abs=1e-6), f"{case.id}: {key}"


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_every_case_checks_its_returns(case):
    checked = {line.check for line in case.lines}
    assert "returns.entry_equity" in checked
    assert "returns.irr" in checked or "returns.net_exit_equity" in checked
    assert sum(1 for line in case.lines if line.check) >= 10


def test_there_are_at_least_twenty_cases_covering_the_plan():
    assert len(CASES) >= 20
    assert len({c.id for c in CASES}) == len(CASES)
    covered = {tag for case in CASES for tag in case.covers}
    assert set(REQUIRED_COVERAGE) <= covered, set(REQUIRED_COVERAGE) - covered


def test_the_march_year_end_is_a_label_only():
    """The yen case's fiscal year ends in March; with December instead every
    figure is the same."""
    case = next(c for c in CASES if "March year-end" in c.covers)
    march = run(case)
    december = run(type(case)(**{**case.__dict__, "inputs": {
        **case.inputs, "fiscal_year_end_month": 12, "first_fiscal_year": None}}))
    for key in ("returns", "operating_model", "cash_flow", "debt_schedule"):
        assert december[key] == march[key]


def test_the_committed_workbook_is_current():
    assert is_current(), "reference_cases.xlsx is stale: run python -m tests.reference.workbook"
