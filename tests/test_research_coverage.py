"""Accepting a reviewed read and the checked workbook (PLAN.md R2), without
a database: ``research/coverage.py`` is pure. The road through the API, the
database and another account is tests/test_research_coverage_api.py.

Every read here replays recorded answers (tests/fixtures/research/reader).
"""
from __future__ import annotations

import copy
import datetime
import io

import pytest
from openpyxl import load_workbook

from research import coverage
from research.reader import checks
from tests.research_support import workbook_values
from tests.test_research_reader import BY_NAME, assembled

NOW = datetime.datetime(2026, 10, 15, 9, 30, tzinfo=datetime.timezone.utc)
AVENUE = "avenue-supermarts_FY27-Q2"


@pytest.fixture(scope="module")
def avenue() -> dict:
    return assembled(BY_NAME[AVENUE])


def everything_accepted(result: dict) -> dict:
    return {"accepted": [list(cell) for cell in sorted(coverage.amber_cells(result))], "corrected": []}


def written(period: dict, name: str = "Avenue Supermarts") -> object:
    data = coverage.workbook_bytes(name, {**period, "accepted_at": NOW}, generated=NOW, source="variater.com")
    return load_workbook(io.BytesIO(data))


def figures_by_sheet(wb, period: dict) -> list[tuple[str, int, int, float]]:
    """(sheet, row, column, stored figure) for every figure the period holds."""
    out = []
    sheets = [s for s in wb.sheetnames if s != "About"]
    for sheet, table in zip(sheets, period["figures"]["tables"]):
        for r, row in enumerate(table["rows"]):
            for c, value in enumerate(row["values"]):
                if value is not None:
                    out.append((sheet, coverage.FIRST_ROW + r, coverage.FIRST_COLUMN + c, value))
    return out


# ------------------------------------------------------------------ accepting

def test_nothing_is_accepted_while_an_amber_figure_is_unanswered(avenue):
    amber = coverage.amber_cells(avenue)
    assert len(amber) == avenue["summary"]["amber"] == 1
    with pytest.raises(coverage.Refused) as refused:
        coverage.accept(avenue, {"accepted": [], "corrected": []})
    assert (refused.value.code, refused.value.count) == ("amber_unreviewed", 1)
    # answered either way, it is accepted
    assert coverage.accept(avenue, everything_accepted(avenue))
    (t, r, c), = amber
    assert coverage.accept(avenue, {"corrected": [{"table": t, "row": r, "column": c, "value": 1.0}]})


def test_a_review_naming_a_figure_the_read_does_not_have_is_refused(avenue):
    for bad in ([99, 0, 0], [0, 999, 0], [0, 0, 99], [0, -1, 0], "x", [0, 0]):
        with pytest.raises(coverage.Refused) as refused:
            coverage.accept(avenue, {"accepted": [bad], **{"corrected": []}})
        assert refused.value.code == "bad_review"
    for value in (float("nan"), float("inf"), 1e16, True, "12"):
        with pytest.raises(coverage.Refused):
            coverage.accept(avenue, {**everything_accepted(avenue),
                                     "corrected": [{"table": 0, "row": 0, "column": 0, "value": value}]})


def test_an_accepted_read_is_one_period_per_scope_with_every_figure_and_its_page(avenue):
    periods = coverage.accept(avenue, everything_accepted(avenue))
    assert [(p["period_end"], p["months"], p["scope"], p["unit"], p["currency"]) for p in periods] == [
        ("2026-09-30", 3, "consolidated", "crores", "INR")]
    figures = periods[0]["figures"]
    assert [t["statement"] for t in figures["tables"]] == ["results", "balance_sheet", "cash_flow"]
    assert coverage.counts(figures) == {"figures": 332, "corrected": 0, "accepted": 1}
    revenue = next(r for r in figures["tables"][0]["rows"] if r["line"] == "revenue_from_operations")
    assert (revenue["values"][0], revenue["page"]) == (19644.01, 9)


def test_a_corrected_figure_is_the_one_stored_and_the_read_is_kept_beside_it(avenue):
    (t, r, c), = coverage.amber_cells(avenue)
    read = avenue["tables"][t]["rows"][r]["cells"][c]["value"]
    revenue = next(i for i, row in enumerate(avenue["tables"][0]["rows"]) if row["line"] == "revenue_from_operations")
    review = {"corrected": [{"table": t, "row": r, "column": c, "value": 123.45},
                            # a figure that was not amber can be corrected too, and blanked
                            {"table": 0, "row": revenue, "column": 1, "value": None}]}
    period, = coverage.accept(avenue, review)
    row = period["figures"]["tables"][t]["rows"][r]
    assert row["values"][c] == 123.45
    assert row["marks"][c] == {"do": "corrected", "read": read, "amber": "reads_differ"}
    blanked = period["figures"]["tables"][0]["rows"][revenue]
    assert blanked["values"][1] is None and blanked["marks"][1]["do"] == "corrected"
    assert coverage.counts(period["figures"])["corrected"] == 2
    # the read itself is untouched
    assert avenue["tables"][t]["rows"][r]["cells"][c]["value"] == read


def test_a_filing_with_both_scopes_gives_a_period_for_each(avenue):
    both = copy.deepcopy(avenue)
    standalone = copy.deepcopy(both["tables"][0])
    standalone["scope"] = "standalone"
    both["tables"].append(standalone)
    periods = coverage.accept(both, everything_accepted(both))
    assert [p["scope"] for p in periods] == ["consolidated", "standalone"]
    assert [t["statement"] for t in periods[1]["figures"]["tables"]] == ["results"]


def test_a_read_without_its_quarter_or_unit_is_refused_until_one_is_given(avenue):
    no_quarter = dict(avenue, quarter_end=None)
    with pytest.raises(coverage.Refused) as refused:
        coverage.accept(no_quarter, everything_accepted(avenue))
    assert refused.value.code == "quarter_unknown"
    no_unit = dict(avenue, unit=None)
    with pytest.raises(coverage.Refused) as refused:
        coverage.accept(no_unit, everything_accepted(avenue))
    assert refused.value.code == "unit_unknown"
    assert coverage.accept(no_unit, everything_accepted(avenue), unit="lakhs")[0]["unit"] == "lakhs"


def test_what_is_stored_feeds_the_next_reads_comparison_in_its_unit(avenue):
    period, = coverage.accept(avenue, everything_accepted(avenue))
    in_lakhs = coverage.stored_for_read([period], "lakhs")
    revenue = next(s for s in in_lakhs if s["line"] == "revenue_from_operations" and s["period_end"] == "2026-09-30"
                   and s["months"] == 3)
    assert revenue["value"] == pytest.approx(19644.01 * 100)
    assert coverage.stored_for_read([period], None) == []
    # the same filing read again agrees with what is kept; a changed figure is amber
    table = period["figures"]["tables"][0]
    stored = coverage.stored_for_read([period], "crores")
    assert checks.against_stored(table, "consolidated", stored) == []
    moved = [dict(s, value=s["value"] + 50) if s["line"] == "revenue_from_operations" else s for s in stored]
    assert checks.against_stored(table, "consolidated", moved)


# ------------------------------------------------------------------ the workbook

def test_the_workbook_recalculates_to_every_stored_figure(avenue):
    period, = coverage.accept(avenue, everything_accepted(avenue))
    wb = written(period)
    assert wb.sheetnames == ["About", "Results consolidated", "Balance sheet consolidated", "Cash flow consolidated"]
    formulas = 0
    for sheet, row, column, stored in figures_by_sheet(wb, period):
        raw = wb[sheet].cell(row, column).value
        formulas += isinstance(raw, str) and raw.startswith("=")
        worked = workbook_values(wb, sheet)(wb[sheet].cell(row, column).coordinate)
        assert worked == pytest.approx(stored, abs=1e-6), (sheet, row, column, raw)
    # the subtotals are formulas: dozens of them on this filing
    assert formulas >= 40


def test_subtotals_are_formulas_of_their_parts_and_figures_are_typed(avenue):
    period, = coverage.accept(avenue, everything_accepted(avenue))
    wb = written(period)
    ws = wb["Results consolidated"]
    rows = period["figures"]["tables"][0]["rows"]
    total = next(i for i, r in enumerate(rows) if r["line"] == "total_income")
    revenue = next(i for i, r in enumerate(rows) if r["line"] == "revenue_from_operations")
    total_cell = ws.cell(coverage.FIRST_ROW + total, coverage.FIRST_COLUMN)
    assert isinstance(total_cell.value, str) and total_cell.value.startswith("=")
    assert f"C{coverage.FIRST_ROW + revenue}" in total_cell.value
    assert ws.cell(coverage.FIRST_ROW + revenue, coverage.FIRST_COLUMN).value == 19644.01


def test_each_figure_has_its_page_in_a_note_and_a_correction_says_so(avenue):
    (t, r, c), = coverage.amber_cells(avenue)
    period, = coverage.accept(avenue, {"corrected": [{"table": t, "row": r, "column": c, "value": 123.45}]})
    wb = written(period)
    sheet = wb.sheetnames[1 + t]
    page = period["figures"]["tables"][t]["rows"][r]["page"]
    note = wb[sheet].cell(coverage.FIRST_ROW + r, coverage.FIRST_COLUMN + c).comment.text
    assert note.startswith(f"Page {page}") and "Corrected by the analyst" in note
    assert wb[sheet].cell(coverage.FIRST_ROW + r, coverage.FIRST_COLUMN + c).value == 123.45
    revenue = next(i for i, row in enumerate(period["figures"]["tables"][0]["rows"])
                   if row["line"] == "revenue_from_operations")
    plain = wb["Results consolidated"].cell(coverage.FIRST_ROW + revenue, coverage.FIRST_COLUMN).comment.text
    assert plain == "Page 9"


def test_a_corrected_figure_moves_the_formula_totals_with_it(avenue):
    """Correct a part of a subtotal: the subtotal's formula follows the new
    figure only if the printed total still holds, else the total stays printed."""
    period, = coverage.accept(avenue, everything_accepted(avenue))
    table = period["figures"]["tables"][0]
    sums, _ = coverage.subtotals(table)
    (target, column), parts = next(iter(sums.items()))
    part = parts[0][0]
    period["figures"]["tables"][0]["rows"][part]["values"][column] += 1000
    wb = written(period)
    ws = wb["Results consolidated"]
    cell = ws.cell(coverage.FIRST_ROW + target, coverage.FIRST_COLUMN + column)
    assert cell.value == table["rows"][target]["values"][column]     # printed: its parts no longer reach it
    assert "its parts add to" in cell.comment.text


def test_words_that_look_like_formulas_stay_words(avenue):
    period, = coverage.accept(avenue, everything_accepted(avenue))
    period["figures"]["tables"][0]["rows"][0]["label"] = "=HYPERLINK(\"http://x\")"
    wb = written(period, name="=1+1")
    ws = wb["Results consolidated"]
    assert ws.cell(coverage.FIRST_ROW, 1).data_type == "s"
    assert wb["About"]["B4"].value == "=1+1" and wb["About"]["B4"].data_type == "s"
    assert wb.properties.creator == "variater.com"
