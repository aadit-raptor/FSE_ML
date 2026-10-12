"""One model definition, two renderings (PLAN.md phase R, rule 3): the
workbook the writer makes holds the same rows as live Excel formulas, and
gives the evaluator's figures on every row.

The workbook is worked out here by `tests/research_support.py`'s own reader of
Excel formulas, which shares no code with `research/`, and by LibreOffice
where it is installed (CI sets ``FSE_REQUIRE_SOFFICE=1`` once the workflow
installs it; see the bottom of this file).
"""
import re

import pytest

from research.definition import ASSUME, Company, InputError, cells
from research.evaluate import evaluate
from research.models import occupancy
from research.workbook import layout, write
from tests import research_support as support
from tests.test_research_model import toy, yatharth

TOLERANCE = 0.01


def model1(slug, **changes):
    return occupancy.MODEL, support.company(support.load(slug), **changes)


CASES = {
    "toy": lambda: (toy(), yatharth()),
    "hospitals": lambda: model1("hospitals"),
    "hotels": lambda: model1("hotels"),
}


def differences(model, company, get):
    """Every cell where the workbook and the evaluator part ways."""
    where, values, out = layout(model), evaluate(model, company).values, []
    for row in model.data_rows():
        for period in model.periods:
            ours, theirs = values[row.key][period.name], get(where.cell(row.key, period.name))
            if (ours is None) != (theirs is None) or (ours is not None and abs(ours - theirs) > TOLERANCE):
                out.append((row.key, period.name, ours, theirs))
    return out


@pytest.mark.parametrize("case", CASES)
def test_the_workbook_agrees_with_the_evaluator_on_every_row(case):
    model, company = CASES[case]()
    wb = write(model, company)
    assert differences(model, company, support.workbook_values(wb)) == []
    assert sum(1 for row in model.data_rows() for _ in model.periods) >= 18


@pytest.mark.parametrize("case", CASES)
def test_every_worked_out_cell_is_a_live_formula_and_every_typed_cell_a_number(case):
    model, company = CASES[case]()
    ws, where = write(model, company)["Model"], layout(model)
    plan = cells(model, company)
    formulas = 0
    for row in model.data_rows():
        for period in model.periods:
            cell, held = plan.get((row.key, period.name)), ws[where.cell(row.key, period.name)].value
            if cell is None:
                assert held is None
            elif cell.tree is not None:
                assert isinstance(held, str) and held.startswith("="), (row.key, period.name, held)
                formulas += 1
            else:
                assert held is None or isinstance(held, (int, float)), (row.key, period.name, held)
    assert formulas > (200 if case != "toy" else 5)


@pytest.mark.parametrize("case", CASES)
def test_the_workbook_holds_plain_arithmetic_and_nothing_else(case):
    model, company = CASES[case]()
    for row in write(model, company)["Model"].iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and cell.value.startswith("="):
                assert set(re.findall(r"[A-Z]{2,}(?=\()", cell.value)) <= {"MAX", "MIN", "IF"}, cell.value
                assert "!" not in cell.value and "[" not in cell.value, cell.value


def test_changing_an_assumption_cell_moves_the_projection_with_no_server():
    """Type a new occupancy into the workbook itself: its own formulas give
    what the server gives for that assumption."""
    model, company = model1("hotels")
    wb, where = write(model, company), layout(model)
    before = support.workbook_values(wb)(where.cell("pat_own", "y1"))
    wb["Model"][where.cell("s1_occ", "y1")].value = 0.80
    after = support.workbook_values(wb)
    changed = model1("hotels", s1_occ={"y1": 0.80})[1]
    assert differences(model, changed, after) == []
    assert after(where.cell("pat_own", "y1")) > before + 1000


def test_every_assumption_cell_has_a_name_and_so_does_every_constant():
    model, company = model1("hospitals")
    wb, where, plan = write(model, company), layout(model), cells(model, company)
    names = {name: wb.defined_names[name].attr_text for name in wb.defined_names}
    assert names["unit_rupees"] == "Model!" + absolute(where.constant("unit_rupees"))
    assumed = [(r, p) for (r, p), cell in plan.items() if cell.typed is ASSUME]
    assert len(assumed) > 90
    for row, period in assumed:
        assert names["%s_%s" % (row, period)] == "Model!" + absolute(where.cell(row, period)), (row, period)
    assert len(names) == len(assumed) + len(model.constants)        # and nothing else is named
    assert wb["Model"][where.constant("unit_rupees")].value == 1_000_000
    for name in names:                                              # a name Excel could take for a cell is refused
        assert not re.fullmatch(r"[A-Za-z]{1,3}[0-9]+", name)


def absolute(address):
    return re.sub(r"([A-Z]+)([0-9]+)", r"$\1$\2", address)


def test_the_sheet_says_what_it_is():
    model, company = model1("hotels")
    wb = write(model, company)
    ws = wb["Model"]
    top = " ".join(str(c.value) for row in ws.iter_rows(max_row=3) for c in row if c.value)
    assert "Lemon Tree Hotels Limited" in top and "NOT YET VETTED" in top and "Rs lakh" in top
    labels = [c.value for c in ws["A"] if isinstance(c.value, str)]
    assert "Average room rate, Rs per night" in labels                      # the company's own words
    assert "Owned and leased rooms: at the start" in labels
    assert not any("{" in label for label in labels)
    header = [c.value for c in ws[layout(model).header_row]]
    assert "H1 FY26 (actual)" in header and "FY28" in header
    notes = " ".join(str(c.value) for row in wb["Notes"].iter_rows() for c in row if c.value)
    assert "Deferred tax is left out" in notes
    assert wb.properties.creator == "variater.com"


def test_a_marked_figure_is_coloured_and_listed_with_its_reason():
    model, company = model1("hospitals")
    wb, where = write(model, company), layout(model)
    cell = wb["Model"][where.cell("s1_occ", "ly")]
    assert cell.fill.fgColor.rgb.endswith("F2A097")                         # red: one press report only
    assert "nine months" in cell.comment.text
    notes = [[c.value for c in row] for row in wb["Notes"].iter_rows()]
    assert any(r[0] == "Model!" + where.cell("s1_occ", "ly") and r[2] == "red" for r in notes)
    plain = wb["Model"][where.cell("rev", "ly")]
    assert plain.comment is None and not plain.fill.fgColor.rgb.endswith("F2A097")


def test_words_that_start_with_an_equals_sign_stay_words():
    """A company name, label or reason is typed by a person: none may turn into a formula."""
    model, company = model1("hotels")
    company.name = "=1+1"
    company.unit_name = "=2+2"
    company.basis = "=3+3"
    company.words = {**company.words, "occ": "=HYPERLINK(\"http://x\")", "s1": "=4+4"}
    company.period_labels = {**company.period_labels, "y1": "=5+5"}
    company.marks = {"rev": {"ly": {"level": "amber", "why": "=6+6"}}}
    wb, where, plan = write(model, company), layout(model), cells(model, company)
    worked_out = {where.cell(r, p) for (r, p), cell in plan.items() if cell.tree is not None}
    formulas = {c.coordinate for row in wb["Model"].iter_rows() for c in row if c.data_type == "f"}
    assert formulas == worked_out
    assert [c for row in wb["Notes"].iter_rows() for c in row if c.data_type == "f"] == []
    assert wb["Model"]["A1"].value.startswith("=1+1")
    assert any(c.value == "=6+6" for row in wb["Notes"].iter_rows() for c in row)


def test_words_must_fill_every_slot():
    model, company = model1("hotels")
    words = {k: v for k, v in company.words.items() if k != "s2_rpu"}
    bare = Company(name="X", sector="", unit_name="u", constants=company.constants, choices=company.choices,
                   values=company.values, words=words)
    with pytest.raises(InputError):
        write(model, bare)


def test_a_sector_brings_its_own_words():
    """Pick a sector and the labels follow, with no words typed at all."""
    model, company = model1("hospitals")
    plain = Company(name="X", sector="hospitals", unit_name="Rs million", constants=company.constants,
                    choices=company.choices, values=company.values, off=company.off)
    labels = [c.value for c in write(model, plain)["Model"]["A"] if isinstance(c.value, str)]
    assert "Beds: at the start" in labels and "Occupancy" in labels
    hotel = model1("hotels")[1]
    plain = Company(name="X", sector="hotels", unit_name="Rs lakh", constants=hotel.constants, choices=hotel.choices,
                    values=hotel.values)
    labels = [c.value for c in write(model, plain)["Model"]["A"] if isinstance(c.value, str)]
    assert "Rooms: at the start" in labels and "Average room rate, Rs per night" in labels


# ---------------------------------------------------------------------------
# A real spreadsheet engine. LibreOffice opens the workbook, works every
# formula out and saves the values; they must be the evaluator's. The test
# exists only where LibreOffice is installed or CI demands it: the `ml` job
# fails on any skipped test, so it is left out, not skipped, elsewhere.
# ---------------------------------------------------------------------------
if support.soffice_wanted():
    @pytest.mark.parametrize("case", CASES)
    def test_libreoffice_works_the_workbook_out_to_the_evaluators_figures(case):
        model, company = CASES[case]()
        recalculated = support.recalculated_by_libreoffice(write(model, company))["Model"]
        found = differences(model, company, lambda address: recalculated[address].value)
        assert found == []
        where = layout(model)
        sample = next(r for r in model.data_rows() if r.bold)
        assert recalculated[where.cell(sample.key, model.periods[-1].name)].value is not None
