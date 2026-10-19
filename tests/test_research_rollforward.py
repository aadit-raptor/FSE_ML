"""Rolling a saved sector model forward on new results (PLAN.md R5): the
model and its assumptions kept, the past moved on, and the estimate made
before the results set beside them, worked out by hand from the model's own
answer before the roll.

The model is Yatharth Hospital's, solved by hand in R4 (three months of FY27
reported); the results are built in the shape research/coverage.py stores.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import actuals, compose
from research.definition import ASSUME, FIGURE, Company, cells
from research.evaluate import evaluate
from research.models import CATALOGUE
from research.rollforward import (
    AS_REPORTED, NEW_YEAR, NO_AS_OF, NO_RESULTS, NOTHING_NEWER, OLDER_RESULTS, SHARED_BY_DAYS, TOO_FAR,
    WITHIN_YEAR, RollRefused, newer, roll,
)
from tests.test_research_actuals import period, results, table

EXAMPLES = json.loads((Path(compose.__file__).parent / "models" / "examples.json").read_text(encoding="utf-8"))
AS_OF = {"ly_end": "2026-03-31", "ytd_end": "2026-06-30", "ytd_months": 3}


def yatharth():
    ex = next(e for e in EXAMPLES if e["id"] == "yatharth-hospital")
    model = CATALOGUE.build(compose.read(ex["structure"], CATALOGUE.specs))
    raw = dict(ex["company"], off=tuple(ex["company"]["off"]))
    return model, Company(**raw)


def half_year(model, company):
    """Six months to 30 September 2026: the year to date grows by a quarter's revenue of 4,100."""
    old_rev = company.values["rev"]["ytd"]
    columns = [("Q2 FY27", "2026-09-30", 3), ("H1 FY27", "2026-09-30", 6), ("FY26", "2026-03-31", 12)]
    balance = table("balance_sheet", [("30 Sep 2026", "2026-09-30", None), ("31 Mar 2026", "2026-03-31", None)],
                    [("inventories", 7, [900.0, 850.0]), ("trade_receivables", 7, [4000.0, 3900.0])])
    return actuals.fill([period([results(columns, [4100.0, old_rev + 4100.0, 12071.72]), balance], unit="millions")])


def year_end():
    """The year to 31 March 2027, reported whole."""
    columns = [("Q4 FY27", "2027-03-31", 3), ("FY27", "2027-03-31", 12), ("FY26", "2026-03-31", 12)]
    balance = table("balance_sheet", [("31 Mar 2027", "2027-03-31", None), ("31 Mar 2026", "2026-03-31", None)],
                    [("inventories", 7, [990.0, 850.0]), ("cash_and_cash_equivalents", 7, [2500.0, 2000.0])])
    return actuals.fill([period([results(columns, [4500.0, 17000.0, 12071.72]), balance], unit="millions")])


# ---------------------------------------------------------------------------- a later quarter of the same year
def test_a_later_quarter_keeps_every_assumption_and_moves_the_year_to_date_on():
    model, company = yatharth()
    fill = half_year(model, company)
    rolled = roll(model, company, AS_OF, fill)
    new = rolled.company
    assert rolled.kind == WITHIN_YEAR and rolled.as_of == {"ly_end": "2026-03-31", "ytd_end": "2026-09-30",
                                                          "ytd_months": 6}
    assert new.values["rev"]["ytd"] == pytest.approx(company.values["rev"]["ytd"] + 4100)
    assert new.values["inv"]["ytd"] == 900
    # the year keeps its length: 91 + 274 days, now 183 reported and 182 to come
    assert new.values["days"]["ytd"] == 183 and new.values["days"]["roy"] == 91 + 274 - 183
    plan = cells(model, company, allow_missing=True)
    for (row, p), cell in plan.items():
        if cell.typed is ASSUME and not (row, p) == ("days", "roy"):
            assert new.values.get(row, {}).get(p) == company.values.get(row, {}).get(p), (row, p)
    assert new.period_labels["ytd"] == "6 months to 2026-09-30" and new.period_labels["y1"] == "FY28"


def test_a_later_quarter_carries_counts_marked_and_leaves_flows_to_type():
    model, company = yatharth()
    fill = half_year(model, company)
    rolled = roll(model, company, AS_OF, fill)
    new = rolled.company
    # the occupancy and the price of three months are kept, amber, until the analyst types six months'
    assert ("s1_occ", "ytd") in rolled.carried and new.values["s1_occ"]["ytd"] == company.values["s1_occ"]["ytd"]
    assert new.marks["s1_occ"]["ytd"]["level"] == "amber"
    assert "quarter before" in new.marks["s1_occ"]["ytd"]["why"]
    # the second line's revenue over three months is not six months' revenue: blank, and listed
    assert ("s2_rev", "ytd") in rolled.to_type and "ytd" not in new.values.get("s2_rev", {})
    # a figure the results gave carries no mark
    assert "ytd" not in new.marks.get("rev", {})
    for row, p in rolled.carried + rolled.to_type:
        assert row not in fill.values or p not in fill.values[row]


def test_a_later_quarter_sets_the_months_reported_beside_the_rest_of_the_years_estimate_shared_by_days():
    model, company = yatharth()
    before = evaluate(model, company, partial=True).values
    rolled = roll(model, company, AS_OF, half_year(model, company))
    c = rolled.comparison
    assert c["basis"] == SHARED_BY_DAYS and c["months"] == 3 and c["period"] == "3 months to 2026-09-30"
    line = next(x for x in c["lines"] if x["row"] == "rev")
    # by hand: the rest of the year's revenue, 92 of its 274 days
    estimate = before["rev"]["roy"] * (183 - 91) / 274
    assert line["estimate"] == pytest.approx(estimate, abs=1e-6)
    assert line["actual"] == pytest.approx(4100, abs=1e-6)
    assert line["difference"] == pytest.approx(4100 - estimate, abs=1e-6)
    assert line["share"] == pytest.approx(4100 / estimate - 1, abs=1e-6)
    assert line["label"] == "Revenue from operations" and line["kind"] == "flow"
    assert {x["row"] for x in c["lines"]} <= set(actuals.FLOWS)


def test_the_rolled_model_works_out_its_current_year_from_the_new_parts():
    model, company = yatharth()
    new = roll(model, company, AS_OF, half_year(model, company)).company
    after = evaluate(model, new, partial=True).values
    assert after["rev"]["cy"] == pytest.approx(after["rev"]["ytd"] + after["rev"]["roy"])
    assert after["rev"]["ytd"] == pytest.approx(company.values["rev"]["ytd"] + 4100)


# ---------------------------------------------------------------------------- a new year
def test_a_new_year_moves_every_assumption_back_a_year_and_repeats_the_last_marked():
    model, company = yatharth()
    rolled = roll(model, company, AS_OF, year_end())
    new = rolled.company
    assert rolled.kind == NEW_YEAR and rolled.as_of == {"ly_end": "2027-03-31", "ytd_end": None, "ytd_months": 0}
    v, old = new.values, company.values
    assert v["mat_sh"]["roy"] == old["mat_sh"]["y1"] and v["mat_sh"]["y1"] == old["mat_sh"]["y2"]
    assert v["capex"]["cy"] == old["capex"]["y1"] and v["capex"]["y1"] == old["capex"]["y2"]
    assert v["mat_sh"]["y2"] == old["mat_sh"]["y2"] and new.marks["mat_sh"]["y2"]["level"] == "amber"
    # FY28 runs from 1 April 2027 to 31 March 2028, a leap year, and nothing of it is reported
    assert v["days"]["roy"] == 366 and v["days"]["ytd"] == 0
    assert v["rev"]["ly"] == 17000 and "ytd" not in v["rev"]
    assert new.period_labels["ly"] == "12 months to 2027-03-31" and "ytd" not in new.period_labels
    assert new.period_labels["cy"] == "FY28" and new.period_labels["y1"] == "FY29" and "y2" not in new.period_labels
    assert new.period_labels["roy"] == "Rest of FY28 (assumed)"


def test_a_new_year_fills_last_years_drivers_with_the_models_estimate_marked_and_starts_counts_where_they_ended():
    model, company = yatharth()
    before = evaluate(model, company, partial=True).values
    rolled = roll(model, company, AS_OF, year_end())
    new = rolled.company
    assert new.values["s1_occ"]["ly"] == pytest.approx(before["s1_occ"]["cy"])
    assert ("s1_occ", "ly") in rolled.carried and "estimate" in new.marks["s1_occ"]["ly"]["why"]
    # nothing of the new year is reported: a count stands at the year's end, unmarked
    assert new.values["s2_end"]["ytd"] == new.values["s2_end"]["ly"] and "ytd" not in new.marks.get("s2_end", {})
    after = evaluate(model, new, partial=True).values
    assert after["s2_open"]["roy"] == new.values["s2_end"]["ly"]


def test_a_new_year_compares_the_old_current_year_with_the_year_as_reported_balance_sheet_too():
    model, company = yatharth()
    before = evaluate(model, company, partial=True).values
    c = roll(model, company, AS_OF, year_end()).comparison
    assert c["basis"] == AS_REPORTED and c["months"] == 12 and c["estimate_label"] == "FY27"
    rev = next(x for x in c["lines"] if x["row"] == "rev")
    assert rev["estimate"] == pytest.approx(before["rev"]["cy"], abs=1e-6) and rev["actual"] == 17000
    inv = next(x for x in c["lines"] if x["row"] == "inv")
    assert inv["kind"] == "stock" and inv["estimate"] == pytest.approx(before["inv"]["cy"], abs=1e-6)
    assert inv["actual"] == 990


# ---------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("as_of, code", [
    (None, NO_AS_OF),
    ({"ly_end": "2026-03-31", "ytd_end": "2026-09-30", "ytd_months": 6}, NOTHING_NEWER),
    ({"ly_end": "2026-03-31", "ytd_end": "2026-12-31", "ytd_months": 9}, OLDER_RESULTS),
    ({"ly_end": "2027-03-31", "ytd_end": None, "ytd_months": 0}, OLDER_RESULTS),
    ({"ly_end": "2024-03-31", "ytd_end": "2024-06-30", "ytd_months": 3}, TOO_FAR),
])
def test_a_roll_that_would_not_move_the_model_on_is_refused(as_of, code):
    model, company = yatharth()
    fill = half_year(model, company)
    with pytest.raises(RollRefused) as refused:
        roll(model, company, as_of, fill)
    assert refused.value.code == code
    assert not newer(as_of, fill.as_of)


def test_a_quarter_whose_year_to_date_cannot_be_put_together_is_refused():
    model, company = yatharth()
    q2_alone = actuals.fill([period([results([("Q2 FY27", "2026-09-30", 3), ("FY26", "2026-03-31", 12)],
                                             [4100.0, 12071.72])], unit="millions")])
    assert newer(AS_OF, q2_alone.as_of)
    with pytest.raises(RollRefused) as refused:
        roll(model, company, AS_OF, q2_alone)
    assert refused.value.code == NO_RESULTS


def test_the_fill_says_how_far_the_accepted_results_reach():
    model, company = yatharth()
    assert half_year(model, company).as_of == {"ly_end": "2026-03-31", "ytd_end": "2026-09-30", "ytd_months": 6}
    assert year_end().as_of == {"ly_end": "2027-03-31", "ytd_end": None, "ytd_months": 0}
    assert actuals.fill([]).as_of is None
    assert newer(AS_OF, year_end().as_of)


def test_every_figure_the_roll_leaves_is_one_the_model_takes():
    model, company = yatharth()
    for fill in (half_year(model, company), year_end()):
        new = roll(model, company, AS_OF, fill).company
        plan = cells(model, new, allow_missing=True)
        for row, by in new.values.items():
            for p in by:
                assert plan[(row, p)].typed in (FIGURE, ASSUME)


def test_results_in_another_unit_come_into_the_models_unit():
    model, company = yatharth()                        # typed in millions
    old_rev = company.values["rev"]["ytd"]
    columns = [("Q2 FY27", "2026-09-30", 3), ("H1 FY27", "2026-09-30", 6), ("FY26", "2026-03-31", 12)]
    in_crore = [410.0, (old_rev + 4100.0) / 10, 1207.172]
    fill = actuals.fill([period([results(columns, in_crore)], unit="crores")])
    rolled = roll(model, company, AS_OF, fill)
    assert rolled.company.values["rev"]["ytd"] == pytest.approx(old_rev + 4100)
    assert rolled.company.values["days"]["ytd"] == 183
    assert next(x for x in rolled.comparison["lines"] if x["row"] == "rev")["actual"] == pytest.approx(4100)
