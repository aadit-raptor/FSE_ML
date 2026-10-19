"""Quarterly actuals feeding the yearly model (PLAN.md R4): the model's past
from the periods an analyst accepted in coverage (R2), in the shape
`research/coverage.py` `accept` stores them.
"""
from __future__ import annotations

import pytest

from research import actuals
from research.models import occupancy
from research.definition import Company
from research.evaluate import evaluate
from tests import research_support as support


def table(statement, columns, rows):
    """columns: [(label, period_end, months)]; rows: [(line, page, values)]."""
    return {"statement": statement, "pages": [1],
            "columns": [{"label": c[0], "period_end": c[1], "months": c[2]} for c in columns],
            "rows": [{"label": line or "x", "line": line, "page": page, "values": list(values)} for line, page, values in rows]}


def results(columns, revenue, materials=None, finance=None, tax=None, other=None):
    n = len(columns)
    rows = [("revenue_from_operations", 3, revenue), ("other_income", 3, other or [10.0] * n),
            ("cost_of_materials", 3, materials or [100.0] * n), ("changes_in_inventories", 3, [-5.0] * n),
            ("employee_benefits", 3, [200.0] * n), ("finance_costs", 3, finance or [30.0] * n),
            ("depreciation_amortisation", 3, [40.0] * n), ("other_expenses", 3, [50.0] * n),
            ("profit_before_tax", 3, [300.0] * n), ("current_tax", 3, [60.0] * n), ("deferred_tax", 3, [5.0] * n),
            ("net_profit", 3, [235.0] * n), (None, 3, [1.0] * n)]
    if tax is not None:
        rows.append(("tax_total", 3, tax))
    return table("results", columns, rows)


def period(tables, unit="lakhs", scope="consolidated", currency="INR", year_end_month=3):
    return {"scope": scope, "unit": unit, "currency": currency,
            "figures": {"format": 1, "year_end_month": year_end_month, "tables": tables}}


H1 = [("Q2 FY27", "2026-09-30", 3), ("Q1 FY27", "2026-06-30", 3), ("Q2 FY26", "2025-09-30", 3),
      ("H1 FY27", "2026-09-30", 6), ("H1 FY26", "2025-09-30", 6), ("FY26", "2026-03-31", 12)]
BALANCE = table("balance_sheet", [("30 Sep 2026", "2026-09-30", None), ("31 Mar 2026", "2026-03-31", None)], [
    ("inventories", 7, [80.0, 70.0]), ("trade_receivables", 7, [90.0, 85.0]),
    ("cash_and_cash_equivalents", 7, [400.0, 350.0]), ("trade_payables", 8, [60.0, 55.0])])


def test_a_half_years_filing_gives_the_year_to_date_and_last_year():
    fill = actuals.fill([period([results(H1, [1000, 900, 950, 1900, 1800, 3800]), BALANCE])])
    assert fill.labels == {"ytd": "6 months to 2026-09-30", "ly": "12 months to 2026-03-31"}
    assert fill.values["rev"] == {"ytd": 1900, "ly": 3800}
    assert fill.values["days"] == {"ytd": 183, "ly": 365}                 # 1 April to 30 September; FY26
    assert fill.values["mat"] == {"ytd": 95, "ly": 95}                    # materials and the change in inventories
    assert fill.values["tax"] == {"ytd": 65, "ly": 65}                    # current and deferred, with no total printed
    assert fill.values["int_debt"]["ly"] == 30 and actuals.LEASES_IN_FINANCE_COSTS in fill.notes
    assert fill.values["inv"] == {"ytd": 80, "ly": 70} and fill.values["pay"] == {"ytd": 60, "ly": 55}
    assert fill.sources["inv"]["ly"] == 7 and fill.sources["rev"]["ytd"] == 3
    assert fill.unit == "lakhs" and fill.unit_rupees == 100_000 and fill.currency == "INR"
    assert "s1_occ" not in fill.values and "pwr" not in fill.values      # drivers and unsplit lines stay to type


def test_a_printed_tax_total_wins_over_its_parts():
    fill = actuals.fill([period([results(H1, [1] * 6, tax=[70.0] * 6)])])
    assert fill.values["tax"]["ytd"] == 70
    assert actuals.NO_BALANCE_SHEET in fill.notes


def test_without_a_cumulative_column_the_years_quarters_are_added_up():
    q1 = period([results([("Q1 FY27", "2026-06-30", 3), ("FY26", "2026-03-31", 12)], [900, 3800])])
    q2 = period([results([("Q2 FY27", "2026-09-30", 3), ("Q2 FY26", "2025-09-30", 3)], [1000, 950],
                         materials=[120.0, 90.0])])
    fill = actuals.fill([q1, q2])
    assert fill.values["rev"] == {"ytd": 1900, "ly": 3800}
    assert fill.values["mat"]["ytd"] == pytest.approx(100 - 5 + 120 - 5)
    assert fill.values["days"]["ytd"] == 183


def test_a_missing_quarter_leaves_the_year_to_date_blank_and_says_so():
    q2 = period([results([("Q2 FY27", "2026-09-30", 3), ("FY26", "2026-03-31", 12)], [1000, 3800])])
    fill = actuals.fill([q2])
    assert actuals.QUARTER_MISSING in fill.notes
    assert fill.values["rev"] == {"ly": 3800}


def test_a_year_end_filing_is_last_year_with_nothing_to_date():
    year = period([results([("Q4 FY26", "2026-03-31", 3), ("FY26", "2026-03-31", 12)], [1000, 3800])])
    fill = actuals.fill([year])
    assert actuals.YEAR_COMPLETE in fill.notes
    assert fill.values["rev"] == {"ly": 3800} and "ytd" not in fill.labels


def test_money_is_brought_to_the_newest_filings_unit_and_another_currency_is_left_out():
    old = period([results([("Q1 FY27", "2026-06-30", 3), ("FY26", "2026-03-31", 12)], [9, 38])], unit="crores")
    new = period([results([("Q2 FY27", "2026-09-30", 3), ("H1 FY27", "2026-09-30", 6)], [1000, 1900])], unit="lakhs")
    fill = actuals.fill([old, new])
    assert fill.values["rev"] == {"ytd": 1900, "ly": 3800}               # 38 crore is 3,800 lakh
    dollars = period([results([("FY26", "2026-03-31", 12)], [5])], currency="USD")
    fill = actuals.fill([new, dollars])
    assert actuals.OTHER_CURRENCY in fill.notes and actuals.NO_LAST_YEAR in fill.notes


def test_only_the_scope_asked_for_is_read():
    alone = period([results(H1, [1] * 6)], scope="standalone")
    assert actuals.fill([alone]).notes == [actuals.NOTHING_ACCEPTED]
    assert actuals.fill([alone], "standalone").values["rev"]["ly"] == 1


def test_without_a_known_year_end_the_longest_column_at_the_filings_date_says_how_far_it_reaches():
    fill = actuals.fill([period([results(H1, [1000, 900, 950, 1900, 1800, 3800])], year_end_month=None)])
    assert fill.values["rev"] == {"ytd": 1900, "ly": 3800}
    alone = period([results([("Q2 FY27", "2026-09-30", 3), ("FY26", "2026-03-31", 12)], [1000, 3800])],
                   year_end_month=None)
    assert actuals.fill([alone]).values["rev"] == {"ytd": 1000}       # read as a first quarter: no year before it


def test_a_52_week_year_ending_within_a_week_of_the_month_end_still_counts_as_last_year():
    filing = period([results([("Q1", "2026-06-30", 3), ("FY", "2026-03-28", 12)], [900, 3800])])
    assert actuals.fill([filing]).values["rev"]["ly"] == 3800


def test_the_fill_goes_into_model_1_and_its_past_works_out():
    fill = actuals.fill([period([results(H1, [1000, 900, 950, 1900, 1800, 3800]), BALANCE])])
    company = Company("Filled", "Rs lakh", sector="hotels", constants={"unit_rupees": fill.unit_rupees},
                      choices={"line1_past": "revenue", "line1_end_to_date": "average", **support.VALUE_CHOICES},
                      values=fill.values)
    v = evaluate(occupancy.MODEL, company, partial=True).values
    assert v["rev"]["ly"] == 3800 and v["ebitda"]["ly"] == pytest.approx(3800 - 95 - 200 - 50)
    assert v["pbt"]["ly"] == pytest.approx(3800 - 95 - 200 - 50 - 40 + 10 - 30)
    assert v["chk_pbt"]["ly"] == pytest.approx(v["pbt"]["ly"] - 300)    # the filing's own total to check against


def test_periods_without_a_results_table_fill_nothing_rather_than_fail():
    alone = period([BALANCE])
    assert actuals.fill([alone]).notes == [actuals.NOTHING_ACCEPTED]


def test_a_52_week_quarter_ending_in_a_months_first_week_counts_as_the_month_before():
    filing = period([results([("Q1", "2026-07-02", 3), ("FY", "2026-03-28", 12)], [900, 3800])])
    fill = actuals.fill([filing])
    assert fill.values["rev"] == {"ytd": 900, "ly": 3800}
