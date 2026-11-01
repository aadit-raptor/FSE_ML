"""Portfolio tracking, the arithmetic (PLAN.md 7.4): quarterly results against the plan, drift alerts, the
reforecast from the last four quarters and conversion to a reporting currency at stated ECB rates.

Every figure here is worked out by hand from the plan's own years, so the module is checked against arithmetic,
never against itself. tests/test_portfolio_api.py runs the same through the API on a real database.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core import portfolio
from core.config import DEFAULTS
from core.deal import DealInputs
from core.plan_actual import _run

DEAL = DealInputs()                     # the default deal: EBITDA 100, growth 5%, hold 5, in millions
CFG = dict(DEFAULTS)


def plan():
    return portfolio.plan_path(_run(DEAL, CFG))


def quarters(rows):
    return portfolio.quarters_from(rows)


def on_plan(year: int, upto: int, share: float = 1.0, debt: float | None = None) -> list[dict]:
    """Quarters of ``year`` 1..``upto`` whose revenue and EBITDA are ``share`` of the plan's quarter."""
    p = plan()
    return [{"year": year, "quarter": q, "revenue": p.revenue[year - 1] / 4 * share,
             "ebitda": p.ebitda[year - 1] / 4 * share, "total_debt": debt} for q in range(1, upto + 1)]


# ------------------------------------------------------------------------------------------------ quarters
def test_quarters_are_sorted_and_a_quarter_given_twice_is_refused():
    got = quarters([{"year": 2, "quarter": 1}, {"year": 1, "quarter": 4}])
    assert [(q.year, q.quarter) for q in got] == [(1, 4), (2, 1)]
    with pytest.raises(portfolio.QuartersRefused) as refused:
        quarters([{"year": 1, "quarter": 2}, {"year": 1, "quarter": 2}])
    assert refused.value.code == "duplicate_quarter"


def test_a_quarter_after_the_hold_is_refused():
    with pytest.raises(portfolio.QuartersRefused) as refused:
        portfolio.check_hold(quarters([{"year": DEAL.hold + 1, "quarter": 1}]), DEAL.hold)
    assert refused.value.code == "beyond_hold"


def test_quarters_in_another_unit_are_converted_to_millions():
    got = portfolio.quarters_in_millions(quarters([{"year": 1, "quarter": 1, "ebitda": 2500.0}]), "thousands")
    assert got[0].figures["ebitda"] == pytest.approx(2.5)


def test_full_years_sum_flows_and_take_the_fourth_quarters_debt():
    rows = [{"year": 1, "quarter": q, "revenue": 10.0 * q, "ebitda": 2.0, "total_debt": 100.0 - q} for q in (1, 2, 3, 4)]
    rows += [{"year": 2, "quarter": 1, "revenue": 50.0}]                 # year 2 is not complete
    years = portfolio.full_years(quarters(rows))
    assert len(years) == 1
    assert years[0]["revenue"] == pytest.approx(100.0) and years[0]["ebitda"] == pytest.approx(8.0)
    assert years[0]["total_debt"] == pytest.approx(96.0)
    assert years[0]["net_income"] is None                               # not given in every quarter


# ------------------------------------------------------------------------------------------------ drift
def test_results_on_plan_raise_no_alert():
    drift = portfolio.drift(plan(), quarters(on_plan(2, 3)), 5.0)
    assert (drift["year"], drift["quarter"]) == (2, 3)
    assert drift["alerts"] == []
    p = plan()
    # year to date: three quarters of year 2's plan
    assert drift["ytd"]["ebitda"]["plan"] == pytest.approx(p.ebitda[1] * 3 / 4)
    assert drift["ytd"]["ebitda"]["gap_pct"] == pytest.approx(0.0, abs=1e-9)


def test_ebitda_below_plan_beyond_the_tolerance_raises_an_alert():
    p = plan()
    rows = on_plan(2, 2)
    for row in rows:
        row["ebitda"] = p.ebitda[1] / 4 * 0.90                           # 10% below, revenue on plan
    drift = portfolio.drift(p, quarters(rows), 5.0)
    assert [a["id"] for a in drift["alerts"]] == ["ebitda_below_plan"]
    alert = drift["alerts"][0]
    assert alert["actual"] == pytest.approx(p.ebitda[1] / 2 * 0.90)
    assert alert["plan"] == pytest.approx(p.ebitda[1] / 2)
    assert alert["gap_pct"] == pytest.approx(-10.0)


def test_a_shortfall_inside_the_tolerance_is_not_an_alert():
    drift = portfolio.drift(plan(), quarters(on_plan(1, 4, share=0.97)), 5.0)
    assert drift["alerts"] == []
    assert drift["ytd"]["revenue"]["gap_pct"] == pytest.approx(-3.0)
    # a tighter tolerance makes the same results an alert
    assert {a["id"] for a in portfolio.drift(plan(), quarters(on_plan(1, 4, share=0.97)), 2.0)["alerts"]} == {
        "revenue_below_plan", "ebitda_below_plan"}


def test_debt_is_compared_with_the_plan_part_way_through_the_year():
    p = plan()
    # half way through year 2: the plan's debt is half way between the end of year 1 and of year 2
    expected = p.debt[0] + (p.debt[1] - p.debt[0]) * 2 / 4
    drift = portfolio.drift(p, quarters(on_plan(2, 2, debt=expected * 1.10)), 5.0)
    assert drift["debt"]["plan"] == pytest.approx(expected)
    assert [a["id"] for a in drift["alerts"]] == ["debt_above_plan"]
    assert drift["alerts"][0]["gap_pct"] == pytest.approx(10.0)
    # year 1's first quarter starts from the debt at close
    first = portfolio.drift(p, quarters(on_plan(1, 1, debt=p.debt_at_close)), 5.0)
    assert first["debt"]["plan"] == pytest.approx(p.debt_at_close + (p.debt[0] - p.debt_at_close) / 4)


def test_a_year_with_a_missing_quarter_has_no_year_to_date():
    rows = on_plan(2, 3)
    del rows[1]                                                           # Q2 missing
    drift = portfolio.drift(plan(), quarters(rows), 5.0)
    assert drift["ytd"]["ebitda"] is None and drift["alerts"] == []


# ------------------------------------------------------------------------------------------------ reforecast
def test_reforecast_growth_passes_through_the_actual_year():
    p = plan()
    rows = on_plan(2, 4, share=0.8) + on_plan(1, 4)                      # year 2 came in 20% below plan
    got = portfolio.reforecast(DEAL, CFG, quarters(rows), n=2000)
    # (1 + g') = (1 + g) * 0.8 ** (1 / 2): two years since close
    assert got["reforecast_growth"] == pytest.approx(((1.05) * 0.8 ** 0.5 - 1) * 100)
    year2 = next(y for y in got["years"] if y["year"] == 2)
    assert year2["reforecast_ebitda"] == pytest.approx(p.ebitda[1] * 0.8, abs=0.02)
    assert year2["actual_ebitda"] == pytest.approx(p.ebitda[1] * 0.8)
    assert got["reforecast"]["irr"] < got["plan"]["irr"]
    assert got["based_on"] == {"year": 2, "quarter": 4}
    assert 0.0 <= got["simulated"]["below_plan"] <= 1.0
    assert got["simulated"]["p5"] <= got["simulated"]["p50"] <= got["simulated"]["p95"]


def test_results_on_plan_reforecast_the_plan():
    got = portfolio.reforecast(DEAL, CFG, quarters(on_plan(1, 4)), n=2000)
    assert got["reforecast_growth"] == pytest.approx(DEAL.growth)
    assert got["reforecast"]["irr"] == pytest.approx(got["plan"]["irr"], abs=1e-4)


def test_reforecast_reads_the_last_four_quarters_across_a_year_end():
    p = plan()
    rows = on_plan(1, 4) + on_plan(2, 2, share=0.9)
    rows = [r for r in rows if not (r["year"] == 1 and r["quarter"] <= 2)]
    got = portfolio.reforecast(DEAL, CFG, quarters(rows), n=2000)
    ltm_plan = p.ebitda[0] / 2 + p.ebitda[1] / 2
    assert got["ltm_ebitda_plan"] == pytest.approx(ltm_plan)
    assert got["ltm_ebitda_actual"] == pytest.approx(p.ebitda[0] / 2 + p.ebitda[1] / 2 * 0.9)
    ratio = got["ltm_ebitda_actual"] / ltm_plan
    assert got["reforecast_growth"] == pytest.approx((1.05 * ratio ** (1 / 1.5) - 1) * 100)


def test_reforecast_needs_four_quarters_in_a_row():
    with pytest.raises(portfolio.QuartersRefused) as refused:
        portfolio.reforecast(DEAL, CFG, quarters(on_plan(1, 3)), n=2000)
    assert refused.value.code == "no_ltm"
    gap = [r for r in on_plan(1, 4) + on_plan(2, 1) if r["quarter"] != 3 or r["year"] != 1]   # year 1's Q3 missing
    with pytest.raises(portfolio.QuartersRefused) as refused:
        portfolio.reforecast(DEAL, CFG, quarters(gap), n=2000)
    assert refused.value.code == "no_ltm"
    rows = on_plan(1, 4)
    for row in rows:
        row["ebitda"] = -1.0
    with pytest.raises(portfolio.QuartersRefused) as refused:
        portfolio.reforecast(DEAL, CFG, quarters(rows), n=2000)
    assert refused.value.code == "ebitda_not_positive"


def test_reforecast_keeps_the_entry_and_the_debt():
    got = portfolio.reforecast(replace(DEAL, growth=8.0), CFG, quarters(on_plan(1, 4, share=1.1)), n=2000)
    assert got["reforecast"]["entry_equity"] == pytest.approx(got["plan"]["entry_equity"])


# ------------------------------------------------------------------------------------------------ currency
RATES = {"USD": 1.08, "GBP": 0.85, "JPY": 160.0}             # per euro, as the ECB publishes them


def test_conversion_at_the_ecb_cross_rate():
    # 100 GBP millions in USD millions: 100 / 0.85 * 1.08
    assert portfolio.fx_factor("GBP", "millions", "USD", "millions", RATES) * 100 == pytest.approx(100 / 0.85 * 1.08)
    assert portfolio.fx_factor("EUR", "millions", "USD", "millions", RATES) == pytest.approx(1.08)
    assert portfolio.fx_factor("USD", "millions", "EUR", "thousands", RATES) == pytest.approx(1000 / 1.08)
    assert portfolio.fx_factor("USD", "thousands", "USD", "millions", RATES) == pytest.approx(0.001)
    assert portfolio.fx_factor("CHF", "millions", "USD", "millions", RATES) is None
    # the rate stated: units of the deal's currency per one of the reporting currency
    assert portfolio.stated_rate("GBP", "USD", RATES) == pytest.approx(0.85 / 1.08)


def test_a_gbp_and_a_eur_holding_total_in_usd():
    holdings = [
        {"id": "a", "currency": "GBP", "unit": "thousands", "local": {"entry_equity": 250_000.0, "ltm_ebitda": None}},
        {"id": "b", "currency": "EUR", "unit": "millions", "local": {"entry_equity": 300.0, "ltm_ebitda": 40.0}},
    ]
    board = portfolio.board(holdings, "USD", "millions", RATES)
    a, b = board["holdings"]
    assert a["converted"]["entry_equity"] == pytest.approx(250 / 0.85 * 1.08)
    assert b["converted"]["entry_equity"] == pytest.approx(300 * 1.08)
    assert board["totals"]["entry_equity"] == {"value": pytest.approx(250 / 0.85 * 1.08 + 300 * 1.08), "holdings": 2}
    assert board["totals"]["ltm_ebitda"] == {"value": pytest.approx(40 * 1.08), "holdings": 1}
    assert board["rates"] == {"EUR": pytest.approx(1 / 1.08), "GBP": pytest.approx(0.85 / 1.08)}
    assert board["missing_rates"] == []


def test_a_holding_without_a_rate_is_left_out_of_the_totals_and_named():
    holdings = [{"id": "c", "currency": "CHF", "unit": "millions", "local": {"entry_equity": 10.0}},
                {"id": "d", "currency": "USD", "unit": "millions", "local": {"entry_equity": 5.0}}]
    board = portfolio.board(holdings, "USD", "millions", RATES)
    assert board["holdings"][0]["converted"] is None
    assert board["totals"]["entry_equity"] == {"value": pytest.approx(5.0), "holdings": 1}
    assert board["missing_rates"] == ["CHF"]
