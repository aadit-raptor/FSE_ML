"""Global tax rules (PLAN.md 2.5), each case checked by hand.

A deal's tax used to be one line: a flat rate on positive profit before tax,
with a loss simply lost. 2.5 adds what the main tax systems actually do to a
leveraged buyout, as rules a deal switches on:

* an **interest limit** -- a share of EBITDA (with an allowance that is
  always deductible) or a fixed amount -- with the disallowed interest
  carried forward to a year with room;
* **losses carried forward**, optionally capped each year at an allowance
  plus a share of the profit above it;
* an optional **minimum tax** on book profit.

The mechanics live in ``lbo_engine/tax.py`` and are the same function for the
deal model and the simulation; the country presets, with their sources and
dates, live in ``core/tax.py``. With every rule off, nothing moves: the golden
snapshot and the pinned simulation are untouched.
"""
import dataclasses

import numpy as np
import pytest

from core.config import resolve_config
from core.deal import DealInputs, run_deal
from lbo_engine.tax import TaxRules, tax_schedule

RATE = 0.25


def schedule(ebitda, ebit, interest, rules, rate=RATE):
    n = len(ebitda)
    return tax_schedule(ebitda=ebitda, ebit=ebit, net_interest=interest,
                        tax_rate=[rate] * n, rules=rules)


# ---------------------------------------------------------------------------
# The mechanics, by hand
# ---------------------------------------------------------------------------
def test_with_no_rules_tax_is_the_rate_on_positive_profit():
    s = schedule([100, 100], [80, 20], [40, 30], TaxRules())
    assert s.taxes == pytest.approx([10.0, 0.0])      # 25% x 40; a loss pays nothing
    assert s.losses_carried == [0.0, 0.0]


def test_a_30_percent_of_ebitda_interest_cap_carries_the_excess_forward():
    """EBITDA 100 a year, so the cap is 30. EBIT 80 a year.

        year   interest   claim (incl. carried)   deducted   carried   taxable   tax 25%
        1      40         40                      30         10        50        12.5
        2      20         30                      30          0        50        12.5
        3      10         10                      10          0        70        17.5

    Without the cap the taxes would be 10.0, 15.0, 17.5: the cap moves 2.5 of
    tax from year 2 to year 1 and none is lost.
    """
    s = schedule([100] * 3, [80] * 3, [40, 20, 10], TaxRules(interest_limit="ebitda_share",
                                                             interest_limit_share=0.30))
    assert s.interest_deductible == pytest.approx([30, 30, 10])
    assert s.interest_carried == pytest.approx([10, 0, 0])
    assert s.taxable_income == pytest.approx([50, 50, 70])
    assert s.taxes == pytest.approx([12.5, 12.5, 17.5])


def test_the_allowance_is_deductible_whatever_the_share_says():
    """EBITDA 20: 30% of it is 6, but an allowance of 10 is always
    deductible, so 10 of the 12 of interest is deducted and 2 carried."""
    s = schedule([20], [15], [12], TaxRules(interest_limit="ebitda_share", interest_limit_share=0.30,
                                            interest_limit_amount=10.0))
    assert s.interest_deductible == pytest.approx([10])
    assert s.interest_carried == pytest.approx([2])
    assert s.taxes == pytest.approx([0.25 * (15 - 10)])


def test_a_fixed_interest_cap():
    """A cap of 25 a year whatever the EBITDA: 40 -> 25 deducted, 15 carried;
    then 10 + 15 = 25 -> all of it."""
    s = schedule([100, 100], [80, 80], [40, 10], TaxRules(interest_limit="fixed",
                                                          interest_limit_amount=25.0))
    assert s.interest_deductible == pytest.approx([25, 25])
    assert s.interest_carried == pytest.approx([15, 0])


def test_losses_carried_forward_with_a_yearly_cap():
    """Losses offset in full up to 10 a year, and 60% of the profit above it.

        year   profit   limit on losses        used   left   taxable   tax 25%
        1      -50      --                     --     50     0         0
        2       30      10 + 60% x 20 = 22     22     28     8         2
        3      100      10 + 60% x 90 = 64     28      0     72        18

    Without carrying losses forward the tax would be 0, 7.5, 25.
    """
    s = schedule([0] * 3, [-50, 30, 100], [0, 0, 0],
                 TaxRules(loss_carryforward=True, loss_limit_share=0.60, loss_limit_amount=10.0))
    assert s.losses_used == pytest.approx([0, 22, 28])
    assert s.losses_carried == pytest.approx([50, 28, 0])
    assert s.taxable_income == pytest.approx([0, 8, 72])
    assert s.taxes == pytest.approx([0, 2, 18])


def test_losses_with_no_cap_offset_everything():
    s = schedule([0] * 2, [-50, 80], [0, 0], TaxRules(loss_carryforward=True))
    assert s.losses_used == pytest.approx([0, 50])
    assert s.taxes == pytest.approx([0, 7.5])


def test_disallowed_interest_can_create_the_profit_a_loss_then_absorbs():
    """Both rules together. EBIT 10, interest 40 against a cap of 30% of
    EBITDA 50 = 15: 25 carried. Profit for tax 10 - 15 = -5 is a loss.
    Year 2: EBIT 60, interest 5; claim 30 against a cap of 15 -> 15, 15
    carried; profit 45, the loss of 5 used in full, taxable 40, tax 10."""
    rules = TaxRules(interest_limit="ebitda_share", interest_limit_share=0.30, loss_carryforward=True)
    s = schedule([50, 50], [10, 60], [40, 5], rules)
    assert s.interest_carried == pytest.approx([25, 15])
    assert s.losses_carried == pytest.approx([5, 0])
    assert s.taxes == pytest.approx([0, 10])


def test_a_minimum_tax_on_book_profit():
    """Book profit 30, but a loss carried in absorbs it all, so the regular
    tax is 0. A 15% minimum tax on the book profit still charges 4.5."""
    s = schedule([0, 0], [-40, 30], [0, 0],
                 TaxRules(loss_carryforward=True, minimum_tax_rate=0.15))
    assert s.regular_tax == pytest.approx([0, 0])
    assert s.minimum_tax_topup == pytest.approx([0, 4.5])
    assert s.taxes == pytest.approx([0, 4.5])


def test_net_interest_income_is_taxed_not_capped():
    """Interest income larger than the expense is income: nothing to cap."""
    s = schedule([10], [5], [-3], TaxRules(interest_limit="fixed", interest_limit_amount=0.0))
    assert s.taxable_income == pytest.approx([8])


def test_the_mechanics_run_on_every_simulated_path_at_once():
    """The simulation hands the same function arrays of paths; each path is
    what the scalar case would give."""
    rules = TaxRules(interest_limit="ebitda_share", interest_limit_share=0.30, loss_carryforward=True)
    ebitda = [np.array([100.0, 50.0]), np.array([100.0, 50.0])]
    ebit = [np.array([80.0, 10.0]), np.array([80.0, 60.0])]
    interest = [np.array([40.0, 40.0]), np.array([20.0, 5.0])]
    s = tax_schedule(ebitda=ebitda, ebit=ebit, net_interest=interest, tax_rate=[RATE, RATE], rules=rules)
    one = schedule([100, 100], [80, 80], [40, 20], rules)
    two = schedule([50, 50], [10, 60], [40, 5], rules)
    for year in range(2):
        assert s.taxes[year] == pytest.approx([one.taxes[year], two.taxes[year]])


# ---------------------------------------------------------------------------
# Through the deal model
# ---------------------------------------------------------------------------
def cfg():
    return resolve_config()


def test_a_deal_with_no_tax_rules_is_untouched():
    r = run_deal(DealInputs(), cfg())
    assert r.tax is None
    assert r.returns.irr == pytest.approx(0.2116, abs=5e-5)


def test_rules_switched_off_are_no_rules():
    off = DealInputs(tax_interest_limit="none", tax_loss_carryforward=False, tax_minimum_pct=0.0)
    assert run_deal(off, cfg()).returns.irr == run_deal(DealInputs(), cfg()).returns.irr


CAPPED = DealInputs(tax_interest_limit="ebitda_share", tax_interest_limit_pct=30.0)


def test_a_capped_deal_pays_more_tax_and_its_schedule_adds_up():
    """The default deal's year-one interest, 46.2, is more than 30% of its
    EBITDA, so the cap bites: more tax in the early years, a lower IRR, and
    each year's tax is the rate on the taxable income the schedule shows."""
    free, capped = run_deal(DealInputs(), cfg()), run_deal(CAPPED, cfg())
    s, om = capped.tax, capped.operating_model
    assert om.taxes[0] > free.operating_model.taxes[0]
    assert capped.returns.irr < free.returns.irr
    for t in range(len(om.years)):
        assert s.interest_deductible[t] <= 0.30 * om.ebitda[t] + 1e-9
        assert s.taxes[t] == pytest.approx(0.25 * s.taxable_income[t])
        assert om.taxes[t] == pytest.approx(s.taxes[t], abs=0.006)
    # Year one by hand: EBIT less 30% of EBITDA, taxed at 25%
    assert s.taxable_income[0] == pytest.approx(om.ebit[0] - 0.30 * om.ebitda[0])


def test_a_loss_making_deal_keeps_its_losses_when_they_carry_forward():
    """Debt at 80% of EV at 12% loses money in year one; the company grows
    20% a year, so the loss, carried forward, cuts year two's tax, and the
    IRR rises.

    The senior loan here does not amortise. With the default 5% a year, year
    two's cash is below the mandatory repayment, the model funds the gap out
    of nothing (CLAUDE.md "Model findings", 11), and the tax saved vanishes
    into that gap: the IRR would not move at all.
    """
    heavy = dict(debt_pct=80.0, base_rate=12.0, growth=20.0, hold=7)
    no_amort = resolve_config({"def_senior_amort": 0.0})
    lost = run_deal(DealInputs(**heavy), no_amort)
    kept = run_deal(DealInputs(**heavy, tax_loss_carryforward=True), no_amort)
    assert min(lost.operating_model.ebt) < 0
    assert sum(kept.operating_model.taxes) < sum(lost.operating_model.taxes)
    assert kept.returns.irr > lost.returns.irr
    assert max(kept.tax.losses_carried) > 0


# ---------------------------------------------------------------------------
# Country presets
# ---------------------------------------------------------------------------
def test_every_preset_says_where_it_comes_from():
    from core.tax import PRESETS
    assert len(PRESETS) >= 8
    for p in PRESETS:
        assert p.code.isupper() and len(p.code) == 2, p.code
        assert p.currency.isupper() and len(p.currency) == 3, p.code
        assert p.source and p.as_of, p.code
        assert 0 < p.rate < 60, p.code
        assert p.interest_limit in ("none", "ebitda_share", "fixed"), p.code


def test_a_preset_sets_every_rule_and_amounts_only_in_its_own_currency():
    from core.tax import apply_preset
    gbp, applied = apply_preset(DealInputs(currency="GBP"), "GB")
    assert applied
    assert gbp.tax == 25.0 and gbp.tax_preset == "GB"
    assert (gbp.tax_interest_limit, gbp.tax_interest_limit_pct, gbp.tax_interest_limit_amount) == \
        ("ebitda_share", 30.0, 2.0)
    assert (gbp.tax_loss_carryforward, gbp.tax_loss_limit_pct, gbp.tax_loss_limit_amount) == (True, 50.0, 5.0)
    # In thousands the same allowance is 2,000
    thousands, _ = apply_preset(DealInputs(currency="GBP", unit="thousands"), "GB")
    assert thousands.tax_interest_limit_amount == 2000.0
    # A dollar deal gets the rates and shares, not pound amounts it cannot convert
    usd, applied = apply_preset(DealInputs(currency="USD"), "GB")
    assert not applied and usd.tax_interest_limit_amount == 0.0 and usd.tax_loss_limit_amount == 0.0
    assert usd.tax_interest_limit_pct == 30.0


def test_changing_preset_changes_the_tax_as_predicted():
    """Ireland's 12.5% and the UK's 25%, both with a 30%-of-EBITDA cap that
    binds on the default deal in year one (46.2 of interest against 31.5):
    the taxable income is the same, so Ireland's year-one tax is exactly
    half the UK's."""
    from core.tax import apply_preset
    uk, _ = apply_preset(DealInputs(currency="GBP"), "GB")
    ie, _ = apply_preset(DealInputs(currency="EUR"), "IE")
    a, b = run_deal(uk, cfg()).tax, run_deal(ie, cfg()).tax
    assert b.taxable_income[0] == pytest.approx(a.taxable_income[0])
    assert b.taxes[0] == pytest.approx(a.taxes[0] / 2)


# ---------------------------------------------------------------------------
# The simulation applies the same rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deal", [
    CAPPED,
    # Losses in year one, profits after (see the loss-making deal above)
    DealInputs(debt_pct=80.0, base_rate=12.0, growth=20.0, hold=7,
               tax_loss_carryforward=True, tax_loss_limit_pct=60.0),
    DealInputs(debt_pct=80.0, base_rate=12.0, growth=20.0, hold=7,
               tax_loss_carryforward=True, tax_minimum_pct=15.0),
], ids=["interest cap", "capped losses", "minimum tax"])
def test_a_simulated_path_at_the_mean_lands_on_the_deal_models_tax(deal):
    from core.montecarlo import MCInputs, build_sim_params
    from simulation.vectorized_simulation import _run_vectorized_core

    params = dataclasses.replace(
        build_sim_params(MCInputs(n=1, rate_mean=deal.base_rate, hold=deal.hold), deal, cfg()), n_interest_passes=50)
    draws = {"growth": np.array([deal.growth / 100]), "exit_multiple": np.array([deal.exit_mult]),
             "interest": np.array([params.interest_mean]),
             "gross_margin": np.array([deal.gross_margin / 100]), "ebitda_shock": np.array([0.0])}
    out = _run_vectorized_core(params, draws)
    r = run_deal(deal, cfg())
    assert out["IRR"][0] == pytest.approx(r.returns.irr, abs=2e-4)


def test_the_simulation_feels_the_rules():
    """The same paths with and without the cap: a cap only ever adds tax, so
    no path does better with it, and most do worse -- the rest are paths whose
    rate draw keeps interest under 30% of EBITDA (about a quarter of them)."""
    from core.montecarlo import MCInputs, build_sim_params
    from simulation.vectorized_simulation import run_vectorized_simulation_full

    def irr(deal):
        return run_vectorized_simulation_full(build_sim_params(MCInputs(n=4000), deal, cfg()), seed=3).irr

    free, capped = irr(DealInputs()), irr(CAPPED)
    assert (capped <= free + 1e-12).all()
    assert (capped < free).mean() > 0.5


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_the_presets_are_served_with_their_sources():
    body = client().get("/api/deal/tax-presets").json()
    gb = next(p for p in body["presets"] if p["code"] == "GB")
    assert gb["rate"] == 25.0 and gb["currency"] == "GBP" and gb["source"] and gb["as_of"]


def test_a_deal_run_reports_its_tax_schedule_in_the_deals_unit():
    c = client()
    base = {"tax_interest_limit": "fixed", "tax_interest_limit_amount": 20.0}
    millions = c.post("/api/deal/run", json={"inputs": base}).json()
    thousands = c.post("/api/deal/run", json={"inputs": {
        **base, "unit": "thousands", "ebitda": 100_000.0, "tax_interest_limit_amount": 20_000.0}}).json()
    assert millions["tax"]["interest_deductible"][0] == pytest.approx(20.0)
    assert thousands["tax"]["interest_deductible"][0] == pytest.approx(20_000.0)
    assert thousands["returns"]["irr"] == pytest.approx(millions["returns"]["irr"], rel=1e-9)
    assert c.post("/api/deal/run", json={"inputs": {}}).json()["tax"] is None


@pytest.mark.parametrize("bad", [
    {"tax_interest_limit": "sometimes"}, {"tax_interest_limit_pct": 150.0},
    {"tax_loss_limit_pct": -1.0}, {"tax_minimum_pct": 101.0}, {"tax_interest_limit_amount": -5.0},
])
def test_impossible_rules_are_refused(bad):
    assert client().post("/api/deal/run", json={"inputs": bad}).status_code == 422


def test_the_simulation_endpoint_takes_the_rules():
    body = {"mc": {"n": 2000}, "seed": 4, "deal": {"tax_interest_limit": "ebitda_share"}}
    resp = client().post("/api/montecarlo/run", json=body)
    assert resp.status_code == 200, resp.text
    assert "tax_rules" not in resp.json()["params"]


def test_a_deal_without_rules_stores_exactly_what_it_stored_before():
    from db import deals as store
    stored = store.clean_inputs({"ebitda": 50})
    assert not any(k.startswith("tax_") for k in stored)
    kept = store.clean_inputs({"ebitda": 50, "tax_loss_carryforward": True})
    assert kept["tax_loss_carryforward"] is True and "tax_minimum_pct" not in kept


def test_the_heatmap_applies_the_rules():
    """Each cell is a deal-model run, so it is taxed the deal's way: the cap
    lowers every cell, and a cell is exactly run_deal's IRR for that growth
    and exit multiple."""
    from core.montecarlo import MCInputs, build_sim_params, growth_exit_heatmap

    mc = MCInputs(n=1000)
    _, _, free = growth_exit_heatmap(build_sim_params(mc, DealInputs(), cfg()), mc, DealInputs())
    g, em, capped = growth_exit_heatmap(build_sim_params(mc, CAPPED, cfg()), mc, CAPPED)
    assert (capped < free).all()
    deal = dataclasses.replace(CAPPED, growth=float(g[3]) * 100, exit_mult=float(em[2]))
    assert capped[2, 3] == pytest.approx(run_deal(deal, cfg()).returns.irr, rel=1e-9)
