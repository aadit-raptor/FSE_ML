"""Accounting standards (PLAN.md 2.6): IFRS and US GAAP, each case by hand.

Two halves:

* **Filings map to model inputs.** ``ml/edgar_extractor.py`` reads a US
  filer's 10-K in the ``us-gaap`` taxonomy and, since 2.6, a foreign filer's
  20-F in the ``ifrs-full`` one, in the filing's own currency. Both are pinned
  by real filings recorded from SEC EDGAR (``tests/fixtures/edgar/``, trimmed
  to the concepts the maps name): SAP SE's 2025 20-F (IFRS, euros) and
  McDonald's 10-K (US GAAP), whose answer was recorded before 2.6 touched the
  extractor and must not move.

* **The IFRS 16 lease setting.** A deal's EBITDA is before lease costs under
  IFRS and after them under US GAAP. The deal is priced on one view:
  ``pre_ifrs16`` (EBITDA after lease costs, leases not debt) or
  ``post_ifrs16`` (EBITDA before lease costs, the lease liability counted
  with net debt). The cash the business earns is the same either way.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.accounting import (
    IFRS, POST_IFRS16, PRE_IFRS16, US_GAAP, InvalidLeases, lease_terms,
)
from core.config import resolve_config
from core.deal import DealInputs, run_deal
from core.montecarlo import MCInputs, build_sim_params
from simulation.vectorized_simulation import _run_vectorized_core

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"
client = TestClient(app)

# Fees off, so entry equity is EV less debt less leases by hand
NO_FEES = resolve_config({"tx_fee_pct": 0.0, "fin_fee_pct": 0.0, "other_uses": 0.0})


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def serve_filing(monkeypatch, name, cik, ticker, company):
    """SEC EDGAR, answered from a recorded filing."""
    import ml.edgar_extractor as ee

    facts = json.loads((FIXTURES / f"{name}_companyfacts.json").read_text(encoding="utf-8"))

    def get(url, headers=None, timeout=None):
        if url == ee.TICKER_URL:
            return _Resp({"0": {"cik_str": cik, "ticker": ticker, "title": company}})
        if "submissions" in url:
            return _Resp({"name": company})
        return _Resp(facts)

    monkeypatch.setattr(ee.requests, "get", get)
    monkeypatch.setattr(ee.time, "sleep", lambda s: None)
    return ee


def test_a_us_gaap_10k_extracts_exactly_as_before_2_6(monkeypatch):
    """Every field, year, warning and history value McDonald's gave before
    the extractor learned a second taxonomy."""
    ee = serve_filing(monkeypatch, "mcd", 63908, "MCD", "MCDONALDS CORP")
    before = json.loads((FIXTURES / "mcd_expected_before_2_6.json").read_text(encoding="utf-8"))
    x = ee.fetch_financials("MCD", n_years=3)
    now = dataclasses.asdict(x)
    for key, value in before["extracted"].items():
        assert now[key] == value, key
    assert ee.financials_to_session_state(x) == before["history"]
    assert x.accounting_standard == US_GAAP and x.currency == "USD"


def test_a_us_gaap_filer_reports_its_operating_leases(monkeypatch):
    """McDonald's, by hand from its 10-Ks: operating lease liability 12,170.3
    at the end of 2023, then no longer tagged in the 10-K; no operating lease
    cost tag, so next year's payments due stand in (1,200.0 for 2025)."""
    ee = serve_filing(monkeypatch, "mcd", 63908, "MCD", "MCDONALDS CORP")
    x = ee.fetch_financials("MCD", n_years=3)
    assert x.leases["lease_liability"] == [pytest.approx(12170.3), 0.0, 0.0]
    assert x.leases["lease_cost"][-1] == pytest.approx(1200.0)


def test_the_answer_says_when_the_latest_year_has_no_lease_liability(monkeypatch):
    serve_filing(monkeypatch, "mcd", 63908, "MCD", "MCDONALDS CORP")
    body = client.get("/api/edgar/MCD").json()
    assert any("lease liability" in w and "2025" in w for w in body["warnings"])
    assert body["deal_inputs"]["lease_liability"] == 0.0


def test_sap_ifrs_20f_maps_to_model_inputs(monkeypatch):
    """SAP SE, 20-F for 2025, in euro millions, read by hand from the filing:

        revenue 36,800   cost of sales 9,986   R&D 6,633
        sales and marketing 8,879 + administration 1,633 = SG&A 10,512
        operating profit 9,617, so the 52 of other operating costs the
        taxonomy doesn't split out are folded into SG&A: 10,564
        finance costs 1,377   finance income 1,911   income tax 2,944
        D&A (cash flow add-back) 1,311   share-based payments 1,695
        capex 739   cash 8,220   trade and other receivables 6,675
        borrowings 6,150 (4,550 long term + 1,600 current)
        current contract liabilities 6,581   trade payables 2,431
        lease payments 299   lease liabilities 1,684 (254 current + 1,430)
    """
    ee = serve_filing(monkeypatch, "sap", 1000184, "SAP", "SAP SE")
    x = ee.fetch_financials("SAP", n_years=3)
    assert x.accounting_standard == IFRS and x.currency == "EUR"
    assert x.years == [2023, 2024, 2025] and x.fiscal_year_end_month == 12
    last = {k: v[-1] for k, v in x.data.items()}
    assert last["revenue"] == 36800
    assert last["cost_of_revenue"] == 9986
    assert last["research_and_development"] == 6633
    assert last["selling_general_admin"] == pytest.approx(8879 + 1633 + 52)
    assert last["revenue"] - last["cost_of_revenue"] - last["research_and_development"] \
        - last["selling_general_admin"] == pytest.approx(9617)
    assert (last["interest_expense"], last["interest_income"]) == (1377, 1911)
    assert last["income_tax_expense"] == 2944
    assert last["depreciation_amortization"] == 1311
    assert last["stock_based_compensation"] == 1695
    assert last["capital_expenditures"] == 739
    assert last["cash_and_equivalents"] == 8220
    assert last["accounts_receivable"] == 6675
    assert last["long_term_debt"] == 6150
    assert last["deferred_revenue"] == 6581
    assert last["accounts_payable"] == 2431
    assert x.leases["lease_cost"][-1] == 299
    assert x.leases["lease_liability"][-1] == 1684
    # Balanced to the reported totals, as a US filer's is
    assert last["total_assets"] == 70362
    history = ee.financials_to_session_state(x)
    assert history["hist_h_rev_2"] == 36800 and history["hist_h_cogs_2"] == -9986


def test_the_edgar_answer_says_the_standard_currency_and_deal_inputs(monkeypatch):
    """The IFRS filing's EBITDA is before lease costs: operating profit 9,617
    plus D&A 1,311 = 10,928. As deal inputs it arrives with its standard and
    its leases, so a deal built from it is priced post-IFRS 16 by default."""
    serve_filing(monkeypatch, "sap", 1000184, "SAP", "SAP SE")
    body = client.get("/api/edgar/SAP").json()
    assert body["money"] == {"currency": "EUR", "unit": "millions"}
    assert body["accounting_standard"] == IFRS
    assert body["deal_inputs"] == {
        "ebitda": pytest.approx(10928), "currency": "EUR", "unit": "millions",
        "accounting_standard": IFRS, "lease_cost": 299, "lease_liability": 1684,
    }
    assert body["leases"]["lease_liability"][-1] == 1684


def test_a_us_filer_still_answers_in_us_dollars(monkeypatch):
    serve_filing(monkeypatch, "mcd", 63908, "MCD", "MCDONALDS CORP")
    body = client.get("/api/edgar/MCD").json()
    assert body["money"] == {"currency": "USD", "unit": "millions"}
    assert body["accounting_standard"] == US_GAAP
    assert body["deal_inputs"]["accounting_standard"] == US_GAAP


# ---------------------------------------------------------------------------
# Leases: the terms, by hand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("standard", ["", IFRS, US_GAAP])
@pytest.mark.parametrize("view", ["", PRE_IFRS16, POST_IFRS16])
def test_no_leases_change_nothing(standard, view):
    t = lease_terms(100.0, standard, view, 0.0, 0.0)
    assert (t.operating_ebitda, t.valuation_addback, t.debt_like) == (100.0, 0.0, 0.0)


def test_lease_terms_for_each_standard_and_view():
    """EBITDA 100 as reported, leases costing 10 a year, liability 60."""
    assert dataclasses.astuple(lease_terms(100, IFRS, POST_IFRS16, 10, 60)) == (POST_IFRS16, 90, 10, 60)
    assert dataclasses.astuple(lease_terms(100, IFRS, PRE_IFRS16, 10, 60)) == (PRE_IFRS16, 90, 0, 0)
    assert lease_terms(100, IFRS, "", 10, 60).view == POST_IFRS16        # IFRS is already post
    assert dataclasses.astuple(lease_terms(100, US_GAAP, PRE_IFRS16, 10, 60)) == (PRE_IFRS16, 100, 0, 0)
    assert dataclasses.astuple(lease_terms(100, US_GAAP, POST_IFRS16, 10, 60)) == (POST_IFRS16, 100, 10, 60)
    assert lease_terms(100, US_GAAP, "", 10, 60).view == PRE_IFRS16


def test_leases_the_ifrs_ebitda_cannot_carry_are_refused():
    with pytest.raises(InvalidLeases):
        lease_terms(10, IFRS, "", 10, 0)
    with pytest.raises(InvalidLeases):
        lease_terms(100, IFRS, "", -1, 0)


# ---------------------------------------------------------------------------
# Leases in the deal model: the hand-checked case
# ---------------------------------------------------------------------------
def deal(**kw):
    return DealInputs(**{"ebitda": 100.0, "entry_mult": 10.0, "exit_mult": 11.0, "debt_pct": 60.0, **kw})


def ifrs(view, **kw):
    return deal(accounting_standard=IFRS, lease_view=view, lease_cost=10.0, lease_liability=60.0, **kw)


def test_pre_ifrs16_is_the_same_business_valued_on_ebitda_after_lease_costs():
    """IFRS EBITDA 100 with leases costing 10: after lease costs it is 90.
    Priced pre-IFRS 16 the deal is exactly the deal with EBITDA 90 and no
    leases: EV 900, debt 540, equity 360, and every year the same."""
    a = run_deal(ifrs(PRE_IFRS16), NO_FEES)
    b = run_deal(deal(ebitda=90.0), NO_FEES)
    assert a.returns.entry_equity == pytest.approx(360.0)
    assert a.returns.irr == b.returns.irr and a.returns.moic == b.returns.moic
    assert a.returns.net_debt_at_exit == b.returns.net_debt_at_exit
    assert a.operating_model.ebitda == b.operating_model.ebitda


def test_post_ifrs16_values_ebitda_before_leases_and_counts_them_as_debt():
    """Priced post-IFRS 16 on the same business:

        entry EV   = 10 x (90 + 10)            = 1,000
        debt       = 60% x 1,000               =   600
        equity     = 1,000 - 600 - leases 60   =   340
        net debt at entry = 600 + 60           =   660   (pre: 540, so 120 more)
        each year's EBITDA is the cash one, after lease costs (90 grown)
        exit EV    = 11 x (exit EBITDA after leases + 10)
        net debt at exit = the debt schedule's + 60
    """
    r = run_deal(ifrs(POST_IFRS16), NO_FEES)
    pre = run_deal(ifrs(PRE_IFRS16), NO_FEES)
    assert r.returns.entry_equity == pytest.approx(340.0)
    assert r.operating_model.ebitda == pre.operating_model.ebitda          # the cash business
    exit_ebitda = r.operating_model.exit_ebitda
    assert r.returns.exit_ev == pytest.approx(11 * (exit_ebitda + 10), abs=0.01)
    assert r.returns.net_debt_at_exit == pytest.approx(r.debt_schedule.net_debt_at_exit + 60, abs=0.01)
    assert r.returns.gross_exit_equity == pytest.approx(r.returns.exit_ev - r.returns.net_debt_at_exit,
                                                        abs=0.01)
    bridge = r.equity_bridge
    assert bridge["entry_equity"] == pytest.approx(340.0)
    assert abs(bridge["residual"]) < 0.05


def test_a_us_gaap_deal_priced_post_ifrs16_adds_its_lease_cost_back():
    """US GAAP EBITDA 100 is already after lease costs: post-IFRS 16 it is
    valued on 110, EV 1,100, debt 660, equity 1,100 - 660 - 60 = 380; the
    operating model still grows the 100."""
    r = run_deal(deal(accounting_standard=US_GAAP, lease_view=POST_IFRS16, lease_cost=10.0,
                      lease_liability=60.0), NO_FEES)
    plain = run_deal(deal(), NO_FEES)
    assert r.returns.entry_equity == pytest.approx(380.0)
    assert r.operating_model.ebitda == plain.operating_model.ebitda


def test_the_exit_sensitivity_grid_values_every_hold_the_same_way():
    """The grid's cell for the deal's own hold and exit multiple is the
    deal's IRR; before, it ignored the leases."""
    r = run_deal(ifrs(POST_IFRS16), NO_FEES)
    grid = r.exit_sensitivity
    hold_col = grid["holding_periods"].index(5)
    row = grid["exit_multiples"].index(11.0)
    assert grid["table"][row][hold_col] == pytest.approx(r.returns.irr, abs=1e-6)
    # Every other hold is its own full run, valued the same way
    other = run_deal(ifrs(POST_IFRS16, hold=4), NO_FEES)
    assert grid["table"][row][grid["holding_periods"].index(4)] == pytest.approx(other.returns.irr, abs=1e-6)


# ---------------------------------------------------------------------------
# The simulation follows the deal model
# ---------------------------------------------------------------------------
def draws_at(d, params):
    full = lambda x: np.full(1, float(x))   # noqa: E731
    return {"growth": full(d.growth / 100), "exit_multiple": full(d.exit_mult),
            "interest": full(params.interest_mean), "gross_margin": full(d.gross_margin / 100),
            "ebitda_shock": full(0.0)}


@pytest.mark.parametrize("view", [PRE_IFRS16, POST_IFRS16])
def test_a_simulated_path_at_the_mean_lands_on_the_deal_model(view):
    d = ifrs(view, exit_mult=10.0)
    mc = MCInputs(n=1, ebitda=d.ebitda, entry_mult=d.entry_mult, exit_mean=d.exit_mult,
                  rate_mean=d.base_rate)
    params = dataclasses.replace(build_sim_params(mc, d, NO_FEES), n_interest_passes=50)
    out = _run_vectorized_core(params, draws_at(d, params))
    r = run_deal(d, NO_FEES)
    assert out["Net Debt Exit"][0] == pytest.approx(r.returns.net_debt_at_exit, abs=0.05)
    assert out["IRR"][0] == pytest.approx(r.returns.irr, abs=1e-4)


def test_the_simulation_feels_the_lease_view():
    def irr(view):
        d = ifrs(view)
        mc = MCInputs(n=2000, ebitda=d.ebitda, entry_mult=d.entry_mult)
        from simulation.vectorized_simulation import run_vectorized_simulation_full
        return run_vectorized_simulation_full(build_sim_params(mc, d, NO_FEES), seed=1).df["IRR"].mean()
    assert irr(POST_IFRS16) != pytest.approx(irr(PRE_IFRS16), abs=1e-4)


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------
def run(inputs, settings=None):
    return client.post("/api/deal/run", json={"inputs": inputs, "settings": settings or {}})


LEASED = {"ebitda": 100, "entry_mult": 10, "exit_mult": 11, "debt_pct": 60,
          "accounting_standard": "ifrs", "lease_view": "post_ifrs16",
          "lease_cost": 10, "lease_liability": 60}


def test_the_deal_answer_carries_its_leases():
    body = run(LEASED, {"tx_fee_pct": 0, "fin_fee_pct": 0, "other_uses": 0}).json()
    leases = body["leases"]
    assert leases["accounting_standard"] == "ifrs" and leases["view"] == "post_ifrs16"
    assert leases["operating_ebitda"] == pytest.approx(90)
    assert leases["valuation_ebitda"] == pytest.approx(100)
    assert leases["entry_ev"] == pytest.approx(1000)
    assert leases["lease_liability"] == pytest.approx(60)
    assert body["returns"]["entry_equity"] == pytest.approx(340)


def test_a_deal_without_leases_answers_no_lease_block():
    assert run({}).json()["leases"] is None


def test_a_leased_deal_in_thousands_is_the_millions_answer_times_1000():
    k = {**LEASED, "ebitda": 100_000, "lease_cost": 10_000, "lease_liability": 60_000, "unit": "thousands"}
    a, b = run(LEASED).json(), run(k).json()
    assert b["leases"]["entry_ev"] == pytest.approx(a["leases"]["entry_ev"] * 1000)
    assert b["leases"]["lease_liability"] == pytest.approx(60_000)
    assert b["returns"]["irr"] == pytest.approx(a["returns"]["irr"])


def test_leases_the_ebitda_cannot_carry_are_a_422():
    resp = run({**LEASED, "ebitda": 10, "lease_cost": 10})
    assert resp.status_code == 422
    assert "lease cost" in resp.json()["detail"].lower()


def test_sources_and_uses_assume_the_leases():
    """The buyer pays EV less the leases it takes over: 1,000 - 60."""
    body = client.post("/api/deal/sources-and-uses", json={
        "ebitda": 100, "entry_mult": 10, "senior_x": 4, "mezz_x": 2,
        "accounting_standard": "ifrs", "lease_cost": 10, "lease_liability": 60,
        "settings": {"tx_fee_pct": 0, "fin_fee_pct": 0, "other_uses": 0}}).json()
    assert body["equity_purchase_price"] == pytest.approx(940)
    assert body["lease_liability"] == pytest.approx(60)


def test_old_deals_store_no_lease_fields():
    from db.deals import OMIT_WHEN_DEFAULT, clean_inputs

    for key in ("accounting_standard", "lease_view", "lease_cost", "lease_liability"):
        assert key in OMIT_WHEN_DEFAULT
    assert not {"accounting_standard", "lease_view", "lease_cost", "lease_liability"} & set(clean_inputs({}))


def test_a_forecast_company_carries_its_standard():
    """The standard is the company's own label: the forecast runs the same
    arithmetic and says which standard its figures follow."""
    defaults = client.get("/api/forecasting/defaults").json()
    history = defaults["history"]
    plain = client.post("/api/forecasting/seed", json={"history": history}).json()
    ifrs_seed = client.post("/api/forecasting/seed", json={"history": history, "accounting_standard": "ifrs"}).json()
    assert plain["accounting_standard"] == "" and ifrs_seed["accounting_standard"] == "ifrs"
    assert ifrs_seed["seeded_assumptions"] == plain["seeded_assumptions"]
    run = client.post("/api/forecasting/run", json={
        "history": history, "assumptions": defaults["seeded_assumptions"] and
        {k: [v] * 3 for k, v in defaults["seeded_assumptions"].items()},
        "accounting_standard": "us_gaap"}).json()
    assert run["accounting_standard"] == "us_gaap"


def test_the_live_surrogate_names_a_deals_leases():
    """The Live sliders' network was trained without leases, so a leased deal
    lists them among the terms the estimate cannot see."""
    from core.surrogate import training_term_differences

    fixed = {"entry_multiple": 10.0, "holding_period": 5, "opex_pct": 0.18, "tax_rate": 0.25,
             "senior_pct": 0.7, "mezz_spread": 0.04, "interest_std": 0.015,
             "transaction_fees_pct": 0.02, "financing_fees_pct": 0.02, "other_uses": 0.0}
    mc = MCInputs()
    cfg = resolve_config({})
    base = {t["term"] for t in training_term_differences(mc, DealInputs(), cfg, fixed)}
    leased = {t["term"] for t in training_term_differences(mc, ifrs(POST_IFRS16), cfg, fixed)}
    assert leased - base == {"lease cost", "lease liability"}


def test_the_heatmap_values_the_leases_too():
    """Each heatmap cell is a deal-model run; left without the lease terms it
    would quietly price a post-IFRS 16 deal as if it had no leases."""
    from core.montecarlo import growth_exit_heatmap

    def grid(d):
        mc = MCInputs(n=1000, ebitda=d.ebitda, entry_mult=d.entry_mult)
        return growth_exit_heatmap(build_sim_params(mc, d, NO_FEES), mc, d)[2]

    post = ifrs(POST_IFRS16)
    as_if_none = dataclasses.replace(post, lease_view=POST_IFRS16, lease_liability=0.0, lease_cost=10.0)
    assert not np.allclose(grid(post), grid(as_if_none))
