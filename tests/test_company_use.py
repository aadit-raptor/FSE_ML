"""A company's summary as a deal's inputs and a forecast's history
(PLAN.md 4.1b, companies/use.py).

Real recorded filings (tests/fixtures/companies, tests/fixtures/edgar), each
figure checked by hand against the summary the connector reads; then the
rows go through the forecast and the deal through the API, so what the
screen fills is checked by the model's own answer.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from companies import companies_house, edinet, http, sec, sources, use
from companies.record import replay

FIXTURES = Path(__file__).parent / "fixtures" / "companies"
EDGAR = Path(__file__).parent / "fixtures" / "edgar"
E2E = Path(__file__).parent.parent / "web" / "e2e" / "fixtures"
TESCO_LEI = "2138002P5RNKC5W2JZ46"
client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_sources(monkeypatch):
    monkeypatch.setenv(companies_house.KEY_ENV, "k")
    monkeypatch.setenv(edinet.KEY_ENV, "k")
    sec.reset_cache()
    edinet.reset_cache()
    edinet.use_index(None)
    yield
    http.use_transport(None)
    edinet.use_index(None)
    sec.reset_cache()
    edinet.reset_cache()


def load(case: str, source: str, company_id: str):
    from tests.e2e_companies import replay_case
    replay_case(case)
    return sources.fetch(source, company_id)


def sap():
    """SAP SE through the SEC connector: its recorded company facts (the
    20-F, IFRS in euros) and a submissions answer naming it."""
    from tests.e2e_companies import replay_sap
    replay_sap()
    return sources.fetch("sec", "1000184")


# ---------------------------------------------------------------------------
# Deal inputs
# ---------------------------------------------------------------------------
def test_an_ifrs_filing_gives_the_same_deal_inputs_as_the_edgar_answer():
    edgar = json.loads((E2E / "edgar-sap.json").read_text(encoding="utf-8"))["deal_inputs"]
    got = use.deal_inputs(sap())
    assert got == {**edgar, "fiscal_year": 2025, "notes": []}


def test_an_ifrs_company_without_a_lease_cost_says_so():
    got = use.deal_inputs(load("esef_tesco", "esef", TESCO_LEI))
    assert got == {"ebitda": 2985.0 + 1895.0, "currency": "GBP", "unit": "millions", "accounting_standard": "ifrs",
                   "lease_cost": 0.0, "lease_liability": 7884.0, "fiscal_year": 2026,
                   "notes": ["lease_cost_missing"]}


def test_a_us_gaap_company_keeps_its_standard_and_leases():
    got = use.deal_inputs(load("sec_mcd", "sec", "63908"))
    assert got["accounting_standard"] == "us_gaap" and got["currency"] == "USD"
    assert got["ebitda"] == 12393.0 + 457.0 and got["lease_cost"] == 1200.0 and got["lease_liability"] == 0.0
    assert got["notes"] == []


def test_a_standard_the_deal_does_not_know_becomes_none_without_leases():
    data = load("edinet_nintendo", "edinet", "E02367")
    assert data.accounting_standard == "jgaap"
    leased = replace(data, years=[*data.years[:-1], replace(data.years[-1], figures={
        **data.years[-1].figures, "lease_cost": 5.0, "lease_liability": 50.0})])
    got = use.deal_inputs(leased)
    assert got["accounting_standard"] == "" and got["currency"] == "JPY"
    assert got["ebitda"] == 360117.0 + 15854.0
    assert (got["lease_cost"], got["lease_liability"]) == (0.0, 0.0)
    assert got["notes"] == ["standard_not_in_deal"]


def test_a_latest_year_without_ebitda_gives_no_deal_inputs():
    data = load("ch_cambridge_united", "companies_house", "00482197")
    assert data.years[-1].figures["ebitda"] is None
    assert use.deal_inputs(data) is None
    assert use.deal_inputs(replace(data, years=[])) is None


# ---------------------------------------------------------------------------
# Forecast history
# ---------------------------------------------------------------------------
def test_tescos_summary_fills_every_forecast_row_by_hand():
    rows = use.forecast_history(load("esef_tesco", "esef", TESCO_LEI))
    assert set(rows) == set(use.HISTORY_ROWS) and all(len(v) == 3 for v in rows.values())
    latest = {k: v[-1] for k, v in rows.items()}
    assert latest == {
        "h_rev": 73712.0,
        "h_cogs": -(73712.0 - 2985.0),               # every operating cost: EBIT is the filing's
        "h_rd": 0.0, "h_sga": 0.0, "h_int_inc": 0.0,
        "h_int_exp": -814.0,
        "h_tax": -616.0,
        "h_other": 1787.0 - (2985.0 - 814.0 - 616.0),  # net income is the filing's too
        "h_da": 1895.0, "h_sbc": 0.0,
        "h_cash": 2515.0, "h_ar": 1318.0, "h_inv": 2840.0, "h_ocurr": 0.0,
        "h_ppe": 39474.0 - 2515.0 - 1318.0 - 2840.0,   # the assets the summary doesn't name
        "h_nca": 0.0, "h_lta": 0.0,
        "h_ap": 10746.0, "h_ocl": 0.0, "h_def": 0.0,
        "h_ltd": 7196.0,
        "h_ncl": 39474.0 - 10746.0 - 7196.0 - 11457.0,  # the liabilities it doesn't name
        "h_cs": 11457.0, "h_re": 0.0, "h_oci": 0.0,
        "h_capex": 0.0,                                # Tesco tags none: a warning says so
        "h_divs": 0.0, "h_buybacks": 0.0,
    }
    assert rows["h_rev"] == [68187.0, 69916.0, 73712.0]


@pytest.mark.parametrize("case, source, company_id", [
    ("esef_tesco", "esef", TESCO_LEI), ("edinet_toyota", "edinet", "E02144"),
    ("edinet_nintendo", "edinet", "E02367"), ("sec_mcd", "sec", "63908"),
])
def test_the_history_balances_and_keeps_the_filings_profits(case, source, company_id):
    data = load(case, source, company_id)
    rows = use.forecast_history(data)
    for i, year in enumerate(data.years):
        f = {k: v or 0.0 for k, v in year.figures.items()}
        r = {k: v[i] for k, v in rows.items()}
        assets = r["h_cash"] + r["h_ar"] + r["h_inv"] + r["h_ocurr"] + r["h_ppe"] + r["h_nca"] + r["h_lta"]
        claims = r["h_ap"] + r["h_ocl"] + r["h_def"] + r["h_ltd"] + r["h_ncl"] + r["h_cs"] + r["h_re"] + r["h_oci"]
        assert assets == pytest.approx(f["total_assets"]) and claims == pytest.approx(assets)
        ebit = r["h_rev"] + r["h_cogs"] + r["h_rd"] + r["h_sga"]
        assert ebit == pytest.approx(f["operating_income"])
        assert ebit + r["h_da"] == pytest.approx(f["ebitda"])
        net = ebit + r["h_int_inc"] + r["h_int_exp"] + r["h_other"] + r["h_tax"]
        assert net == pytest.approx(f["net_income"])


def test_missing_total_assets_never_makes_ppe_negative():
    data = load("esef_tesco", "esef", TESCO_LEI)
    years = [replace(y, figures={**y.figures, "total_assets": None}) for y in data.years]
    rows = use.forecast_history(replace(data, years=years))
    assert rows["h_ppe"] == [0.0, 0.0, 0.0]
    latest = {k: v[-1] for k, v in rows.items()}
    assert latest["h_ncl"] == 2515.0 + 1318.0 + 2840.0 - 10746.0 - 7196.0 - 11457.0


def test_a_year_without_revenue_gives_no_forecast_history():
    data = load("ch_cambridge_united", "companies_house", "00482197")
    assert use.forecast_history(data) is None
    assert use.forecast_history(replace(data, years=[])) is None


def test_the_forecast_reads_the_filings_margins_from_the_rows():
    """Verify by output: the forecast's own historical ratios are Tesco's."""
    rows = use.forecast_history(load("esef_tesco", "esef", TESCO_LEI))
    body = {"history": rows, "money": {"currency": "GBP", "unit": "millions"}, "accounting_standard": "ifrs"}
    resp = client.post("/api/forecasting/seed", json=body)
    assert resp.status_code == 200, resp.text
    latest = resp.json()["historical_metrics"][-1]
    assert latest["revenue"] == 73712.0
    assert latest["ebitda_margin"] == pytest.approx(4880.0 / 73712.0)
    assert latest["revenue_growth"] == pytest.approx(73712.0 / 69916.0 - 1)
    seeded = {k: [v] * 5 for k, v in resp.json()["seeded_assumptions"].items()}
    run = client.post("/api/forecasting/run", json={**body, "simulate": False, "assumptions": seeded})
    assert run.status_code == 200, run.text
    answer = run.json()
    assert answer["ltm"]["revenue"] == 73712.0
    assert answer["opening_balance_gap"] == pytest.approx(0.0, abs=1e-6)   # the rows balance
    assert answer["years"][0]["revenue"] == pytest.approx(73712.0 * (1 + seeded["rev_g"][0] / 100))


def test_tescos_deal_inputs_run_as_a_deal():
    """Verify by output: the deal values Tesco on its EBITDA, in pounds."""
    got = use.deal_inputs(load("esef_tesco", "esef", TESCO_LEI))
    inputs = {k: got[k] for k in ("ebitda", "currency", "unit", "accounting_standard", "lease_cost", "lease_liability")}
    resp = client.post("/api/deal/run", json={"inputs": inputs})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["money"] == {"currency": "GBP", "unit": "millions"}
    assert body["leases"]["lease_liability"] == 7884.0


# ---------------------------------------------------------------------------
# The API answer
# ---------------------------------------------------------------------------
def test_the_load_answer_carries_deal_inputs_and_forecast_rows():
    from tests.e2e_companies import replay_case
    replay_case("esef_tesco")
    body = client.post("/api/companies/load", json={"source": "esef", "id": TESCO_LEI}).json()
    assert body["deal_inputs"]["ebitda"] == 4880.0 and body["deal_inputs"]["notes"] == ["lease_cost_missing"]
    assert body["forecast_history"]["h_rev"] == [68187.0, 69916.0, 73712.0]


def test_a_company_without_figures_answers_nulls():
    from tests.e2e_companies import replay_case
    replay_case("ch_cambridge_united")
    body = client.post("/api/companies/load", json={"source": "companies_house", "id": "00482197"}).json()
    assert body["deal_inputs"] is None and body["forecast_history"] is None


# ---------------------------------------------------------------------------
# The browser tests' recorded answers (web/e2e/fixtures/companies.json)
# ---------------------------------------------------------------------------
def test_the_browser_tests_replay_the_apis_current_answers():
    """The browser tests answer from a file; it must be what the API answers
    today for the recorded filings, or they would pass on a stale shape."""
    from tests.e2e_companies import OUT, build
    assert json.loads(OUT.read_text(encoding="utf-8")) == build(), \
        "web/e2e/fixtures/companies.json is stale: run python -m tests.e2e_companies"
