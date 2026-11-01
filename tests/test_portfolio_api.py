"""Portfolio tracking through the API (PLAN.md 7.4), on a real database: a GBP and a EUR holding totalled in US
dollars at stated ECB rates, below-plan quarters raising an alert, the reforecast as a job kept on the deal,
refusals, team roles, and one audit entry per change.

Every expected figure comes from the deal model's own answer (``POST /api/deal/run``) and the stated rates, worked
by hand here, never from the portfolio code.
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from api.auth import AuthUser, require_user
from api.main import app
from economy.model import FxDay
from jobs import config as jobs_config
from jobs.runner import Runner

client = TestClient(app)

PROFILE = {"country": "US", "preferred_currency": "USD", "locale": "en-US", "time_zone": "America/New_York"}
GBP = {"ebitda": 50_000.0, "entry_mult": 8.0, "exit_mult": 8.5, "hold": 5, "currency": "GBP", "unit": "thousands"}
EUR = {"ebitda": 80.0, "entry_mult": 10.0, "exit_mult": 10.0, "hold": 5, "currency": "EUR", "unit": "millions"}
# Units per euro, as the ECB publishes them, on a Friday
RATES = {"USD": 1.08, "GBP": 0.85, "JPY": 160.0}
RATE_DAY = date(2026, 10, 30)


def as_user(subject: str) -> None:
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject, is_dev=True)
    assert client.post("/api/account", json=PROFILE).status_code == 200


@pytest.fixture
def account(fresh_db):  # noqa: ARG001 -- the database
    from db import economy

    as_user("test:alice")
    economy.save_fx([FxDay(RATE_DAY, RATES)])
    return as_user


def saved(inputs: dict, name: str = "Holding") -> str:
    made = client.post("/api/deals", json={"name": name, "inputs": inputs, "settings": {}})
    assert made.status_code == 201, made.text
    return made.json()["id"]


def plan(inputs: dict) -> dict:
    answer = client.post("/api/deal/run", json={"inputs": inputs, "settings": {}})
    assert answer.status_code == 200, answer.text
    return answer.json()


def own(deal_id: str, **holding) -> dict:
    marked = client.put(f"/api/deals/{deal_id}/portfolio", json=holding)
    assert marked.status_code == 200, marked.text
    return marked.json()


def quarters(deal_id: str, rows: list[dict], currency: str, unit: str):
    return client.put(f"/api/deals/{deal_id}/quarters", json={"currency": currency, "unit": unit, "quarters": rows})


def shortfall_rows(inputs: dict, share: float, upto: int = 4, year: int = 1) -> list[dict]:
    """Quarters of ``year`` whose EBITDA is ``share`` of the plan's quarter (revenue on plan), in the deal's unit."""
    om = plan(inputs)["operating_model"]
    return [{"year": year, "quarter": q, "revenue": om["revenue"][year - 1] / 4,
             "ebitda": om["ebitda"][year - 1] / 4 * share} for q in range(1, upto + 1)]


def run_job():
    return Runner(jobs_config.get_queue).run_next()


# ------------------------------------------------------------------------------------------------ done when
def test_a_gbp_and_a_eur_portfolio_totals_in_usd_at_the_stated_rates(account):  # noqa: ARG001
    gbp, eur = saved(GBP, "Albion"), saved(EUR, "Rhein")
    own(gbp), own(eur)
    board = client.get("/api/portfolio", params={"currency": "USD", "on": "2026-11-01"})
    assert board.status_code == 200, board.text
    got = board.json()
    # Sunday the 1st: the last ECB publication before it, stated with its rates
    assert got["published_on"] == "2026-10-30"
    assert got["money"] == {"currency": "USD", "unit": "millions"}
    assert got["rates"] == {"EUR": pytest.approx(1 / 1.08), "GBP": pytest.approx(0.85 / 1.08)}
    assert got["missing_rates"] == []

    gbp_equity = plan(GBP)["returns"]["entry_equity"]           # thousands of pounds
    eur_equity = plan(EUR)["returns"]["entry_equity"]           # millions of euros
    by_name = {h["name"]: h for h in got["holdings"]}
    assert by_name["Albion"]["local"]["entry_equity"] == pytest.approx(gbp_equity)
    assert by_name["Albion"]["money"] == {"currency": "GBP", "unit": "thousands"}
    # GBP thousands to USD millions: / 1,000, / 0.85 pounds a euro, x 1.08 dollars a euro
    assert by_name["Albion"]["converted"]["entry_equity"] == pytest.approx(gbp_equity / 1000 / 0.85 * 1.08)
    assert by_name["Rhein"]["converted"]["entry_equity"] == pytest.approx(eur_equity * 1.08)
    total = got["totals"]["entry_equity"]
    assert total["holdings"] == 2
    assert total["value"] == pytest.approx(gbp_equity / 1000 / 0.85 * 1.08 + eur_equity * 1.08)
    # entry value is EBITDA x the entry multiple, in each deal's money
    assert got["totals"]["entry_ev"]["value"] == pytest.approx(50 * 8 / 0.85 * 1.08 + 80 * 10 * 1.08)
    # the plan's IRR is the deal screen's
    assert by_name["Rhein"]["plan_irr"] == pytest.approx(plan(EUR)["returns"]["irr"])

    # another reporting currency and unit: the same holdings in thousands of pounds
    pounds = client.get("/api/portfolio", params={"currency": "GBP", "unit": "thousands"}).json()
    assert pounds["totals"]["entry_equity"]["value"] == pytest.approx(gbp_equity + eur_equity * 0.85 * 1000)
    assert pounds["rates"] == {"EUR": pytest.approx(1 / 0.85)}


def test_below_plan_actuals_raise_an_alert(account):  # noqa: ARG001
    deal = saved(EUR, "Rhein")
    own(deal, tolerance=5.0)
    assert quarters(deal, shortfall_rows(EUR, 0.85, upto=2), "EUR", "millions").status_code == 200
    held = client.get(f"/api/deals/{deal}/portfolio").json()
    ebitda_1 = plan(EUR)["operating_model"]["ebitda"][0]
    assert (held["drift"]["year"], held["drift"]["quarter"]) == (1, 2)
    assert [a["id"] for a in held["drift"]["alerts"]] == ["ebitda_below_plan"]
    alert = held["drift"]["alerts"][0]
    assert alert["actual"] == pytest.approx(ebitda_1 / 2 * 0.85)
    assert alert["plan"] == pytest.approx(ebitda_1 / 2)
    assert alert["gap_pct"] == pytest.approx(-15.0)
    board = client.get("/api/portfolio", params={"currency": "USD"}).json()
    assert board["alerts"] == 1
    assert [a["id"] for a in board["holdings"][0]["alerts"]] == ["ebitda_below_plan"]
    # on plan within the tolerance: no alert
    assert quarters(deal, shortfall_rows(EUR, 0.97, upto=2), "EUR", "millions").status_code == 200
    assert client.get(f"/api/deals/{deal}/portfolio").json()["drift"]["alerts"] == []
    assert client.get("/api/portfolio").json()["alerts"] == 0


def test_quarters_in_another_unit_are_compared_in_the_deals(account):  # noqa: ARG001
    deal = saved(EUR)
    own(deal)
    rows = [{**r, "revenue": r["revenue"] * 1000, "ebitda": r["ebitda"] * 1000} for r in shortfall_rows(EUR, 0.8, upto=1)]
    assert quarters(deal, rows, "EUR", "thousands").status_code == 200
    alert = client.get(f"/api/deals/{deal}/portfolio").json()["drift"]["alerts"][0]
    assert alert["plan"] == pytest.approx(plan(EUR)["operating_model"]["ebitda"][0] / 4)     # millions, the deal's


# ------------------------------------------------------------------------------------------------ reforecast
def test_a_reforecast_runs_as_a_job_and_is_kept_on_the_deal(account):  # noqa: ARG001
    deal = saved(EUR, "Rhein")
    own(deal)
    assert quarters(deal, shortfall_rows(EUR, 0.9), "EUR", "millions").status_code == 200
    started = client.post(f"/api/deals/{deal}/reforecast")
    assert started.status_code == 202, started.text
    assert run_job() == "succeeded"
    done = client.get(f"/api/deals/{deal}/reforecast/jobs/{started.json()['id']}").json()
    assert done["status"] == "succeeded", done
    result = done["result"]
    # one year since close at 90% of plan: (1 + g') = 1.05 x 0.9
    assert result["reforecast_growth"] == pytest.approx((1.05 * 0.9 - 1) * 100, abs=1e-3)
    assert result["years"][0]["reforecast_ebitda"] == pytest.approx(plan(EUR)["operating_model"]["ebitda"][0] * 0.9,
                                                                     abs=0.02)
    assert result["reforecast"]["irr"] < result["plan"]["irr"]
    assert result["money"] == {"currency": "EUR", "unit": "millions"}

    held = client.get(f"/api/deals/{deal}/portfolio").json()
    assert held["reforecast"]["reforecast_growth"] == pytest.approx(result["reforecast_growth"])
    assert held["reforecast_current"] is True
    board = client.get("/api/portfolio", params={"currency": "EUR"}).json()
    assert board["holdings"][0]["reforecast_irr"] == pytest.approx(result["reforecast"]["irr"])
    assert board["totals"]["reforecast_exit_equity"]["value"] == pytest.approx(result["reforecast"]["exit_equity"])
    # new quarters make the reforecast out of date until it runs again
    assert quarters(deal, shortfall_rows(EUR, 0.95), "EUR", "millions").status_code == 200
    assert client.get(f"/api/deals/{deal}/portfolio").json()["reforecast_current"] is False


def test_a_reforecast_needs_four_quarters(account):  # noqa: ARG001
    deal = saved(EUR)
    own(deal)
    assert client.post(f"/api/deals/{deal}/reforecast").json()["detail"] == {"code": "no_ltm"}
    assert quarters(deal, shortfall_rows(EUR, 0.9, upto=3), "EUR", "millions").status_code == 200
    refused = client.post(f"/api/deals/{deal}/reforecast")
    assert refused.status_code == 422 and refused.json()["detail"] == {"code": "no_ltm"}


# ------------------------------------------------------------------------------------------------ refusals and roles
def test_quarters_must_be_in_the_deals_currency_and_inside_its_hold(account):  # noqa: ARG001
    deal = saved(EUR)
    other = quarters(deal, shortfall_rows(EUR, 1.0, upto=1), "USD", "millions")
    assert other.status_code == 422 and other.json()["detail"] == {"code": "other_currency"}
    late = quarters(deal, [{"year": 6, "quarter": 1, "ebitda": 10.0}], "EUR", "millions")
    assert late.status_code == 422 and late.json()["detail"] == {"code": "beyond_hold"}
    twice = quarters(deal, [{"year": 1, "quarter": 1}, {"year": 1, "quarter": 1}], "EUR", "millions")
    assert twice.status_code == 422
    # the deal's hold shortened after the quarters were saved: they are kept and said not to fit
    assert quarters(deal, [{"year": 3, "quarter": 1, "ebitda": 10.0}], "EUR", "millions").status_code == 200
    assert client.put(f"/api/deals/{deal}/draft", json={"inputs": {**EUR, "hold": 2}, "settings": {}}).status_code == 200
    held = client.get(f"/api/deals/{deal}/portfolio").json()
    assert held["quarters_problem"] == "beyond_hold" and held["drift"] is None
    # and in another currency since: kept, said, and left out of the dashboard's figures
    assert client.put(f"/api/deals/{deal}/draft", json={"inputs": {**EUR, "currency": "GBP"}, "settings": {}}).status_code == 200
    own(deal)
    assert client.get(f"/api/deals/{deal}/portfolio").json()["quarters_problem"] == "other_currency"
    row = client.get("/api/portfolio").json()["holdings"][0]
    assert row["quarters_problem"] == "other_currency" and row["as_of"] is None


def test_another_accounts_deal_is_not_found(account):
    deal = saved(EUR)
    own(deal)
    account("test:bob")
    for answer in (client.get(f"/api/deals/{deal}/portfolio"), client.put(f"/api/deals/{deal}/portfolio", json={}),
                   client.delete(f"/api/deals/{deal}/portfolio"),
                   quarters(deal, shortfall_rows(EUR, 1.0, upto=1), "EUR", "millions"),
                   client.delete(f"/api/deals/{deal}/quarters"), client.post(f"/api/deals/{deal}/reforecast")):
        assert answer.status_code == 404, answer.text
    assert client.get("/api/portfolio").json()["holdings"] == []


def test_a_team_viewer_reads_a_shared_holding_and_changes_nothing(account):
    deal = saved(EUR, "Rhein")
    own(deal)
    team = client.post("/api/teams", json={"name": "Fund"}).json()["id"]
    token = client.post(f"/api/teams/{team}/invites", json={"role": "viewer", "uses": 1}).json()["token"]
    assert client.put(f"/api/deals/{deal}/shares", json={"team_id": team}).status_code == 200
    account("test:vera")
    assert client.post("/api/teams/join", json={"token": token}).status_code in (200, 201)
    assert client.get(f"/api/deals/{deal}/portfolio").json()["access"] == "viewer"
    assert [h["name"] for h in client.get("/api/portfolio").json()["holdings"]] == ["Rhein"]
    for answer in (client.put(f"/api/deals/{deal}/portfolio", json={"tolerance": 1.0}),
                   client.delete(f"/api/deals/{deal}/portfolio"),
                   quarters(deal, shortfall_rows(EUR, 1.0, upto=1), "EUR", "millions"),
                   client.post(f"/api/deals/{deal}/reforecast")):
        assert answer.status_code == 403, answer.text


def test_a_holding_without_a_rate_is_named_and_left_out(account):  # noqa: ARG001
    own(saved({**EUR, "currency": "CHF"}, "Alpen"))
    own(saved(EUR, "Rhein"))
    got = client.get("/api/portfolio", params={"currency": "USD"}).json()
    assert got["missing_rates"] == ["CHF"]
    assert got["totals"]["entry_equity"]["holdings"] == 1
    assert next(h for h in got["holdings"] if h["name"] == "Alpen")["converted"] is None


def test_archived_and_unmarked_deals_are_not_in_the_portfolio(account):  # noqa: ARG001
    kept, gone, never = saved(EUR, "Kept"), saved(EUR, "Gone"), saved(EUR, "Never")
    own(kept), own(gone)
    assert client.patch(f"/api/deals/{gone}", json={"archived": True}).status_code == 200
    assert [h["name"] for h in client.get("/api/portfolio").json()["holdings"]] == ["Kept"]
    out = client.delete(f"/api/deals/{kept}/portfolio").json()
    assert out["owned"] is False
    assert client.get("/api/portfolio").json()["holdings"] == []
    assert never


# ------------------------------------------------------------------------------------------------ history
def test_one_audit_entry_per_change(account):  # noqa: ARG001
    deal = saved(EUR)
    own(deal)
    own(deal)                                                   # the same again: nothing changed
    own(deal, tolerance=10.0)
    rows = shortfall_rows(EUR, 0.9)
    quarters(deal, rows, "EUR", "millions")
    quarters(deal, rows, "EUR", "millions")
    client.post(f"/api/deals/{deal}/reforecast")
    run_job()
    client.delete(f"/api/deals/{deal}/quarters")
    client.delete(f"/api/deals/{deal}/portfolio")
    client.delete(f"/api/deals/{deal}/portfolio")
    actions = [e["action"] for e in client.get(f"/api/deals/{deal}/history").json()["entries"]]
    assert actions[:6] == ["portfolio_removed", "quarters_cleared", "reforecast_run", "quarters_saved",
                           "portfolio_changed", "portfolio_added"]
