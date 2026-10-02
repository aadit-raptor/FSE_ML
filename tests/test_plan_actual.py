"""Plan vs actual for any deal (PLAN.md 2.7).

Done when: a user's own saved deal is backtested end to end (here through the
API and a real database; web/e2e/backtest.spec.ts through the screen); with
the library off the backtest still works; attribution still adds up.

The plan side is checked against the deal model's own answer (/api/deal/run),
and the actual side by feeding the plan's own projections back in as actuals:
then every variance and every part of the attribution must be zero.
"""
import pytest
from fastapi.testclient import TestClient

from api.auth import AuthUser, require_user
from api.main import app
from core.money import in_unit

client = TestClient(app)

PATH = "/api/backtesting/plan-vs-actual"
LINES = ("revenue", "ebitda", "net_income", "fcf", "total_debt")
# A deal nothing like the defaults, as a user would save it
PLAN = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
        "gross_margin": 44.0, "opex": 17.0, "tax": 29.0, "da": 3.5, "debt_pct": 55.0,
        "senior_pct": 75.0, "base_rate": 5.25, "mezz_spread": 4.5, "capex": 3.0, "nwc": 0.5,
        "mincash": 15.0, "currency": "EUR", "unit": "millions"}
SETTINGS = {"tx_fee_pct": 3.0}
N = 2000


def post(body: dict, status: int = 200) -> dict:
    resp = client.post(PATH, json=body)
    assert resp.status_code == status, resp.text
    return resp.json()


def deal_run(plan: dict, settings: dict = SETTINGS) -> dict:
    resp = client.post("/api/deal/run", json={"inputs": plan, "settings": settings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def plan_as_actuals(plan: dict, settings: dict = SETTINGS, years: int | None = None,
                    sold: bool = True) -> dict:
    """The plan's own projections typed in as if they had happened."""
    run = deal_run(plan, settings)
    om, cf, debt = run["operating_model"], run["cash_flow"], run["debt_schedule"]
    n = years or plan["hold"]
    rows = [{"revenue": om["revenue"][i], "ebitda": om["ebitda"][i], "net_income": om["net_income"][i],
             "fcf": cf["levered_fcf"][i],
             "total_debt": debt["total_ending_debt"][i]} for i in range(n)]
    actuals = {"currency": plan["currency"], "unit": plan["unit"], "years": rows}
    if sold:
        r = run["returns"]
        actuals["exit"] = {"exit_ev": r["exit_ev"], "net_debt_at_exit": r["net_debt_at_exit"],
                           "sponsor_equity_entry": r["entry_equity"]}
    return actuals


def body(plan=PLAN, actuals=None, settings=SETTINGS, **extra) -> dict:
    return {"plan": plan, "settings": settings,
            "actuals": actuals if actuals is not None else plan_as_actuals(plan, settings), "n": N, **extra}


def attribution_total(answer: dict) -> float:
    return sum(answer["attribution"].values())


# ---------------------------------------------------------------------------
# The plan is the deal model's own answer for the deal
# ---------------------------------------------------------------------------
def test_the_plan_side_is_the_deal_models_answer_for_the_saved_deal():
    run = deal_run(PLAN)
    answer = post(body())
    assert [y["plan_revenue"] for y in answer["years"]] == run["operating_model"]["revenue"]
    assert [y["plan_ebitda"] for y in answer["years"]] == run["operating_model"]["ebitda"]
    assert [y["plan_total_debt"] for y in answer["years"]] == run["debt_schedule"]["total_ending_debt"]
    returns = run["returns"]
    assert answer["plan"]["irr"] == pytest.approx(returns["irr"], abs=1e-12)
    assert answer["plan"]["moic"] == pytest.approx(returns["moic"], abs=1e-12)
    assert answer["plan"]["exit_ev"] == returns["exit_ev"]
    assert answer["plan"]["exit_equity"] == pytest.approx(returns["exit_ev"] - returns["net_debt_at_exit"])
    # The settings are the deal's: without them the plan is a different one
    assert post(body(settings={}))["plan"]["irr"] != pytest.approx(answer["plan"]["irr"], abs=1e-4)


def test_actuals_equal_to_the_plan_have_no_variance_and_nothing_to_attribute():
    answer = post(body())
    for year in answer["years"]:
        for line in LINES:
            assert year[f"variance_{line}"] == pytest.approx(0.0, abs=1e-9), (year["year_index"], line)
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0.0, "exit_multiple": 0.0, "net_debt": 0.0},
                                                  abs=1e-9)
    actual, plan = answer["actual"], answer["plan"]
    assert actual["exit_multiple"] == pytest.approx(plan["exit_multiple"], rel=1e-4)
    # The MOIC and IRR left out are computed from equity in and out (the
    # engine rounds its MOIC to four places)
    assert not actual["irr_given"] and not actual["moic_given"]
    assert actual["moic"] == pytest.approx(plan["moic"], abs=1e-4)
    assert actual["irr"] == pytest.approx(plan["irr"], abs=1e-4)
    # The plan's own IRR sits inside its simulated range
    assert 5 < actual["percentile"] < 95
    assert plan["irr_p5"] < plan["irr"] < plan["irr_p95"]


# ---------------------------------------------------------------------------
# Attribution adds up, and each part is the part it says
# ---------------------------------------------------------------------------
def test_the_attribution_adds_up_to_the_exit_equity_gap_and_each_part_is_its_own():
    actuals = plan_as_actuals(PLAN)
    actuals["years"][-1]["ebitda"] *= 1.2
    actuals["exit"] = {"exit_ev": 3100.0, "net_debt_at_exit": 900.0, "sponsor_equity_entry": 1000.0}
    answer = post(body(actuals=actuals))
    gap = answer["actual"]["exit_equity"] - answer["plan"]["exit_equity"]
    assert attribution_total(answer) == pytest.approx(gap, abs=1e-9)
    plan, actual, parts = answer["plan"], answer["actual"], answer["attribution"]
    assert parts["exit_ebitda"] == pytest.approx(
        (actual["exit_ebitda"] - plan["exit_ebitda"]) * plan["exit_multiple"], rel=1e-12)
    # The engine rounds its EV to the cent; the multiple part absorbs that
    assert parts["exit_multiple"] == pytest.approx(
        (actual["exit_multiple"] - plan["exit_multiple"]) * actual["exit_ebitda"], abs=0.01)
    assert parts["net_debt"] == pytest.approx(plan["net_debt_at_exit"] - 900.0)
    assert actual["exit_equity"] == 2200.0


def test_an_entered_irr_and_moic_are_kept_as_given():
    actuals = plan_as_actuals(PLAN)
    actuals["exit"].update(moic=2.5, irr=17.0)
    actual = post(body(actuals=actuals))["actual"]
    assert actual["moic"] == 2.5 and actual["irr"] == pytest.approx(0.17)
    assert actual["moic_given"] and actual["irr_given"]


def test_a_computed_irr_is_the_moic_over_the_years_held():
    actuals = plan_as_actuals(PLAN)
    actuals["exit"] = {"exit_ev": 4000.0, "net_debt_at_exit": 1000.0, "sponsor_equity_entry": 1000.0}
    actual = post(body(actuals=actuals))["actual"]
    assert actual["moic"] == pytest.approx(3.0)
    assert actual["irr"] == pytest.approx(3.0 ** (1 / 6) - 1)


# ---------------------------------------------------------------------------
# Deals still held, deals sold early
# ---------------------------------------------------------------------------
def test_a_deal_still_held_compares_its_years_so_far():
    answer = post(body(actuals=plan_as_actuals(PLAN, years=3, sold=False)))
    assert answer["years_compared"] == 3 and len(answer["years"]) == 3
    assert answer["exit_year"] is None and answer["actual"] is None and answer["attribution"] is None
    # The plan's returns stay the plan's own, at its full hold
    assert answer["plan"]["irr"] == pytest.approx(deal_run(PLAN)["returns"]["irr"], abs=1e-12)


def test_an_early_exit_is_compared_with_the_plan_sold_that_year():
    actuals = plan_as_actuals({**PLAN, "hold": 4}, years=4)
    answer = post(body(actuals=actuals))
    sold_early = deal_run({**PLAN, "hold": 4})["returns"]
    assert answer["exit_year"] == 4
    assert answer["plan"]["irr"] == pytest.approx(sold_early["irr"], abs=1e-12)
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0, "exit_multiple": 0, "net_debt": 0}, abs=1e-9)
    # The IRR worked out from the MOIC runs over the four years actually held
    assert answer["actual"]["irr"] == pytest.approx(sold_early["irr"], abs=1e-4)


def test_the_plans_free_cash_flow_is_before_debt_repayment():
    """A reported free cash flow never has the loan repayments taken off."""
    plan = {**PLAN, "tranches": [{"name": "Term loan A", "kind": "amortising_term_loan", "amount": 900.0,
                                  "fixed_rate": 6.0, "amort_pct": 10.0}]}
    run = deal_run(plan)
    cf = run["cash_flow"]
    assert min(run["debt_schedule"]["total_mandatory_repayment"]) > 0
    before_repayment = [ni + da + pik - capex - nwc for ni, da, pik, capex, nwc in zip(
        cf["net_income"], cf["da"], cf["non_cash_interest"], cf["capex"], cf["delta_nwc"])]
    years = post(body(plan=plan, actuals=plan_as_actuals(plan, sold=False)))["years"]
    assert [y["plan_fcf"] for y in years] == pytest.approx(before_repayment, abs=0.02)


def test_the_plan_is_simulated_around_its_own_assumptions():
    """Its growth, exit multiple, rate and margin are the means, so a
    faster-growing plan's whole distribution moves up with it."""
    slow, fast = (post(body(plan={**PLAN, "growth": g}, actuals=plan_as_actuals({**PLAN, "growth": g}, sold=False)))["plan"]
                  for g in (2.0, 12.0))
    assert fast["irr_mean"] > slow["irr_mean"] + 0.03
    for plan in (slow, fast):
        assert plan["irr_mean"] == pytest.approx(plan["irr"], abs=0.03)


def test_unknown_figures_have_no_variance():
    actuals = plan_as_actuals(PLAN, sold=False)
    del actuals["years"][1]["net_income"]
    actuals["years"][2]["fcf"] = None
    years = post(body(actuals=actuals))["years"]
    assert years[1]["actual_net_income"] is None and years[1]["variance_net_income"] is None
    assert years[2]["variance_fcf"] is None
    assert years[1]["variance_revenue"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Any deal: leases, facilities, money units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extra", [
    {"accounting_standard": "ifrs", "lease_cost": 20.0, "lease_liability": 120.0},
    {"accounting_standard": "us_gaap", "lease_view": "post_ifrs16", "lease_cost": 20.0, "lease_liability": 120.0},
    {"tranches": [
        {"name": "Term loan B", "kind": "institutional_term_loan", "amount": 900.0, "floating": True,
         "reference_rate": "EURIBOR", "reference_level": 4.0, "margin": 3.5, "floor": 0.5, "sweep": True},
        {"name": "PIK notes", "kind": "pik_notes", "amount": 250.0, "fixed_rate": 11.0, "pik_share": 100.0,
         "maturity_years": 8}]},
], ids=["ifrs-leases", "us-gaap-post-view", "tranches"])
def test_any_deal_compared_with_its_own_projections_attributes_nothing(extra):
    plan = {**PLAN, **extra}
    answer = post(body(plan=plan, actuals=plan_as_actuals(plan)))
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0, "exit_multiple": 0, "net_debt": 0}, abs=1e-9)
    assert answer["plan"]["irr"] == pytest.approx(deal_run(plan)["returns"]["irr"], abs=1e-12)
    if "lease_cost" in extra:
        assert answer["lease_addback"] == pytest.approx(20.0)


def test_a_deal_in_thousands_answers_as_in_millions_times_a_thousand():
    millions = post(body())
    plan_k = {**PLAN, "unit": "thousands", "ebitda": 240_000.0, "mincash": 15_000.0}
    settings_k = {**SETTINGS}
    answer = post(body(plan=plan_k, settings=settings_k, actuals=plan_as_actuals(plan_k, settings_k)))
    assert answer["money"] == {"currency": "EUR", "unit": "thousands"}
    assert answer["plan"]["irr"] == pytest.approx(millions["plan"]["irr"], abs=1e-12)
    assert answer["plan"]["exit_equity"] == pytest.approx(millions["plan"]["exit_equity"] * 1000)
    assert answer["years"][0]["plan_revenue"] == pytest.approx(millions["years"][0]["plan_revenue"] * 1000)
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0, "exit_multiple": 0, "net_debt": 0}, abs=1e-6)


def test_actuals_in_another_unit_are_converted_to_the_deals():
    actuals = plan_as_actuals(PLAN)
    k = in_unit(1.0, "thousands")
    in_thousands = {**actuals, "unit": "thousands",
                    "years": [{line: v * k for line, v in y.items()} for y in actuals["years"]],
                    "exit": {key: v * k for key, v in actuals["exit"].items()}}
    answer = post(body(actuals=in_thousands))
    assert answer["money"]["unit"] == "millions"
    assert answer["years"][0]["actual_revenue"] == pytest.approx(actuals["years"][0]["revenue"])
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0, "exit_multiple": 0, "net_debt": 0}, abs=1e-6)


def test_every_money_figure_is_converted_and_nothing_else(monkeypatch):
    """A figure missing from PLAN_ACTUAL_MONEY_KEYS would come back in millions."""
    import api.routers.backtesting as bt_router
    from tests.test_money import assert_same_leaves, raw_engine, without_money
    plan_k = {**PLAN, "unit": "thousands", "ebitda": 240_000.0, "mincash": 15_000.0}
    # Ten per cent off plan everywhere, so no variance is a rounding-sized zero
    plan_figures = plan_as_actuals(plan_k)
    actuals = {**plan_figures,
               "years": [{line: v * 1.1 for line, v in y.items()} for y in plan_figures["years"]],
               "exit": {key: v * 1.1 for key, v in plan_figures["exit"].items()}}
    actuals["exit"]["exit_ev"] *= 1.2   # and a different multiple
    request = body(plan=plan_k, actuals=actuals)
    converted = without_money(post(request))
    raw_engine(monkeypatch, bt_router)
    monkeypatch.setattr(bt_router, "in_millions", lambda d, cfg: (d, cfg))
    monkeypatch.setattr(bt_router, "actuals_in_millions", lambda a, unit: a)
    raw = without_money(post(request))
    # The engine rounds to a hundredth of the unit it runs in, so "raw" (run
    # in thousands) differs from "converted" (run in millions) in the last
    # places, never by a factor of a thousand
    assert_same_leaves(converted, raw, rel=1e-2)


# ---------------------------------------------------------------------------
# Refusals: answered with a sentence, never a 500
# ---------------------------------------------------------------------------
def test_actuals_in_another_currency_are_refused():
    actuals = {**plan_as_actuals(PLAN), "currency": "USD"}
    detail = post(body(actuals=actuals), status=422)["detail"]
    assert "USD" in detail and "EUR" in detail


def test_more_years_than_the_plan_holds_are_refused():
    actuals = plan_as_actuals({**PLAN, "hold": 7})
    assert "1 to 6 years" in post(body(actuals=actuals), status=422)["detail"]


def test_an_exit_without_the_exit_years_ebitda_is_refused():
    actuals = plan_as_actuals(PLAN)
    actuals["years"][-1]["ebitda"] = None
    assert "EBITDA" in post(body(actuals=actuals), status=422)["detail"]


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_non_numbers_are_refused_at_the_edge(bad):
    raw = (f'{{"plan": {{"currency": "USD"}}, "actuals": {{"currency": "USD", "unit": "millions", '
           f'"years": [{{"revenue": {bad}}}]}}}}')
    resp = client.post(PATH, content=raw, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_an_unfinanceable_plan_is_a_422_not_a_500():
    plan = {**PLAN, "tranches": [{"name": "Too much", "kind": "senior_notes", "amount": 99_999.0,
                                  "fixed_rate": 8.0}]}
    actuals = {"currency": "EUR", "unit": "millions", "years": [{"revenue": 1.0}]}
    assert client.post(PATH, json=body(plan=plan, actuals=actuals)).status_code == 422


# ---------------------------------------------------------------------------
# The example library is optional
# ---------------------------------------------------------------------------
def test_each_example_runs_as_a_plan_and_its_attribution_adds_up():
    library = client.get("/api/backtesting/examples").json()
    assert library["enabled"] and len(library["examples"]) == 4
    for example in library["examples"]:
        answer = post({"plan": example["plan"], "actuals": example["actuals"], "n": N})
        gap = answer["actual"]["exit_equity"] - answer["plan"]["exit_equity"]
        assert attribution_total(answer) == pytest.approx(gap, abs=1e-9), example["name"]
        assert answer["money"] == {"currency": "USD", "unit": "millions"}
        assert answer["actual"]["irr_given"], "the example's own IRR is shown as recorded"


def test_an_examples_plan_is_the_deal_model_run_the_old_backtest_predicted_with():
    """Continuity with finding 5: the old backtest's predicted exit came from
    this same deal-model run on the example's entry assumptions."""
    from core.backtesting import PRELOADED_DEALS, prediction_lbo_params
    from core.config import resolve_config
    from lbo_engine.model import run_lbo
    for example in client.get("/api/backtesting/examples").json()["examples"]:
        old = run_lbo(prediction_lbo_params(PRELOADED_DEALS[example["name"]]["entry"], resolve_config()))
        plan = post({"plan": example["plan"], "actuals": example["actuals"], "n": N})["plan"]
        assert plan["exit_ev"] == pytest.approx(old.returns.exit_ev, abs=1e-9), example["name"]
        assert plan["irr"] == pytest.approx(old.returns.irr, abs=1e-12), example["name"]


def test_with_the_library_off_the_backtest_still_works(monkeypatch):
    monkeypatch.setenv("FSE_EXAMPLE_LIBRARY", "0")
    assert client.get("/api/backtesting/examples").json() == {"enabled": False, "examples": []}
    assert client.get("/api/backtesting/deals").json() == []
    answer = post(body())
    assert answer["attribution"] == pytest.approx({"exit_ebitda": 0, "exit_multiple": 0, "net_debt": 0}, abs=1e-9)


# ---------------------------------------------------------------------------
# Saved actuals: a real database, end to end
# ---------------------------------------------------------------------------
PROFILE = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE", "time_zone": "Europe/Berlin"}


@pytest.fixture
def sign_in():
    def as_user(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        assert client.post("/api/account", json=PROFILE).status_code == 200
    yield as_user


def test_a_saved_deal_is_backtested_end_to_end(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    created = client.post("/api/deals", json={"name": "Project Alpine", "inputs": PLAN, "settings": SETTINGS})
    assert created.status_code == 201, created.text
    deal_id = created.json()["id"]
    assert client.get(f"/api/deals/{deal_id}/actuals").json() == {"actuals": None, "updated_at": None}

    actuals = plan_as_actuals(PLAN)
    actuals["years"][-1]["ebitda"] *= 1.15
    actuals["years"][0]["net_income"] = None
    saved = client.put(f"/api/deals/{deal_id}/actuals", json=actuals)
    assert saved.status_code == 200, saved.text
    assert saved.json()["updated_at"]

    # Reopened: the saved deal is the plan, its stored actuals what happened
    deal = client.get(f"/api/deals/{deal_id}").json()
    stored = client.get(f"/api/deals/{deal_id}/actuals").json()["actuals"]
    assert stored["years"][-1]["ebitda"] == pytest.approx(actuals["years"][-1]["ebitda"])
    assert stored["years"][0]["net_income"] is None
    answer = post({"plan": deal["inputs"], "settings": deal["settings"], "actuals": stored, "n": N})
    assert answer["plan"]["irr"] == pytest.approx(deal_run(PLAN)["returns"]["irr"], abs=1e-12)
    assert answer["attribution"]["exit_ebitda"] > 0
    gap = answer["actual"]["exit_equity"] - answer["plan"]["exit_equity"]
    assert attribution_total(answer) == pytest.approx(gap, abs=1e-9)


def test_actuals_are_not_a_version_of_the_plan(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = client.post("/api/deals", json={"name": "Alpine", "inputs": PLAN}).json()
    client.put(f"/api/deals/{deal['id']}/actuals", json=plan_as_actuals(PLAN, sold=False))
    after = client.get(f"/api/deals/{deal['id']}").json()
    assert after["latest_version"] == deal["latest_version"]
    assert after["updated_at"] == deal["updated_at"]
    assert client.get(f"/api/deals/{deal['id']}/versions").json()["versions"].__len__() == 1


def test_actuals_belong_to_the_deals_owner(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = client.post("/api/deals", json={"name": "Alpine", "inputs": PLAN}).json()
    client.put(f"/api/deals/{deal['id']}/actuals", json=plan_as_actuals(PLAN, sold=False))
    sign_in("user_ben")
    assert client.get(f"/api/deals/{deal['id']}/actuals").status_code == 404
    assert client.put(f"/api/deals/{deal['id']}/actuals",
                      json=plan_as_actuals(PLAN, sold=False)).status_code == 404
    assert client.delete(f"/api/deals/{deal['id']}/actuals").status_code == 404
    sign_in("user_anna")
    assert client.get(f"/api/deals/{deal['id']}/actuals").json()["actuals"] is not None


def test_actuals_can_be_cleared_and_bad_ones_are_refused(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = client.post("/api/deals", json={"name": "Alpine", "inputs": PLAN}).json()
    url = f"/api/deals/{deal['id']}/actuals"
    client.put(url, json=plan_as_actuals(PLAN, sold=False))
    assert client.delete(url).json() == {"actuals": None, "updated_at": None}
    assert client.get(url).json()["actuals"] is None
    too_many = {"currency": "EUR", "unit": "millions", "years": [{"revenue": 1.0}] * 16}
    assert client.put(url, json=too_many).status_code == 422
    assert client.put(url, json={"currency": "EUR", "unit": "millions", "years": []}).status_code == 422


def test_stored_actuals_leave_unknown_figures_out():
    from db.deals import clean_actuals
    cleaned = clean_actuals({"currency": "EUR", "unit": "millions",
                             "years": [{"revenue": 10.0, "ebitda": None}]})
    assert cleaned == {"currency": "EUR", "unit": "millions", "years": [{"revenue": 10.0}]}
