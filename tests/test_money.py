"""Currency and money units (PLAN.md 2.2).

Done when: the same inputs give identical results in any currency; every
response and export says which currency and unit its money is in. Plus: a deal
entered in thousands or billions gives the same answer as in millions, scaled.
"""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.forecasting import default_history
from core.money import Money, in_unit

client = TestClient(app)

CURRENCIES = ["USD", "EUR", "JPY", "INR", "GBP", "CHF", "BRL"]
SCALE = {"thousands": 1e3, "millions": 1.0, "billions": 1e-3}   # figures per million
MONEY_INPUTS = ("ebitda", "mincash")
MC = {"n": 2000}


def post(path: str, body: dict) -> dict:
    resp = client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def without_money(answer: dict) -> dict:
    return {k: v for k, v in answer.items() if k not in ("money", "elapsed_ms")}


def deal(currency="USD", unit="millions", **inputs) -> dict:
    return {"ebitda": 100.0, "mincash": 10.0, **inputs, "currency": currency, "unit": unit}


def scaled(inputs: dict, unit: str) -> dict:
    return {**inputs, **{k: inputs[k] * SCALE[unit] for k in MONEY_INPUTS}, "unit": unit}


# ---------------------------------------------------------------------------
# Done when: the same inputs give identical results in any currency
# ---------------------------------------------------------------------------
def test_a_deal_gives_identical_results_in_any_currency():
    baseline = post("/api/deal/run", {"inputs": deal("USD")})
    for currency in CURRENCIES[1:]:
        answer = post("/api/deal/run", {"inputs": deal(currency)})
        assert without_money(answer) == without_money(baseline), currency
        assert answer["money"] == {"currency": currency, "unit": "millions"}


def test_a_simulation_gives_identical_results_in_any_currency():
    body = {"mc": MC, "seed": 7}
    usd = post("/api/montecarlo/run", {**body, "deal": deal("USD")})
    eur = post("/api/montecarlo/run", {**body, "deal": deal("EUR")})
    assert without_money(eur) == without_money(usd)
    assert eur["money"]["currency"] == "EUR"
    scenarios = post("/api/montecarlo/scenarios", {**body, "deal": deal("JPY")})
    assert scenarios["money"] == {"currency": "JPY", "unit": "millions"}


def test_a_forecast_gives_identical_results_in_any_currency():
    seeded = post("/api/forecasting/seed", {"history": default_history(3)})
    body = {"history": default_history(3),
            "assumptions": {k: [v] * 5 for k, v in seeded["seeded_assumptions"].items()}}
    usd = post("/api/forecasting/run", body)
    inr = post("/api/forecasting/run", {**body, "money": {"currency": "INR", "unit": "millions"}})
    assert without_money(inr) == without_money(usd)
    assert usd["money"] == {"currency": "USD", "unit": "millions"}
    assert inr["money"] == {"currency": "INR", "unit": "millions"}


def backtest_body(**extra) -> dict:
    example = client.get("/api/backtesting/deals").json()[0]
    return {"entry": example["entry"], "actual": example["actual"],
            "actual_exit": example["actual_exit"], "n": 1000, **extra}


def test_a_backtest_gives_identical_results_in_any_currency():
    usd = post("/api/backtesting/run", backtest_body())
    chf = post("/api/backtesting/run", backtest_body(money={"currency": "CHF", "unit": "millions"}))
    assert without_money(chf) == without_money(usd)
    assert chf["money"]["currency"] == "CHF"


# ---------------------------------------------------------------------------
# Units: thousands and billions answer as millions do, scaled
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit", ["thousands", "billions"])
def test_a_deal_in_another_unit_gives_the_same_returns_and_scaled_money(unit):
    millions = post("/api/deal/run", {"inputs": deal("EUR")})
    other = post("/api/deal/run", {"inputs": scaled(deal("EUR"), unit)})
    assert other["money"] == {"currency": "EUR", "unit": unit}
    assert other["returns"]["irr"] == pytest.approx(millions["returns"]["irr"], rel=1e-9)
    assert other["returns"]["moic"] == pytest.approx(millions["returns"]["moic"], rel=1e-9)
    for key in ("entry_equity", "net_exit_equity", "exit_ev", "net_debt_at_exit"):
        assert other["returns"][key] == pytest.approx(millions["returns"][key] * SCALE[unit], rel=1e-9), key
    assert other["operating_model"]["ebitda"] == pytest.approx(
        [v * SCALE[unit] for v in millions["operating_model"]["ebitda"]], rel=1e-9)


@pytest.mark.parametrize("unit", ["thousands", "billions"])
def test_small_fees_show_in_the_bridge_whatever_the_unit(unit):
    """A 1M EBITDA deal: its fees (a few hundred thousand) are a bridge step in
    every unit, not hidden because the number is small in billions."""
    small = deal("EUR", ebitda=1.0, mincash=0.0)
    millions = post("/api/deal/run", {"inputs": small})
    other = post("/api/deal/run", {"inputs": scaled(small, unit)})
    assert "Fees" in [s["key"] for s in millions["bridge_steps"]]
    assert [s["key"] for s in other["bridge_steps"]] == [s["key"] for s in millions["bridge_steps"]]


def test_sources_and_uses_carry_the_money_and_balance_in_any_unit():
    body = {"ebitda": 100_000.0, "entry_mult": 10.0, "senior_x": 3.4, "mezz_x": 0.8,
            "money": {"currency": "EUR", "unit": "thousands"}}
    su = post("/api/deal/sources-and-uses", body)
    assert su["balanced"] and su["money"] == {"currency": "EUR", "unit": "thousands"}


def test_a_forecast_balance_check_uses_the_companys_unit():
    """0.3M off balance passes the 0.5M check in millions, and 300k in thousands must too."""
    history = default_history(3)
    history["h_cash"] = [v + 0.3 for v in history["h_cash"]]
    seeded = post("/api/forecasting/seed", {"history": history})
    assumptions = {k: [v] * 5 for k, v in seeded["seeded_assumptions"].items()}
    millions = post("/api/forecasting/run", {"history": history, "assumptions": assumptions})
    thousands_history = {k: [v * 1000 for v in vals] for k, vals in history.items()}
    thousands = post("/api/forecasting/run", {
        "history": thousands_history, "assumptions": assumptions,
        "money": {"currency": "EUR", "unit": "thousands"}})
    assert millions["opening_balance_gap"] == pytest.approx(0.3)
    assert thousands["opening_balance_gap"] == pytest.approx(300.0)
    assert millions["balanced"] and thousands["balanced"]


def test_fixed_amounts_convert_between_units():
    assert in_unit(0.001, "thousands") == 1.0
    assert in_unit(0.001, "millions") == 0.001
    assert in_unit(5.0, "billions") == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Every response says what its money is counted in
# ---------------------------------------------------------------------------
def test_defaults_and_examples_say_they_are_us_dollar_millions():
    usd_m = {"currency": "USD", "unit": "millions"}
    assert client.get("/api/forecasting/defaults").json()["money"] == usd_m
    assert all(d["money"] == usd_m for d in client.get("/api/backtesting/deals").json())
    assert post("/api/deal/run", {"inputs": {}})["money"] == usd_m


def test_a_saved_deal_keeps_its_currency_and_unit():
    """The deal's currency lives in its inputs, so saving and reopening keeps it
    (tests/test_deals.py stores a EUR thousands deal through the database)."""
    from db.deals import clean_inputs
    cleaned = clean_inputs({"currency": "SEK", "unit": "billions"})
    assert cleaned["currency"] == "SEK" and cleaned["unit"] == "billions"
    assert clean_inputs({})["currency"] == "USD" and clean_inputs({})["unit"] == "millions"


@pytest.mark.parametrize("money", [
    {"currency": "eur"}, {"currency": "EURO"}, {"currency": "E1R"}, {"currency": "€"},
    {"unit": "hundreds"}, {"unit": "M"},
])
def test_impossible_currencies_and_units_are_refused(money):
    resp = client.post("/api/deal/run", json={"inputs": {**deal(), **money}})
    assert resp.status_code == 422, resp.text
    with pytest.raises(ValueError):
        Money(**{"currency": "USD", "unit": "millions", **money})


def test_a_workbook_says_what_its_money_is_counted_in():
    resp = client.post("/api/export/workbook", json={
        "filename": "pl.xlsx", "money": {"currency": "EUR", "unit": "thousands"},
        "sheets": [{"name": "P&L", "columns": ["", "Year 1"], "rows": [["Revenue", 1250.0]]}]})
    assert resp.status_code == 200, resp.text
    about = pd.read_excel(io.BytesIO(resp.content), sheet_name="About")
    assert dict(zip(about["Item"], about["Value"])) == {"Currency": "EUR", "Money unit": "thousands"}


def test_the_simulation_sample_says_what_its_money_is_counted_in():
    resp = client.post("/api/export/montecarlo-sample",
                       json={"mc": MC, "seed": 1, "deal": deal("GBP", "thousands", ebitda=100_000.0)})
    assert resp.status_code == 200, resp.text
    about = pd.read_excel(io.BytesIO(resp.content), sheet_name="About")
    values = dict(zip(about["Item"], about["Value"]))
    assert values["Currency"] == "GBP" and values["Money unit"] == "thousands"


# ---------------------------------------------------------------------------
# The money keys are right: converting to millions and back matches the
# engine run directly in thousands (up to the engine's rounding), leaf by leaf.
# A key missing from the list, or listed wrongly, is off by a factor of 1,000.
# ---------------------------------------------------------------------------
def leaves(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value


def assert_same_leaves(converted: dict, raw: dict, rel=5e-3):
    got, want = dict(leaves(converted)), dict(leaves(raw))
    assert got.keys() == want.keys()
    wrong = {p: (got[p], want[p]) for p in got
             if got[p] != pytest.approx(want[p], rel=rel, abs=0.5)}
    assert not wrong, wrong


def raw_engine(monkeypatch, module):
    """Run an endpoint without its conversion: the engine sees thousands as they are."""
    monkeypatch.setattr(module, "rescale", lambda value, factor, keys: value)


def test_a_deal_answer_converts_every_money_figure_and_nothing_else(monkeypatch):
    import api.routers.deal as deal_router
    body = {"inputs": deal("EUR", "thousands", ebitda=100_000.0, mincash=10_000.0),
            "settings": {"other_uses": 2_000.0}}
    converted = without_money(post("/api/deal/run", body))
    raw_engine(monkeypatch, deal_router)
    monkeypatch.setattr(deal_router, "in_millions", lambda d, cfg: (d, cfg))
    assert_same_leaves(converted, without_money(post("/api/deal/run", body)))


def test_sources_and_uses_convert_every_money_figure_and_nothing_else(monkeypatch):
    import api.routers.deal as deal_router
    body = {"ebitda": 100_000.0, "entry_mult": 10.0, "senior_x": 3.4, "mezz_x": 0.8, "mincash": 5_000.0,
            "settings": {"other_uses": 2_000.0}, "money": {"currency": "EUR", "unit": "thousands"}}
    converted = without_money(post("/api/deal/sources-and-uses", body))
    raw_engine(monkeypatch, deal_router)
    monkeypatch.setattr(deal_router, "to_millions", lambda v, unit: v)
    assert_same_leaves(converted, without_money(post("/api/deal/sources-and-uses", body)))


def test_a_backtest_answer_converts_every_money_figure_and_nothing_else(monkeypatch):
    import api.routers.backtesting as bt_router
    example = client.get("/api/backtesting/deals").json()[0]
    k = 1000.0
    body = {"entry": {**example["entry"], "entry_ebitda": example["entry"]["entry_ebitda"] * k},
            "actual": {key: [v * k for v in vals] for key, vals in example["actual"].items()},
            "actual_exit": {**example["actual_exit"],
                            **{key: example["actual_exit"][key] * k
                               for key in ("exit_ev", "net_debt_at_exit", "sponsor_equity_entry")}},
            "n": 1000, "money": {"currency": "USD", "unit": "thousands"}}
    converted = without_money(post("/api/backtesting/run", body))
    raw_engine(monkeypatch, bt_router)
    monkeypatch.setattr(bt_router, "backtest_in_millions", lambda e, a, x, cfg, unit: (e, a, x, cfg))
    raw = without_money(post("/api/backtesting/run", body))
    # Predicted EBITDA is rounded to 0.1 of the unit the engine runs in
    # (100 thousand here), hence the looser tolerance; still far from 1,000x
    assert_same_leaves(converted, raw, rel=1e-2)


def test_a_simulation_answer_converts_its_money_parameters(monkeypatch):
    import api.routers.montecarlo as mc_router
    body = {"mc": {**MC, "ebitda": 100_000.0}, "seed": 3, "settings": {"other_uses": 2_000.0},
            "deal": deal("EUR", "thousands", ebitda=100_000.0)}
    converted = post("/api/montecarlo/run", body)
    assert converted["params"]["entry_ebitda"] == pytest.approx(100_000.0)
    assert converted["params"]["other_uses"] == pytest.approx(2_000.0)
    millions = post("/api/montecarlo/run", {**body, "mc": {**MC, "ebitda": 100.0},
                                            "settings": {"other_uses": 2.0},
                                            "deal": deal("EUR", "millions")})
    assert converted["summary"] == millions["summary"]
    assert converted["heatmap"] == millions["heatmap"]


@pytest.mark.parametrize("unit", ["thousands", "billions"])
def test_a_company_in_another_unit_seeds_the_same_assumptions(unit):
    """Seeded money assumptions (dividends, buybacks, minimum cash) are rounded
    to the same precision whatever the unit; the rest are ratios and days."""
    money_seeds = {"other_inc", "divs", "buybacks", "ltd_chg", "min_cash"}
    history = default_history(3)
    millions = post("/api/forecasting/seed", {"history": history})["seeded_assumptions"]
    other = post("/api/forecasting/seed", {
        "history": {k: [v * SCALE[unit] for v in vals] for k, vals in history.items()},
        "money": {"currency": "EUR", "unit": unit}})["seeded_assumptions"]
    for key, value in millions.items():
        expected = value * SCALE[unit] if key in money_seeds else value
        assert other[key] == pytest.approx(expected, rel=1e-9, abs=1e-12), key
