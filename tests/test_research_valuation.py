"""Valuation (PLAN.md R6): a discounted cash flow and a target multiple in
the model definition, so the server and the workbook give the same value per
share; both methods match the cases solved by hand to 0.01; and nothing the
screen or the workbook can show words a recommendation.

The hand cases (`tests/fixtures/research/valuation.json`) read only the
hand-built workbooks' own figures (`model1_<slug>.json` "expected"), never
the model's. **Never change them to agree with the model.**
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from economy.catalogue import series_key
from economy.model import Series
from research import compose, valuation
from research.definition import Company, Heading, label, words_for
from research.evaluate import EvaluationError, evaluate
from research.models import CATALOGUE, occupancy
from research.rollforward import roll
from research.sheetback import RESTRUCTURED, SheetRefused, read_back
from research.workbook import layout, sheets, write
from tests import research_support as support
from tests.test_research_actuals import H1, BALANCE, period, results
from tests.test_research_rollforward import AS_OF, half_year, year_end, yatharth
from tests.test_uploads import uploads  # noqa: F401 -- the fixture

MODEL = occupancy.MODEL
SLUGS = ("hospitals", "hotels")
HAND = json.loads((support.FIXTURES / "valuation.json").read_text(encoding="utf-8"))
ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)

# Money to 0.01 of the statements' unit, a value per share to 0.01 of the currency, a factor or a share to 0.01 of
# a percentage point
TOLERANCE = {"money": 0.01, "rupees2": 0.01, "factor": 1e-6, "share": 1e-4}


def company(slug, **changes):
    return support.company(support.load(slug), **changes)


def unit_of(row):
    return next(r.unit for r in MODEL.data_rows() if r.key == row)


# ---------------------------------------------------------------------------- the cases solved by hand
def hand_working(slug):
    """The hand case's working, line for line as `valuation.json` writes it, on the hand workbook's figures."""
    case, e = HAND["cases"][slug], support.load(slug)["expected"]
    a = {k: v["ly"] for k, v in case["assumptions"].items()}
    rate, g, mult, price = a["v_rate"], a["v_g"], a["v_mult"], a["v_price"]
    fcf = {p: e["ebitda"][p] - e["tax_rate"][p] * max(e["ebitda"][p] - e["dep"][p], 0) + e["cf_wc"][p] - e["capex"][p]
           for p in ("cy", "y1", "y2")}
    df = {"cy": 1 / (1 + rate)}
    df["y1"] = df["cy"] / (1 + rate)
    df["y2"] = df["y1"] / (1 + rate)
    tv = fcf["y2"] * (1 + g) / (rate - g)
    ev = fcf["cy"] * df["cy"] + fcf["y1"] * df["y1"] + fcf["y2"] * df["y2"] + tv * df["y2"]
    eq = ev - e["netdebt"]["ly"] - e["nci"]["ly"]
    ps = eq / e["shares"]["ly"]
    year = case["choices"]["value_year"]
    if case["choices"]["value_multiple"] == "ev_ebitda":
        mps = (mult * e["ebitda"][year] - e["netdebt"]["ly"] - e["nci"]["ly"]) / e["shares"]["ly"]
    else:
        mps = mult * e["eps"][year]
    return {"v_fcf": fcf, "v_df": df, "v_tv": {"y2": tv}, "v_ev": {"ly": ev}, "v_eq": {"ly": eq}, "v_ps": {"ly": ps},
            "v_mps": {"ly": mps}, "v_gap_dcf": {"ly": ps / price - 1}, "v_gap_mult": {"ly": mps / price - 1}}


@pytest.mark.parametrize("slug", SLUGS)
def test_the_hand_cases_figures_are_their_own_working(slug):
    """Guards the fixture: its figures are the working written in it, on the hand workbook's numbers."""
    worked, written = hand_working(slug), HAND["cases"][slug]["expected"]
    assert set(worked) == set(written)
    for row, by in written.items():
        for p, v in by.items():
            assert worked[row][p] == pytest.approx(v, abs=1e-6), (row, p)


@pytest.mark.parametrize("slug", SLUGS)
def test_both_methods_match_the_case_solved_by_hand(slug):
    values = evaluate(MODEL, company(slug)).values
    wrong = [(row, p, hand, values[row][p]) for row, by in HAND["cases"][slug]["expected"].items()
             for p, hand in by.items() if abs(values[row][p] - hand) > TOLERANCE[unit_of(row)]]
    assert wrong == []
    case = HAND["cases"][slug]
    assert {case["choices"]["value_multiple"] for case in HAND["cases"].values()} == set(valuation.MULTIPLES)
    # the date typed with the price is the day it names
    assert date(1899, 12, 30).toordinal() + values["v_date"]["ly"] == date.fromisoformat(case["price_date"]).toordinal()


@pytest.mark.parametrize("slug", SLUGS)
def test_the_workbook_gives_the_servers_valuation(slug):
    c = company(slug)
    values, where = evaluate(MODEL, c).values, layout(MODEL)
    get = support.workbook_values(write(MODEL, c))
    for row, by in HAND["cases"][slug]["expected"].items():
        for p in by:
            assert get(where.cell(row, p)) == pytest.approx(values[row][p], abs=TOLERANCE[unit_of(row)]), (row, p)


if support.soffice_wanted():
    @pytest.mark.parametrize("slug", SLUGS)
    def test_libreoffice_gives_the_hand_solved_valuation(slug):
        recalculated = support.recalculated_by_libreoffice(write(MODEL, company(slug)))["Model"]
        where = layout(MODEL)
        for row, by in HAND["cases"][slug]["expected"].items():
            for p, hand in by.items():
                assert recalculated[where.cell(row, p)].value == pytest.approx(hand, abs=TOLERANCE[unit_of(row)])


# ---------------------------------------------------------------------------- what moves the value
def test_a_higher_discount_rate_lowers_the_value_and_the_year_chosen_is_the_years_value():
    base = evaluate(MODEL, company("hospitals")).values
    dearer = evaluate(MODEL, company("hospitals", v_rate={"ly": 0.15})).values
    assert dearer["v_ps"]["ly"] < base["v_ps"]["ly"] and dearer["v_ev"]["ly"] < base["v_ev"]["ly"]
    assert dearer["v_mps"]["ly"] == base["v_mps"]["ly"]              # the multiple does not discount
    for year in ("cy", "y1", "y2"):
        for multiple in valuation.MULTIPLES:
            c = replace(company("hospitals"), choices={**company("hospitals").choices, "value_year": year,
                                                       "value_multiple": multiple})
            v = evaluate(MODEL, c).values
            assert v["v_mps"]["ly"] == v["v_mv"][year]
            e = support.load("hospitals")["expected"]
            hand = (25.0 * e["ebitda"][year] - e["netdebt"]["ly"] - e["nci"]["ly"]) / e["shares"]["ly"] \
                if multiple == "ev_ebitda" else 25.0 * e["eps"][year]
            assert v["v_mps"]["ly"] == pytest.approx(hand, abs=0.01)
    higher = evaluate(MODEL, company("hospitals", v_mult={"ly": 30.0})).values
    assert higher["v_mps"]["ly"] > base["v_mps"]["ly"] and higher["v_ps"]["ly"] == base["v_ps"]["ly"]
    cheaper = evaluate(MODEL, company("hospitals", v_price={"ly": 325.0})).values
    assert cheaper["v_gap_dcf"]["ly"] == pytest.approx((1 + base["v_gap_dcf"]["ly"]) * 2 - 1)


def test_growth_at_the_discount_rate_gives_no_value_rather_than_a_number():
    c = company("hotels", v_g={"ly": 0.12})
    with pytest.raises(EvaluationError):
        evaluate(MODEL, c)
    partial = evaluate(MODEL, c, partial=True)
    assert partial.values["v_tv"]["y2"] is None and partial.values["v_ps"]["ly"] is None
    assert ("v_tv", "y2") in partial.undefined
    assert partial.values["v_mps"]["ly"] is not None                # the multiple still works out


def test_with_no_valuation_typed_the_past_and_the_forecast_still_work_out():
    c = company("hotels")
    bare = replace(c, values={k: v for k, v in c.values.items() if not k.startswith("v_")})
    whole, part = evaluate(MODEL, c).values, evaluate(MODEL, bare, partial=True)
    assert part.values["pat_own"]["y2"] == whole["pat_own"]["y2"]
    assert part.values["v_fcf"]["y2"] == whole["v_fcf"]["y2"]       # the cash flow needs no valuation input
    for row in ("v_df", "v_pv", "v_ev", "v_ps", "v_mps", "v_gap_dcf"):
        assert all(v is None for v in part.values[row].values()), row
    assert {r for r, _ in part.missing} == {"v_rate", "v_g", "v_mult", "v_price", "v_date"}


def test_every_forecast_year_is_valued_and_a_year_the_model_no_longer_has_goes_back_to_the_first():
    longer = occupancy.build(forecast_years=4)
    rows = {r.key: r for r in longer.data_rows()}
    assert set(rows["v_tv"].rules) == {"y4"} and "v_pv[y4]" in rows["v_ev"].rules["ly"]
    assert longer.choices()["value_year"] == {"cy", "y1", "y2", "y3", "y4"}
    c = replace(company("hotels"), choices={**company("hotels").choices, "value_year": "y4"})
    fitted, _ = compose.fit(MODEL, c)
    assert fitted.choices["value_year"] == "y1"
    wrong = replace(c, choices={**c.choices, "value_multiple": "price_to_book"})
    assert compose.fit(MODEL, wrong)[0].choices["value_multiple"] == "price_to_book"   # still refused by cells
    assert [r.key for r in longer.data_rows()] == [r.key for r in MODEL.data_rows()]


# ---------------------------------------------------------------------------- kept through a roll and a sheet
def test_a_roll_keeps_the_valuation_and_a_new_year_marks_the_price():
    model, c = yatharth()
    c = replace(c, values={**c.values, "v_price": {"ly": 650.0}, "v_date": {"ly": 46311.0}})
    quarter = roll(model, c, AS_OF, half_year(model, c)).company
    assert quarter.values["v_rate"] == c.values["v_rate"] and quarter.values["v_price"] == {"ly": 650.0}
    assert "v_price" not in quarter.marks
    year = roll(model, c, AS_OF, year_end())
    v = year.company.values
    assert v["v_rate"] == c.values["v_rate"] and v["v_mult"] == c.values["v_mult"] and v["v_price"] == {"ly": 650.0}
    assert year.company.marks["v_price"]["ly"]["level"] == "amber" and ("v_date", "ly") in year.carried
    assert "valuation is now as at the end of" in year.company.marks["v_price"]["ly"]["why"]


def test_a_price_and_its_date_typed_in_excel_come_back():
    c = company("hotels")
    wb = write(MODEL, c, saved="abc")
    names = {n: wb.defined_names[n].attr_text.split("!")[1] for n in wb.defined_names}
    ws = wb["Model"]
    ws[names["v_price_ly"].replace("$", "")].value = 151.5
    ws[names["v_date_ly"].replace("$", "")].value = 46320           # a day number, as Excel holds a date
    ws[names["v_rate_ly"].replace("$", "")].value = 0.115
    grid = [[support.as_excel_holds_it(cell.value) if cell.value is not None else "" for cell in row]
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column)]
    back = read_back(MODEL, c, "abc", grid, names)
    assert back.company.values["v_price"]["ly"] == 151.5 and back.company.values["v_date"]["ly"] == 46320
    assert back.company.values["v_rate"]["ly"] == 0.115
    assert {(r, p) for r, p, _, _ in back.changed} == {("v_price", "ly"), ("v_date", "ly"), ("v_rate", "ly")}


def test_a_sheet_loaded_before_the_valuation_rows_is_refused_saying_to_load_it_again():
    c = company("hotels")
    wb = write(MODEL, c, saved="abc")
    names = {n: wb.defined_names[n].attr_text.split("!")[1] for n in wb.defined_names}
    ws = wb["Model"]
    first = layout(MODEL).rows["v_rate"] - 1                          # the VALUATION heading's row
    grid = [[support.as_excel_holds_it(cell.value) if cell.value is not None else "" for cell in row]
            for row in ws.iter_rows(min_row=1, max_row=first - 1, max_col=ws.max_column)]
    with pytest.raises(SheetRefused) as refused:
        read_back(MODEL, c, "abc", grid, names)
    assert refused.value.code == RESTRUCTURED and "gained rows since this sheet was loaded" in str(refused.value)


# ---------------------------------------------------------------------------- no recommendation, anywhere
CALLS = re.compile(r"\b(buy|sell|outperform|underperform|overweight|underweight|recommend\w*|rating|"
                   r"target price|price target|upside|downside|undervalued|overvalued|cheap|expensive|attractive|"
                   r"bullish|bearish)\b", re.IGNORECASE)
# A rating is written capitalised ("Hold", "HOLD"); "the columns hold" is a verb, and "Add a line" and "Reduce" are
# the screens' own verbs, so only their upper-case spelling would be a call
RATINGS = re.compile(r"\b(Hold|HOLD|Accumulate|ACCUMULATE|REDUCE|Neutral|NEUTRAL)\b")
# Sentences that say the tool picks none (lower.NOTES)
SAYS_IT_PICKS_NONE = ("picks no rating",)
RESEARCH_NAMESPACES = ("models", "coverage", "reader", "excel")


def calls_in(text: str) -> list:
    for phrase in SAYS_IT_PICKS_NONE:
        text = text.replace(phrase, "")
    return [m.group(0) for m in CALLS.finditer(text)] + [m.group(0) for m in RATINGS.finditer(text)]


def strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)
    elif isinstance(o, str):
        yield o


def test_the_research_screens_words_hold_no_recommendation():
    catalogue = json.loads((ROOT / "web" / "messages" / "en.json").read_text(encoding="utf-8"))
    found = [(ns, text) for ns in RESEARCH_NAMESPACES for text in strings(catalogue[ns]) if calls_in(text)]
    assert found == []


def test_no_model_and_no_workbook_words_a_recommendation():
    texts = []
    for spec in CATALOGUE.specs.values():
        model = CATALOGUE.build(spec.default)
        texts += [spec.about, *model.notes, *model.constants.values()]
        for sector, words in model.sectors.items():
            texts += list(words.values())
            texts += [label(r.label, words) for r in model.rows]
    for slug in SLUGS:
        c = company(slug)
        for sheet in sheets(MODEL, c, saved="abc"):
            texts += [cell["text"] for cell in sheet["cells"] if cell.get("text")]
        texts += [label(r.label, words_for(MODEL, c)) for r in MODEL.rows]
    assert len(texts) > 500
    assert [t for t in texts if calls_in(t)] == []


def test_the_check_finds_a_recommendation_when_there_is_one():
    assert calls_in("Value per share against the price: BUY") == ["BUY"]
    assert calls_in("Target price, Rs") == ["Target price"]
    assert calls_in("Rating: HOLD") == ["Rating", "HOLD"] and calls_in("the columns hold the assumptions") == []
    assert calls_in("The model states no view and picks no rating.") == []


# ---------------------------------------------------------------------------- the risk-free rate offered
def yields(area="IN", value=7.06, month="2026-09"):
    s = Series("bond_yield_10y", area, "oecd", "DF_FINMARK/IND.M.IRLT", "M", "https://data-explorer.oecd.org/",
               (("2026-08", 6.84), (month, value)))
    return {series_key("bond_yield_10y", area, "oecd"): s}


def test_the_risk_free_rate_is_the_reporting_countrys_10_year_yield():
    answer = valuation.risk_free(["INR", "INR"], yields(), date(2026, 10, 19))
    assert answer["status"] == "ok" and answer["country"] == "IN" and answer["value"] == pytest.approx(0.0706)
    assert answer["period"] == "2026-09" and answer["current"] is True and answer["source"] == "oecd"
    assert valuation.risk_free([], yields(), date(2026, 10, 19))["status"] == valuation.NO_CURRENCY
    assert valuation.risk_free(["INR", "USD"], yields(), date(2026, 10, 19))["status"] == valuation.NO_CURRENCY
    assert valuation.risk_free(["EUR"], yields(), date(2026, 10, 19))["status"] == valuation.NO_COUNTRY
    assert valuation.risk_free(["XYZ"], yields(), date(2026, 10, 19))["status"] == valuation.NO_COUNTRY
    assert valuation.risk_free(["USD"], yields(), date(2026, 10, 19))["status"] == valuation.NO_DATA


def test_the_risk_free_rate_reads_the_callers_company_only(uploads, monkeypatch):  # noqa: F811
    from db import coverage as rows, economy
    monkeypatch.setattr(economy, "all_series", lambda: (yields(), None))
    made = client.post("/api/research/companies", json={"name": "Hotel Co"})
    assert made.status_code == 201
    company_id = made.json()["id"]
    assert client.get(f"/api/research/companies/{company_id}/risk-free").json()["status"] == "no_currency"
    stored = period([results(H1, [1000, 900, 950, 1900, 1800, 3800]), BALANCE])
    rows.save_periods("test:alice", uuid.UUID(company_id), [{"period_end": "2026-09-30", "months": 6, **stored}], None)
    answer = client.get(f"/api/research/companies/{company_id}/risk-free")
    assert answer.status_code == 200, answer.text
    assert answer.json()["value"] == pytest.approx(0.0706) and answer.json()["country"] == "IN"
    uploads("test:bob")
    assert client.get(f"/api/research/companies/{company_id}/risk-free").status_code == 404


def test_a_run_answers_the_valuation_rows_and_their_choices():
    ex = client.get("/api/research/models").json()["examples"][1]
    body = client.post("/api/research/models/run", json={"structure": ex["structure"], "company": ex["company"]}).json()
    rows = {r["key"]: r for r in body["rows"] if r["key"]}
    hand = HAND["cases"]["hotels"]["expected"]
    assert rows["v_ps"]["cells"]["ly"]["value"] == pytest.approx(hand["v_ps"]["ly"], abs=0.01)
    assert rows["v_mps"]["cells"]["ly"]["choice"] == "value_year"
    assert rows["v_date"]["unit"] == "date" and rows["v_price"]["cells"]["ly"]["kind"] == "assumption"
    assert body["choices"]["value_multiple"] == ["ev_ebitda", "pe"] and body["choices_made"]["value_year"] == "y2"
