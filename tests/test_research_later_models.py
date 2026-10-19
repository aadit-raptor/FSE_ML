"""Models 2, Per tonne, and 3, Stores (PLAN.md R7): the companies solved by
hand in Excel (`research-samples/r7/` on the Mac: JK Lakshmi Cement and Star
Cement for model 2, Avenue Supermarts and V-Mart Retail for model 3) come out
of the model definition to the paisa.

`tests/fixtures/research/model2_*.json` and `model3_*.json` hold the figures
typed from the public filings and what the hand-built workbooks' own formulas
gave, worked out by LibreOffice. **Never change a fixture to agree with the
model**: a real difference is a finding (as PLAN.md 3.4).
"""
import pytest

from research import formula
from research.definition import cells
from research.evaluate import evaluate
from research.models import CATALOGUE, per_tonne, stores
from tests import research_support as support

PERIODS = ("ly", "ytd", "roy", "cy", "y1", "y2")
CEMENT = ("model2_jk_lakshmi", "model2_star_cement")
RETAIL = ("model3_avenue_supermarts", "model3_v_mart")
ALL = CEMENT + RETAIL
MODELS = {"per_tonne": per_tonne.MODEL, "stores": stores.MODEL}


def model_of(name):
    return MODELS[support.load(name)["model"]]


def solved(name, **changes):
    return evaluate(model_of(name), support.company(support.load(name), **changes)).values


@pytest.mark.parametrize("name", ALL)
def test_the_filings_own_sums_hold_on_the_typed_figures(name):
    """Guards the fixture: every sum the page prints, in every column, within its tolerance."""
    fixture, checked = support.load(name), 0
    figures = support.reported(fixture)
    for section in fixture["sections"]:
        for column in section["columns"]:
            names = {key: by[column] for key, by in figures.items() if column in by}
            for sum_ in section["checks"]:
                assert abs(formula.value(sum_["formula"], names)) <= sum_["tolerance"], (sum_["label"], column)
                checked += 1
    assert checked >= 30


@pytest.mark.parametrize("name", ALL)
def test_the_model_gives_the_hand_workbooks_numbers_on_every_row(name):
    expected, values, model = support.load(name)["expected"], solved(name), model_of(name)
    rows = {r.key for r in model.data_rows()} - {"r_pbt", "r_pat", "r_own"}       # the filing's own totals
    assert set(expected) == rows                                                    # every row of the hand-built sheet
    wrong = []
    for row, by_period in expected.items():
        for period in PERIODS:
            hand, ours = by_period[period], values[row][period]
            if (hand is None) != (ours is None) or (hand is not None and abs(hand - ours) > 0.01):
                wrong.append((row, period, hand, ours))
    assert wrong == []
    compared = sum(v is not None for by_period in expected.values() for v in by_period.values())
    assert compared >= {"per_tonne": 470, "stores": 430}[support.load(name)["model"]]


@pytest.mark.parametrize("name", ALL)
def test_the_past_ties_to_the_filing_and_the_balance_sheet_balances(name):
    v = solved(name)
    for check_row in ("chk_pbt", "chk_pat", "chk_own"):
        for period in ("ly", "ytd"):
            assert abs(v[check_row][period]) < 0.005, (check_row, period)
    for period in ("cy", "y1", "y2"):
        assert abs(v["chk_bs"][period]) < 1e-6
        assert v["cash"][period] == pytest.approx(v["cf_close"][period])


def test_headline_figures_of_the_hand_build():
    """A few numbers a person can find on the pages or work out with a calculator, typed here."""
    jk, star, dm, vm = (solved(n) for n in ALL)
    assert jk["rev"]["ly"] == 6192.62 and jk["pbt"]["ly"] == pytest.approx(402.90)
    assert jk["s1_price"]["ly"] == pytest.approx(6192.62e7 / 12_129_000)               # Rs 5,106 a tonne
    assert jk["s1_util"]["ytd"] == pytest.approx(6_169_000 * 365 / (17_250_000 * 183))
    assert jk["frt_m2"]["ytd"] == pytest.approx(748.43e7 / 6_169_000)                    # freight, Rs 1,213 a tonne
    assert star["s1_vol"]["ytd"] == pytest.approx(1_354_000)                             # cement and clinker
    assert star["frt"]["ly"] == pytest.approx(68885.28 + 34652.84)                       # both carriage lines
    assert star["pat_own"]["ly"] == pytest.approx(39342.42)
    assert star["ebitda_pu"]["ly"] == pytest.approx(star["ebitda"]["ly"] * 1e5 / 5_496_000)
    assert dm["s1_avg"]["ly"] == pytest.approx(438.875)                                 # from the quarter ends
    assert dm["s1_price"]["ly"] == pytest.approx(66968.03e7 / (438.875 * 365))           # Rs 41.8 lakh a store a day
    assert dm["s1_end"]["roy"] == 518 + 40
    assert vm["eps"]["ly"] == pytest.approx(5.78, abs=0.005)                             # the page's own, after the bonus
    assert vm["oth"]["ly"] == pytest.approx(38365 - 2418)                                # the exceptional gain taken off


@pytest.mark.parametrize("pair", [CEMENT, RETAIL])
def test_one_model_serves_both_companies_and_only_the_typed_past_differs(pair):
    model = model_of(pair[0])
    a, b = (cells(model, support.company(support.load(n))) for n in pair)
    assert set(a) == set(b)
    future = [k for k in a if k[1] in ("roy", "cy", "y1", "y2")]
    assert len(future) > 250
    for key in future:                                               # every forecast cell is the same rule,
        if key[0] != "v_mv":                                         # but for the multiple each values on
            assert a[key].tree == b[key].tree and a[key].typed == b[key].typed, key


def test_costs_follow_tonnes_sold_and_stores_open_per_day():
    base = solved("model2_star_cement")
    busier = solved("model2_star_cement", s1_util={"y1": 0.682})                      # 10% more tonnes in y1
    assert busier["s1_vol"]["y1"] == pytest.approx(base["s1_vol"]["y1"] * 1.1)
    for cost in ("mat", "emp", "pwr", "frt", "oth"):                                  # each per tonne, so 10% more
        assert busier[cost]["y1"] == pytest.approx(base[cost]["y1"] * 1.1)
    assert busier["ebitda_pu"]["y1"] == pytest.approx(base["ebitda_pu"]["y1"])         # per tonne, unmoved
    shop = solved("model3_v_mart")
    more = solved("model3_v_mart", s1_add={"y1": 120})                                # 60 more stores opened in y1
    added = more["s1_avg"]["y1"] - shop["s1_avg"]["y1"]
    assert added == pytest.approx(30)
    assert more["emp"]["y1"] - shop["emp"]["y1"] == pytest.approx(23_500 * added * 365 / 1e5)
    assert more["s1_rev"]["y1"] / shop["s1_rev"]["y1"] == pytest.approx(more["s1_avg"]["y1"] / shop["s1_avg"]["y1"])


def test_the_models_are_offered_and_marked_not_vetted():
    for model_id in ("per_tonne", "stores"):
        spec = CATALOGUE.spec(model_id)
        assert spec.offered is True and spec.vetted is False
        assert MODELS[model_id].vetted is False
    assert {r.key for r in per_tonne.MODEL.data_rows()} >= {"frt", "frt_pu", "ebitda_pu", "s1_util", "s1_vol"}
    assert "pwr" not in {r.key for r in stores.MODEL.data_rows()}
