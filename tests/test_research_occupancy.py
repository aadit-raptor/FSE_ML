"""Model 1, Occupancy (PLAN.md R4): the two companies solved by hand in Excel
(Yatharth Hospital and Lemon Tree Hotels, `research-samples/r4a/` on the
Mac) come out of the model definition to the paisa.

`tests/fixtures/research/` holds the figures typed from the public filings and
what the hand-built workbooks' own formulas gave. **Never change a fixture to
agree with the model**: a real difference is a finding (as PLAN.md 3.4).
"""
import json

import pytest

from research import formula
from research.definition import ASSUME, FIGURE, Model, Pick, cells, check
from research.evaluate import evaluate
from research.models import occupancy
from tests import research_support as support

SLUGS = ("hospitals", "hotels")
PERIODS = ("ly", "ytd", "roy", "cy", "y1", "y2")
MODEL = occupancy.MODEL


def solved(slug, **changes):
    return evaluate(MODEL, support.company(support.load(slug), **changes)).values


@pytest.mark.parametrize("slug", SLUGS)
def test_the_filings_own_sums_hold_on_the_typed_figures(slug):
    """Guards the fixture: every sum the page prints, in every column, within its tolerance."""
    fixture, checked = support.load(slug), 0
    figures = support.reported(fixture)
    for section in fixture["sections"]:
        for column in section["columns"]:
            names = {key: by_column[column] for key, by_column in figures.items() if column in by_column}
            for sum_ in section["checks"]:
                assert abs(formula.value(sum_["formula"], names)) <= sum_["tolerance"], (sum_["label"], column)
                checked += 1
    assert checked >= {"hospitals": 30, "hotels": 82}[slug]


@pytest.mark.parametrize("slug", SLUGS)
def test_the_model_gives_the_hand_workbooks_numbers_on_every_row(slug):
    expected, values = support.load(slug)["expected"], solved(slug)
    assert len(expected) == 103                                     # every row of the hand-built sheet
    wrong = []
    for row, by_period in expected.items():
        for period in PERIODS:
            hand, ours = by_period[period], values[row][period]
            if (hand is None) != (ours is None) or (hand is not None and abs(hand - ours) > 0.01):
                wrong.append((row, period, hand, ours))
    assert wrong == []
    compared = sum(v is not None for by_period in expected.values() for v in by_period.values())
    assert compared >= {"hospitals": 420, "hotels": 440}[slug]


@pytest.mark.parametrize("slug", SLUGS)
def test_the_past_ties_to_the_filing_and_the_balance_sheet_balances(slug):
    v = solved(slug)
    for check_row in ("chk_pbt", "chk_pat", "chk_own"):
        for period in ("ly", "ytd"):
            assert abs(v[check_row][period]) < 0.005, (check_row, period)
    for period in ("cy", "y1", "y2"):
        assert abs(v["chk_bs"][period]) < 1e-6
        assert v["ta"][period] > 0
        assert v["cash"][period] == pytest.approx(v["cf_close"][period])


def test_headline_figures_of_the_hand_build():
    """A few numbers a person can find in the hand-built sheets, typed here."""
    y, lt = solved("hospitals"), solved("hotels")
    assert y["rev"]["ly"] == 12071.72 and y["pbt"]["ly"] == pytest.approx(2235.60) and y["pat_own"]["ly"] == pytest.approx(1753.80)
    assert y["s1_avg"]["ly"] == pytest.approx(1512.91, abs=0.01)             # beds in use, worked out
    assert y["rev"]["cy"] == pytest.approx(16497.77, abs=0.01)
    assert y["eps"]["y2"] == pytest.approx(36.39, abs=0.01)
    assert lt["rev"]["ly"] == 128607.77 and lt["pbt"]["ly"] == pytest.approx(29622.64) and lt["pat_own"]["ytd"] == pytest.approx(7293.38)
    assert lt["s1_rev"]["ly"] == pytest.approx(5759 * 365 * 0.717 * 6381 / 100000)   # rooms x days x occupancy x rate
    assert lt["mi"]["ly"] / lt["pat"]["ly"] == pytest.approx(0.19, abs=0.005)        # 19% of profit is the minorities'


def test_one_model_serves_both_companies_and_only_the_choices_differ():
    hospital, hotel = (support.company(support.load(s)) for s in SLUGS)
    assert hospital.choices == {"line1_past": "revenue", "line1_end_to_date": "average",
                                "value_multiple": "ev_ebitda", "value_year": "y1"}
    assert hotel.choices == {"line1_past": "drivers", "line1_end_to_date": "typed",
                             "value_multiple": "pe", "value_year": "y2"}
    a, b = cells(MODEL, hospital), cells(MODEL, hotel)
    assert set(a) == set(b)
    future = [k for k in a if k[1] in ("roy", "cy", "y1", "y2")]
    assert len(future) > 300
    for key in future:                                               # every forecast cell is the same rule,
        if key[0] != "v_mv":                                         # but for the multiple each values on
            assert a[key].tree == b[key].tree and a[key].typed == b[key].typed, key
    differ = {k[0] for k in a if k[1] in ("ly", "ytd") and (a[k].tree is None) != (b[k].tree is None)}
    assert differ == {"s1_avg", "s1_rev", "s1_end"}                 # which figure is worked out
    assert a[("s1_avg", "ly")].tree is not None and a[("s1_rev", "ly")].typed is FIGURE
    assert b[("s1_avg", "ly")].typed is FIGURE and b[("s1_rev", "ly")].tree is not None


def test_a_switched_off_segment_divides_by_nothing():
    """Yatharth has no second or third revenue line: every row of them is nil, none is an error."""
    v = solved("hospitals")
    for row in MODEL.data_rows():
        if row.segment in ("s2", "s3"):
            assert all(x in (0, None) for x in v[row.key].values()), row.key
    assert {r.segment for r in MODEL.data_rows()} == {None, "s2", "s3"}


@pytest.mark.parametrize("slug", SLUGS)
def test_days_and_the_unit_of_the_statements_carry_through(slug):
    fixture = support.load(slug)
    base = solved(slug)
    longer = solved(slug, days={"y1": fixture["assumptions"]["days"]["y1"] + 1})
    days = fixture["assumptions"]["days"]["y1"]
    assert longer["s1_rev"]["y1"] == pytest.approx(base["s1_rev"]["y1"] * (days + 1) / days)
    assert longer["emp"]["y1"] == pytest.approx(base["emp"]["y1"] * (days + 1) / days)       # per place per day
    assert longer["mat"]["y1"] > base["mat"]["y1"]                                            # a share of revenue
    rupees = base["s1_avg"]["y1"] * days * base["s1_occ"]["y1"] * base["s1_price"]["y1"]
    assert base["s1_rev"]["y1"] == pytest.approx(rupees / fixture["unit_rupees"])


def test_the_lower_part_has_leases_minorities_associates_and_other_income():
    base = solved("hotels")
    # leases: interest at the rate on the liability at the start, and the liability counts as debt
    more = solved("hotels", lease_rate={"y1": 0.105})
    assert more["int_lease"]["y1"] - base["int_lease"]["y1"] == pytest.approx(0.01 * base["lease"]["cy"])
    assert more["pbt"]["y1"] - base["pbt"]["y1"] == pytest.approx(-0.01 * base["lease"]["cy"])
    assert base["netdebt"]["y1"] == pytest.approx(base["debt"]["y1"] + base["lease"]["y1"] - base["cash"]["y1"])
    paid = solved("hotels", lease_red={"y1": 500})
    assert paid["lease"]["y1"] == pytest.approx(base["lease"]["y1"] - 500)
    assert paid["cash"]["y1"] == pytest.approx(base["cash"]["y1"] - 500)
    # non-controlling interests: a share of profit after tax, kept out of the owners' profit and equity
    less = solved("hotels", mi_share={"y1": 0.25})
    assert less["mi"]["y1"] == pytest.approx(0.25 * base["pat"]["y1"])
    assert less["pat_own"]["y1"] == pytest.approx(0.75 * base["pat"]["y1"])
    assert less["nci"]["y1"] - base["nci"]["y1"] == pytest.approx(0.06 * base["pat"]["y1"])
    assert less["eq"]["y1"] - base["eq"]["y1"] == pytest.approx(-0.06 * base["pat"]["y1"])
    # associates: profit without cash, carried in investments
    assoc = solved("hotels", assoc={"y1": 100})
    assert assoc["pbt"]["y1"] - base["pbt"]["y1"] == pytest.approx(100)
    assert assoc["onca"]["y1"] - base["onca"]["y1"] == pytest.approx(100)
    assert assoc["cf_ops"]["y1"] - base["cf_ops"]["y1"] == pytest.approx(-100 * base["tax_rate"]["y1"])
    # other income: a yield on the cash and investments held at the start
    richer = solved("hotels", oi_yield={"y1": 0.12})
    assert richer["othinc"]["y1"] - base["othinc"]["y1"] == pytest.approx(0.02 * base["cash"]["cy"])
    for changed in (more, paid, less, assoc, richer):
        assert abs(changed["chk_bs"]["y1"]) < 1e-6 and abs(changed["chk_bs"]["y2"]) < 1e-6


def test_a_ratio_is_worked_out_in_the_past_and_typed_in_the_future():
    plan = cells(MODEL, support.company(support.load("hotels")))
    for ratio in ("tax_rate", "mi_share", "dep_rate", "oi_yield", "debt_rate", "lease_rate"):
        assert plan[(ratio, "ytd")].tree is not None, ratio
        assert plan[(ratio, "cy")].tree is not None, ratio                  # as it came out
        assert plan[(ratio, "roy")].typed is ASSUME and plan[(ratio, "y1")].typed is ASSUME, ratio


def test_the_model_is_checked_data_and_says_what_it_leaves_out():
    check(MODEL)
    assert Model.from_dict(json.loads(json.dumps(MODEL.to_dict()))) == MODEL
    assert MODEL.vetted is False
    assert any("Deferred tax is left out" in note for note in MODEL.notes)
    assert set(MODEL.sectors) == {"hospitals", "hotels"}
    assert MODEL.choices() == {"line1_past": {"drivers", "revenue"}, "line1_end_to_date": {"typed", "average"},
                               "value_multiple": {"ev_ebitda", "pe"}, "value_year": {"cy", "y1", "y2"}}
    picks = [r.key for r in MODEL.data_rows() if any(isinstance(rule, Pick) for rule in r.rules.values())]
    assert sorted(picks) == ["s1_avg", "s1_end", "s1_rev", "v_mps", "v_mv"]


def test_more_forecast_years_are_more_columns_of_the_same_rows():
    longer = occupancy.build(forecast_years=5)
    check(longer)
    assert [p.name for p in longer.periods] == ["ly", "ytd", "roy", "cy", "y1", "y2", "y3", "y4", "y5"]
    assert [r.key for r in longer.data_rows()] == [r.key for r in MODEL.data_rows()]
