"""The revenue blocks (PLAN.md R4, and R7's plant and stores), each solved by hand.

Every case is one line of revenue in the General model's frame (no other
line, the lower part left unfilled: `evaluate(..., partial=True)` works the
revenue out without it). Statements are in lakh (unit_rupees 100,000), the
past is last year (365 days) and a half year (183), the rest of the year 182
days, then 365 and 366. The expected figures are worked by hand, one line of
arithmetic each, from the block's rule as `research/blocks.py` states it.
"""
from __future__ import annotations

import pytest

from research import compose
from research.definition import Company
from research.evaluate import evaluate
from research.models import CATALOGUE
from tests import research_support as support

L = 100_000
DAYS = {"ly": 365, "ytd": 183, "roy": 182, "y1": 365, "y2": 366}


def run(block, values, choices=None, second=None):
    values = {"rev": {"ly": 1, "ytd": 1}, **values}         # the lower part's growth row divides by last year's
    segments = (compose.Segment("s1", block),) + ((second,) if second else ())
    structure = compose.Structure("general", segments, "s1" if block in ("capacity", "units", "subscribers", "people", "stores",
                                                                     "plant") else None)
    model = CATALOGUE.build(structure)
    company = Company(name="Case", unit_name="Rs lakh", sector="pharma", constants={"unit_rupees": L},
                      choices={**support.VALUE_CHOICES, **(choices or {})}, values={"days": DAYS, **values})
    return evaluate(model, company, partial=True).values


def close(a, b):
    return a == pytest.approx(b, rel=1e-12, abs=1e-9)


def test_capacity_places_times_share_filled_times_price_per_day():
    v = run("capacity", {
        "s1_end": {"ly": 100, "ytd": 110}, "s1_avg": {"ly": 95, "ytd": 105},
        "s1_occ": {"ly": 0.6, "ytd": 0.65, "roy": 0.7, "y1": 0.72, "y2": 0.75},
        "s1_price": {"ly": 10000, "ytd": 11000, "roy": 12000},
        "s1_add": {"roy": 10, "y1": 20, "y2": 0}, "s1_cls": {"roy": 0, "y1": 5, "y2": 0},
        "s1_pg": {"y1": 0.05, "y2": 0.04}}, choices={"line1_past": "drivers", "line1_end_to_date": "typed"})
    assert close(v["s1_rev"]["ly"], 95 * 365 * 0.6 * 10000 / L)                 # 2,080.5
    assert close(v["s1_rev"]["ytd"], 105 * 183 * 0.65 * 11000 / L)
    assert close(v["s1_avg"]["roy"], (110 + 120) / 2)                           # opens at the half year's end
    assert close(v["s1_rev"]["roy"], 115 * 182 * 0.7 * 12000 / L)
    cy_avg = (105 * 183 + 115 * 182) / 365
    assert close(v["s1_avg"]["cy"], cy_avg)
    cy_occ = (0.65 * 105 * 183 + 0.7 * 115 * 182) / (cy_avg * 365)
    assert close(v["s1_occ"]["cy"], cy_occ)
    cy_rev = 105 * 183 * 0.65 * 11000 / L + 115 * 182 * 0.7 * 12000 / L
    assert close(v["s1_rev"]["cy"], cy_rev)
    cy_price = cy_rev * L / (cy_avg * 365 * cy_occ)
    assert close(v["s1_price"]["cy"], cy_price)
    assert close(v["s1_end"]["y1"], 120 + 20 - 5)
    assert close(v["s1_rev"]["y1"], (120 + 135) / 2 * 365 * 0.72 * cy_price * 1.05 / L)
    assert close(v["s1_rev"]["y2"], 135 * 366 * 0.75 * cy_price * 1.05 * 1.04 / L)


def test_capacity_run_backwards_works_the_places_out_from_revenue():
    v = run("capacity", {"s1_rev": {"ly": 2080.5, "ytd": 1373.8725}, "s1_end": {"ly": 100},
                         "s1_occ": {"ly": 0.6, "ytd": 0.65}, "s1_price": {"ly": 10000, "ytd": 11000}},
            choices={"line1_past": "revenue", "line1_end_to_date": "average"})
    assert close(v["s1_avg"]["ly"], 2080.5 * L / (365 * 0.6 * 10000))           # 95 places in use
    assert close(v["s1_avg"]["ytd"], 105)
    assert close(v["s1_end"]["ytd"], 105)                                       # the half year ends at its average


def test_units_times_revenue_per_unit_a_year():
    v = run("units", {"s1_end": {"ly": 50, "ytd": 52}, "s1_avg": {"ly": 48, "ytd": 51}, "s1_rev": {"ly": 240, "ytd": 130},
                      "s1_add": {"roy": 4, "y1": 6, "y2": 0}, "s1_rpu": {"roy": 520000, "y1": 540000, "y2": 560000}})
    assert close(v["s1_rpu"]["ytd"], 130 * L / (51 * 183 / 365))                # as it came out, a year's rate
    assert close(v["s1_rev"]["roy"], 54 * 520000 * 182 / 365 / L)
    assert close(v["s1_rev"]["cy"], 130 + 54 * 520000 * 182 / 365 / L)
    assert close(v["s1_rev"]["y1"], 59 * 540000 / L)
    assert close(v["s1_rev"]["y2"], 62 * 560000 * 366 / 365 / L)


def test_price_times_volume():
    v = run("price_volume", {"s1_vol": {"ly": 1000, "ytd": 520, "roy": 540}, "s1_rev": {"ly": 5000, "ytd": 2700},
                             "s1_price": {"roy": 530000}, "s1_vg": {"y1": 0.1, "y2": 0.05}, "s1_pg": {"y1": 0.03, "y2": 0.02}})
    assert close(v["s1_price"]["ly"], 5000 * L / 1000)                         # 5 lakh a unit, worked out
    assert close(v["s1_rev"]["roy"], 540 * 530000 / L)                         # 2,862
    assert close(v["s1_vol"]["cy"], 1060)
    cy_price = (2700 + 2862) * L / 1060
    assert close(v["s1_price"]["cy"], cy_price)
    assert close(v["s1_rev"]["y1"], 1060 * 1.1 * cy_price * 1.03 / L)
    assert close(v["s1_rev"]["y2"], 1060 * 1.1 * 1.05 * cy_price * 1.03 * 1.02 / L)


def test_plant_capacity_times_utilisation_times_price_per_unit_sold():
    v = run("plant", {"s1_end": {"ly": 10_000_000, "ytd": 10_000_000}, "s1_avg": {"ly": 9_500_000, "ytd": 10_000_000},
                      "s1_vol": {"ly": 7_600_000, "ytd": 4_000_000}, "s1_rev": {"ly": 380_000, "ytd": 216_000},
                      "s1_add": {"roy": 1_000_000, "y1": 0, "y2": 2_000_000},
                      "s1_util": {"roy": 0.85, "y1": 0.86, "y2": 0.8}, "s1_price": {"roy": 5500},
                      "s1_pg": {"y1": 0.03, "y2": 0.02}})
    assert close(v["s1_util"]["ly"], 7_600_000 / 9_500_000)                    # 0.8 of the year's capacity
    assert close(v["s1_util"]["ytd"], 4_000_000 * 365 / (10_000_000 * 183))     # a part year's capacity is its days'
    assert close(v["s1_price"]["ly"], 380_000 * L / 7_600_000)                  # Rs 5,000 a tonne, worked out
    assert close(v["s1_price"]["ytd"], 5400)
    roy_avg = (10_000_000 + 11_000_000) / 2
    roy_vol = roy_avg * 0.85 * 182 / 365
    assert close(v["s1_vol"]["roy"], roy_vol)
    assert close(v["s1_rev"]["roy"], roy_vol * 5500 / L)
    cy_vol, cy_rev = 4_000_000 + roy_vol, 216_000 + roy_vol * 5500 / L
    assert close(v["s1_vol"]["cy"], cy_vol)
    assert close(v["s1_price"]["cy"], cy_rev * L / cy_vol)
    cy_avg = (10_000_000 * 183 + roy_avg * 182) / 365
    assert close(v["s1_util"]["cy"], cy_vol * 365 / (cy_avg * 365))
    y1_vol = 11_000_000 * 0.86
    assert close(v["s1_rev"]["y1"], y1_vol * cy_rev * L / cy_vol * 1.03 / L)
    y2_vol = 12_000_000 * 0.8 * 366 / 365
    assert close(v["s1_rev"]["y2"], y2_vol * cy_rev * L / cy_vol * 1.03 * 1.02 / L)


def test_stores_times_revenue_per_store_per_day():
    v = run("stores", {"s1_end": {"ly": 400, "ytd": 420}, "s1_avg": {"ly": 380, "ytd": 410},
                       "s1_rev": {"ly": 55_480, "ytd": 30_012}, "s1_add": {"roy": 30, "y1": 50, "y2": 60},
                       "s1_cls": {"roy": 2, "y1": 4, "y2": 0}, "s1_price": {"roy": 420_000},
                       "s1_sg": {"y1": 0.06, "y2": 0.05}})
    assert close(v["s1_price"]["ly"], 55_480 * L / (380 * 365))                 # Rs 40,000 a store a day
    assert close(v["s1_price"]["ytd"], 30_012 * L / (410 * 183))
    assert close(v["s1_end"]["roy"], 420 + 30 - 2)
    roy_rev = (420 + 448) / 2 * 420_000 * 182 / L
    assert close(v["s1_rev"]["roy"], roy_rev)
    cy_avg = (410 * 183 + 434 * 182) / 365
    assert close(v["s1_avg"]["cy"], cy_avg)
    cy_price = (30_012 + roy_rev) * L / (cy_avg * 365)
    assert close(v["s1_price"]["cy"], cy_price)
    assert close(v["s1_rev"]["y1"], (448 + 494) / 2 * cy_price * 1.06 * 365 / L)
    assert close(v["s1_rev"]["y2"], (494 + 554) / 2 * cy_price * 1.06 * 1.05 * 366 / L)


def test_costs_follow_a_volume_sold_per_unit_and_places_per_day():
    """The plant's costs are per unit sold, no days; the stores' per store per day."""
    def costs(block, extra):
        structure = compose.Structure("per_tonne" if block == "plant" else "stores", (compose.Segment("s1", block),), "s1")
        model = CATALOGUE.build(structure)
        company = Company(name="Case", unit_name="Rs lakh", sector="cement" if block == "plant" else "retail",
                          constants={"unit_rupees": L}, choices=support.VALUE_CHOICES,
                          values={"days": DAYS, "rev": {"ly": 1, "ytd": 1}, **extra})
        return evaluate(model, company, partial=True).values

    plant = costs("plant", {"s1_end": {"ly": 100, "ytd": 100}, "s1_avg": {"ly": 100, "ytd": 100},
                            "s1_vol": {"ly": 80, "ytd": 40}, "s1_rev": {"ly": 400, "ytd": 210}, "frt": {"ly": 96, "ytd": 50},
                            "s1_add": {"roy": 0}, "s1_util": {"roy": 0.9}, "s1_price": {"roy": 520_000},
                            "frt_sh": {"roy": 0.01}, "frt_pu": {"roy": 125_000}})
    assert close(plant["frt_m2"]["ly"], 96 * L / 80)                             # Rs 1.2 lakh a unit sold
    roy_vol = 100 * 0.9 * 182 / 365
    assert close(plant["frt"]["roy"], 0.01 * roy_vol * 520_000 / L + 125_000 * roy_vol / L)
    shop = costs("stores", {"s1_end": {"ly": 10, "ytd": 10}, "s1_avg": {"ly": 10, "ytd": 10}, "s1_rev": {"ly": 100, "ytd": 50},
                            "emp": {"ly": 7.3, "ytd": 3.66}, "s1_add": {"roy": 0}, "s1_cls": {"roy": 0},
                            "s1_price": {"roy": 30_000}, "emp_sh": {"roy": 0}, "emp_pd": {"roy": 2_100}})
    assert close(shop["emp_m2"]["ly"], 7.3 * L / (10 * 365))                     # Rs 200 a store a day
    assert close(shop["emp"]["roy"], 2_100 * 10 * 182 / L)
    assert "frt" not in shop and "pwr" not in shop and "pwr" in plant


def test_subscribers_less_churn():
    v = run("subscribers", {"s1_end": {"ly": 1000, "ytd": 1050}, "s1_avg": {"ly": 980, "ytd": 1030},
                            "s1_rev": {"ly": 2000, "ytd": 1100}, "s1_add": {"roy": 120, "y1": 200, "y2": 200},
                            "s1_churn": {"roy": 0.02, "y1": 0.02, "y2": 0.015},
                            "s1_arpu": {"roy": 18000, "y1": 18500, "y2": 19000}})
    assert close(v["s1_arpu"]["ytd"], 1100 * L / (1030 * 183 * 12 / 365))       # a month's revenue per subscriber
    lost = 0.02 * 1050 * 182 * 12 / 365
    assert close(v["s1_cls"]["roy"], lost)
    end = 1050 + 120 - lost
    assert close(v["s1_end"]["roy"], end)
    assert close(v["s1_rev"]["roy"], (1050 + end) / 2 * 18000 * 182 * 12 / 365 / L)
    lost_y1 = 0.02 * end * 12
    assert close(v["s1_rev"]["y1"], (end + end + 200 - lost_y1) / 2 * 18500 * 12 / L)


def test_headcount_times_utilisation_times_billing_rate():
    v = run("people", {"s1_end": {"ly": 500, "ytd": 520}, "s1_avg": {"ly": 490, "ytd": 510},
                       "s1_util": {"ly": 0.8, "ytd": 0.82, "roy": 0.84, "y1": 0.85, "y2": 0.85},
                       "s1_rev": {"ly": 4900, "ytd": 2600}, "s1_add": {"roy": 20, "y1": 40, "y2": 0},
                       "s1_rate": {"roy": 1_300_000, "y1": 1_350_000, "y2": 1_400_000}})
    assert close(v["s1_rate"]["ly"], 4900 * L / (490 * 0.8))                    # 12.5 lakh a billed person
    assert close(v["s1_rev"]["roy"], 530 * 0.84 * 1_300_000 * 182 / 365 / L)
    assert close(v["s1_util"]["cy"], (0.82 * 510 * 183 + 0.84 * 530 * 182) / ((510 * 183 + 530 * 182)))
    assert close(v["s1_rev"]["y1"], 560 * 0.85 * 1_350_000 / L)


def test_order_book_times_the_share_executed():
    v = run("order_book", {"s1_ob_end": {"ly": 10000, "ytd": 10500}, "s1_ob_in": {"ly": 6000, "ytd": 3000, "roy": 3500,
                                                                               "y1": 7000, "y2": 7000},
                           "s1_rev": {"ly": 5000, "ytd": 2500}, "s1_exec": {"roy": 0.55, "y1": 0.5, "y2": 0.5}})
    assert close(v["s1_exec"]["ytd"], 2500 * 365 / 183 / 10000)                 # a year's pace on the opening book
    executed = 0.55 * 10500 * 182 / 365
    assert close(v["s1_rev"]["roy"], executed)
    assert close(v["s1_ob_end"]["roy"], 10500 + 3500 - executed)
    assert close(v["s1_ob_end"]["cy"], 10500 + 3500 - executed)
    assert close(v["s1_exec"]["cy"], (2500 + executed) / 10000)
    assert close(v["s1_ob_in"]["cy"], 6500)
    book = 10500 + 3500 - executed
    assert close(v["s1_rev"]["y1"], 0.5 * book)
    assert close(v["s1_rev"]["y2"], 0.5 * (book + 7000 - 0.5 * book) * 366 / 365)


def test_growth_on_the_year_before():
    v = run("growth", {"s1_rev": {"ly": 1000, "ytd": 540}, "s1_g": {"cy": 0.1, "y1": 0.08, "y2": 0.06}})
    assert close(v["s1_rev"]["roy"], 1000 * 1.1 - 540)                         # the year's growth, less the part reported
    assert close(v["s1_rev"]["cy"], 1100)
    assert close(v["s1_rev"]["y1"], 1188)
    assert close(v["s1_rev"]["y2"], 1188 * 1.06)


def test_a_share_of_another_line_and_an_analysts_own_formula():
    values = {"rev": {"ly": 1300, "ytd": 700}, "s1_rev": {"ly": 1000, "ytd": 540}, "s1_g": {"cy": 0.1, "y1": 0.1, "y2": 0.1},
              "s2_share": {"roy": 0.3, "y1": 0.32, "y2": 0.34}}
    v = run("growth", values, second=compose.Segment("s2", "share", of="s1", rest_of_revenue=True))
    assert close(v["s2_rev"]["ly"], 1300 - 1000)                               # for the past, revenue less line 1
    assert close(v["s2_share"]["ytd"], 160 / 540)
    assert close(v["s2_rev"]["roy"], 0.3 * 560)
    assert close(v["s2_rev"]["y1"], 0.32 * 1210)
    own = compose.Segment("s2", "share", formula="s1_rev * 0.1 + 5")
    values.pop("s2_share")
    v = run("growth", {**values, "s2_rev": {"ly": 105, "ytd": 59}}, second=own)
    assert close(v["s2_rev"]["roy"], 56 + 5)
    assert close(v["s2_rev"]["y2"], 1331 * 0.1 + 5)
    assert close(v["rev"]["y2"], 1331 + 1331 * 0.1 + 5)                       # revenue adds up every line


def test_a_model_that_counts_no_places_has_costs_as_a_share_of_revenue_alone():
    v = run("growth", {"s1_rev": {"ly": 1000, "ytd": 540}, "s1_g": {"cy": 0.1, "y1": 0.08},
                       "mat_sh": {"roy": 0.4, "y1": 0.42}, "mat": {"ly": 390, "ytd": 220}, "rev": {"ly": 1000, "ytd": 540}})
    assert close(v["mat"]["roy"], 0.4 * 560)
    assert close(v["mat"]["cy"], 220 + 0.4 * 560)
    assert close(v["mat"]["y1"], 0.42 * 1188)
    assert close(v["mat_m1"]["ly"], 0.39)
