"""The model definition format and its evaluator (PLAN.md R4 part b), on a
model small enough to solve by hand."""
import json

import pytest

from research import pieces
from research.definition import (ASSUME, FIGURE, Company, Heading, InputError, Model, ModelError, Period, Pick, Row,
                                 check)
from research.evaluate import EvaluationError, evaluate

PERIODS = (Period("fy25", "past_year"), Period("fy26", "forecast", prev="fy25"), Period("fy27", "forecast", prev="fy26"))
FORWARDS = "places * days * occ * price / unit_rupees"


def toy(extra=()):
    """One revenue line: places x share filled x price per day, the past run either way."""
    return Model(
        id="toy", name="Toy", periods=PERIODS, constants={"unit_rupees": "Rupees in one unit of the statements"},
        rows=(
            Heading("Revenue [{line}]"),
            Row("days", "Days in the period", "days", {"past": FIGURE, "forecast": ASSUME}),
            Row("occ", "Share filled", "share", {"past": FIGURE, "forecast": ASSUME}),
            Row("price_growth", "Price growth", "share", {"forecast": ASSUME}),
            Row("price", "Price per day", "rupees", {"past": FIGURE, "forecast": "price[prev] * (1 + price_growth)"}),
            Row("places", "{places}", "count", {
                "past": Pick("past", {"drivers": FIGURE, "revenue": "room_rev * unit_rupees / (days * occ * price)"}),
                "forecast": "places[prev] + added"}),
            Row("added", "Added", "count", {"forecast": ASSUME}),
            Row("room_rev", "Revenue", "money", {
                "past": Pick("past", {"drivers": FORWARDS, "revenue": FIGURE}), "forecast": FORWARDS}, bold=True),
            *extra,
        ),
        sectors={"hospitals": {"line": "Beds", "places": "Beds in use"}, "hotels": {"line": "Rooms", "places": "Rooms"}})


ASSUMED = {"days": {"fy26": 365, "fy27": 366}, "occ": {"fy26": 0.65, "fy27": 0.66},
           "price_growth": {"fy26": 0.05, "fy27": 0.04}, "added": {"fy26": 100, "fy27": 0}}


def yatharth(**more):
    """Yatharth's FY25 as its presentation prints it: revenue, occupancy and price. Rs million."""
    values = {"days": {"fy25": 365}, "occ": {"fy25": 0.61}, "price": {"fy25": 30829}, "room_rev": {"fy25": 8805}}
    for source in (ASSUMED, more):
        for row, by_period in source.items():
            values.setdefault(row, {}).update(by_period)
    return Company(name="Y", sector="hospitals", unit_name="Rs million", constants={"unit_rupees": 1_000_000},
                   choices={"past": "revenue"}, values=values)


def printed_beds():
    """The same year with the 1,605 beds the presentation prints, run forwards."""
    co = yatharth()
    values = {k: dict(v) for k, v in co.values.items() if k != "room_rev"}
    values["places"] = {"fy25": 1605}
    return Company(name="Y", sector="hospitals", unit_name="Rs million", constants=co.constants,
                   choices={"past": "drivers"}, values=values)


def test_the_past_runs_backwards_when_the_company_prints_revenue():
    """PLAN.md R4, "Model 1 by hand": 8,805 x 1,000,000 / (365 x 0.61 x 30,829) = 1,282.77 beds in use."""
    v = evaluate(toy(), yatharth()).values
    assert v["places"]["fy25"] == pytest.approx(1282.77, abs=0.01)
    assert v["room_rev"]["fy25"] == 8805


def test_the_past_runs_forwards_when_the_company_prints_its_places():
    """The printed 1,605 beds give 11,017, not the 8,805 reported: the finding of the hand build."""
    v = evaluate(toy(), printed_beds()).values
    assert v["places"]["fy25"] == 1605
    assert v["room_rev"]["fy25"] == pytest.approx(1605 * 365 * 0.61 * 30829 / 1e6, abs=1e-9)
    assert v["room_rev"]["fy25"] == pytest.approx(11016.84, abs=0.01)


def test_the_forecast_is_the_same_formula_whichever_way_the_past_ran():
    v = evaluate(toy(), yatharth()).values
    beds = 8805e6 / (365 * 0.61 * 30829)
    assert v["price"]["fy26"] == pytest.approx(30829 * 1.05)
    assert v["price"]["fy27"] == pytest.approx(30829 * 1.05 * 1.04)
    assert v["places"]["fy27"] == pytest.approx(beds + 100)
    assert v["room_rev"]["fy26"] == pytest.approx((beds + 100) * 365 * 0.65 * 30829 * 1.05 / 1e6)
    assert v["room_rev"]["fy27"] == pytest.approx((beds + 100) * 366 * 0.66 * 30829 * 1.05 * 1.04 / 1e6)


def test_an_empty_cell_is_none_and_a_blank_past_figure_counts_as_nil():
    model = toy(extra=(Row("other", "Other revenue", "money", {"past": FIGURE}),
                       Row("total", "Total", "money", {"past": "room_rev + other"})))
    v = evaluate(model, yatharth()).values
    assert v["price_growth"]["fy25"] is None        # no rule
    assert v["other"]["fy25"] is None               # a figure nobody typed
    assert v["total"]["fy25"] == 8805               # read as nil, as Excel reads an empty cell
    assert v["total"]["fy26"] is None


def test_a_missing_assumption_is_named():
    co = yatharth()
    del co.values["occ"]["fy26"]
    del co.values["added"]["fy27"]
    with pytest.raises(InputError) as e:
        evaluate(toy(), co)
    assert "occ" in str(e.value) and "fy26" in str(e.value) and "added" in str(e.value)


def test_a_figure_typed_where_the_model_works_it_out_is_refused():
    with pytest.raises(InputError, match="places"):
        evaluate(toy(), yatharth(places={"fy25": 1605}))
    with pytest.raises(InputError, match="room_rev"):
        evaluate(toy(), yatharth(room_rev={"fy26": 1}))


@pytest.mark.parametrize("values", [{"nothing": {"fy25": 1}}, {"occ": {"fy99": 1}}, {"occ": {"fy26": "0.6"}},
                                    {"occ": {"fy26": True}}, {"occ": {"fy26": float("nan")}}, {"occ": {"fy26": float("inf")}}])
def test_figures_must_be_numbers_for_rows_and_periods_that_exist(values):
    with pytest.raises(InputError):
        evaluate(toy(), yatharth(**values))


def test_choices_must_be_made_from_the_options():
    co = yatharth()
    for choices in ({}, {"past": "sideways"}, {"past": "revenue", "extra": "x"}):
        with pytest.raises(InputError):
            evaluate(toy(), Company(name="Y", sector="hospitals", unit_name="u", constants=co.constants,
                                    choices=choices, values=co.values))


def test_constants_and_sector_must_be_given():
    co = yatharth()
    with pytest.raises(InputError, match="unit_rupees"):
        evaluate(toy(), Company(name="Y", sector="hospitals", unit_name="u", constants={}, choices=co.choices, values=co.values))
    with pytest.raises(InputError, match="cinemas"):
        evaluate(toy(), Company(name="Y", sector="cinemas", unit_name="u", constants=co.constants, choices=co.choices,
                                values=co.values))


def test_a_plain_division_by_nil_names_the_row_and_the_period():
    with pytest.raises(EvaluationError) as e:
        evaluate(toy(), yatharth(occ={"fy25": 0}))
    assert "places" in str(e.value) and "fy25" in str(e.value)


def _model(*rows, periods=PERIODS):
    return Model(id="m", name="M", periods=periods, constants={}, rows=rows)


@pytest.mark.parametrize("rows, why", [
    ((Row("a", "A", "money", {"all": "b + 1"}),), "b"),                                           # an unknown row
    ((Row("a", "A", "money", {"all": "1"}), Row("a", "A again", "money", {"all": "2"})), "a"),    # the same key twice
    ((Row("a", "A", "money", {"all": "a[fy99]"}),), "fy99"),                                      # an unknown period
    ((Row("a", "A", "money", {"all": "a[prev] + 1"}),), "fy25"),                                  # nothing before the first period
    ((Row("a", "A", "furlongs", {"all": "1"}),), "furlongs"),                                     # an unknown unit
    ((Row("a", "A", "money", {"sometimes": "1"}),), "sometimes"),                                 # an unknown kind of period
    ((Row("a", "A", "money", {"all": "1 +"}),), "a"),                                             # a formula that cannot be read
    ((Row("max", "A", "money", {"all": "1"}),), "max"),                                           # a reserved word
    ((Row("prev", "A", "money", {"all": "1"}),), "prev"),
    ((Row("fy25", "A", "money", {"all": "1"}),), "fy25"),
    ((Row("1a", "A", "money", {"all": "1"}),), "1a"),
    ((Row("a", "A", "money", {"past": Pick("c", {"x": "1", "y": "2"}), "forecast": Pick("c", {"x": "1"})}),), "c"),
    ((Row("a", "A", "money", {"past": Pick("c", {"x": Pick("d", {"z": "1"})})}),), "c"),          # a choice inside a choice
])
def test_a_wrong_definition_is_refused_and_says_what_is_wrong(rows, why):
    with pytest.raises(ModelError, match=why):
        check(_model(*rows))


def test_periods_must_be_well_formed():
    with pytest.raises(ModelError, match="fy24"):
        check(_model(Row("a", "A", "money", {"all": "1"}), periods=(Period("fy25", "forecast", prev="fy24"),)))
    with pytest.raises(ModelError, match="sideways"):
        check(_model(Row("a", "A", "money", {"all": "1"}), periods=(Period("fy25", "sideways"),)))
    with pytest.raises(ModelError, match="fy25"):
        check(_model(Row("a", "A", "money", {"all": "1"}), periods=(Period("fy25", "forecast"), Period("fy25", "forecast"))))


def test_a_constant_has_no_period_and_cannot_share_a_rows_name():
    with pytest.raises(ModelError, match="k"):
        check(Model(id="m", name="M", periods=PERIODS, constants={"k": "K"}, rows=(Row("a", "A", "money", {"forecast": "k[prev]"}),)))
    with pytest.raises(ModelError, match="a"):
        check(Model(id="m", name="M", periods=PERIODS, constants={"a": "K"}, rows=(Row("a", "A", "money", {"all": "1"}),)))


def test_rows_that_go_in_a_circle_are_refused_with_the_circle_named():
    model = _model(Row("a", "A", "money", {"all": "b + 1"}), Row("b", "B", "money", {"all": "c * 2"}),
                   Row("c", "C", "money", {"all": "a"}))
    with pytest.raises(ModelError) as e:
        evaluate(model, Company(name="X", unit_name="u"))
    assert all(("%s[fy25]" % k) in str(e.value) for k in "abc")


def test_a_rule_is_found_by_period_name_then_kind_then_group():
    periods = (Period("ly", "past_year"), Period("ytd", "to_date"), Period("roy", "rest"),
               Period("cy", "current", prev="ly"), Period("y1", "forecast", prev="cy"))
    model = _model(
        Row("a", "A", "money", {"all": "1", "year_end": "2", "past": "3", "to_date": "4", "past_year": "6", "ly": "5"}),
        Row("b", "B", "money", {"year_end": "2"}),
        Row("c", "C", "money", {"all": "1", "forecast": None}),
        periods=periods)
    v = evaluate(model, Company(name="X", unit_name="u")).values
    assert v["a"] == {"ly": 5, "ytd": 4, "roy": 1, "cy": 2, "y1": 2}
    assert v["b"] == {"ly": 2, "ytd": 2, "roy": None, "cy": 2, "y1": 2}
    assert v["c"] == {"ly": 1, "ytd": 1, "roy": 1, "cy": 1, "y1": None}       # None switches a wider rule off


def test_a_segment_switched_off_reads_nil_everywhere_and_divides_nothing():
    model = toy(extra=(
        Row("s2_units", "Units", "count", {"past": FIGURE, "forecast": ASSUME}, segment="s2"),
        Row("s2_rev", "Fees", "money", {"past": FIGURE, "forecast": "s2_units * s2_fee / unit_rupees"}, segment="s2"),
        Row("s2_fee", "Fee per unit", "rupees", {"past": "div(s2_rev * unit_rupees, s2_units)", "forecast": ASSUME}, segment="s2"),
        Row("total", "Total", "money", {"all": "room_rev + s2_rev"}),
    ))
    co = yatharth()
    off = Company(name="Y", sector="hospitals", unit_name="u", constants=co.constants, choices=co.choices,
                  values=co.values, off=("s2",))
    v = evaluate(model, off).values
    assert v["s2_units"] == {"fy25": 0, "fy26": 0, "fy27": 0}
    assert v["s2_fee"]["fy25"] == 0 and v["s2_rev"]["fy27"] == 0
    assert v["total"]["fy25"] == 8805
    with pytest.raises(InputError, match="s2"):         # switched on, its assumptions are asked for
        evaluate(model, co)
    with pytest.raises(InputError, match="s9"):
        evaluate(model, Company(name="Y", sector="hospitals", unit_name="u", constants=co.constants,
                                choices=co.choices, values=co.values, off=("s9",)))


def test_a_model_is_data_and_comes_back_from_json_the_same():
    model = toy()
    text = json.dumps(model.to_dict())
    again = Model.from_dict(json.loads(text))
    assert again == model
    assert evaluate(again, yatharth()).values == evaluate(model, yatharth()).values


# ---- the shared pieces, each by hand ------------------------------------------------

YEAR = (Period("ly", "past_year"), Period("ytd", "to_date"), Period("roy", "rest"),
        Period("cy", "current", prev="ly"), Period("y1", "forecast", prev="cy"))


def piece_model(*rows):
    return Model(id="p", name="P", periods=YEAR, constants={"unit_rupees": "Rupees"}, rows=rows)


def piece_company(**values):
    return Company(name="P", unit_name="Rs lakh", constants={"unit_rupees": 100_000}, values=values)


DAYS = Row("days", "Days", "days", {"past": FIGURE, "rest": ASSUME, "current": "days[ytd] + days[roy]", "forecast": ASSUME})
DAY_VALUES = {"ly": 365, "ytd": 183, "roy": 182, "y1": 365}


def test_a_running_count_opens_where_the_last_one_ended_and_averages_by_days():
    model = piece_model(DAYS, *pieces.running_count("rooms", "Rooms", closures=True))
    v = evaluate(model, piece_company(
        days=DAY_VALUES, rooms_end={"ly": 5000, "ytd": 5100}, rooms_avg={"ly": 4900, "ytd": 5050},
        rooms_add={"roy": 200, "y1": 300}, rooms_cls={"roy": 50, "y1": 0})).values
    assert v["rooms_open"] == {"ly": None, "ytd": None, "roy": 5100, "cy": None, "y1": 5250}
    assert v["rooms_end"] == {"ly": 5000, "ytd": 5100, "roy": 5250, "cy": 5250, "y1": 5550}
    assert v["rooms_avg"]["roy"] == 5175                                    # (5,100 + 5,250) / 2
    assert v["rooms_avg"]["cy"] == pytest.approx((5050 * 183 + 5175 * 182) / 365)
    assert v["rooms_avg"]["y1"] == 5400


def test_a_cost_is_a_share_of_revenue_plus_an_amount_per_place_per_day():
    rev = Row("rev", "Revenue", "money", {"past": FIGURE, "rest": ASSUME, "current": "rev[ytd] + rev[roy]", "forecast": ASSUME})
    rooms = Row("rooms", "Rooms", "count", {"all": FIGURE, "rest": ASSUME, "forecast": ASSUME})
    model = piece_model(DAYS, rev, rooms, *pieces.cost("emp", "Employees", revenue="rev", places="rooms", place="room"))
    v = evaluate(model, piece_company(
        days=DAY_VALUES, rev={"ly": 120000, "ytd": 60000, "roy": 70000, "y1": 140000},
        rooms={"ly": 5000, "ytd": 5000, "roy": 5200, "cy": 5100, "y1": 5400},
        emp={"ly": 21000, "ytd": 11000}, emp_sh={"roy": 0.02, "y1": 0}, emp_pd={"roy": 1100, "y1": 1200})).values
    # the rest of the year: 2% of 70,000 + Rs 1,100 x 5,200 rooms x 182 days, in lakh
    assert v["emp"]["roy"] == pytest.approx(0.02 * 70000 + 1100 * 5200 * 182 / 100000)
    assert v["emp"]["roy"] == pytest.approx(11810.40)
    assert v["emp"]["cy"] == pytest.approx(11000 + 11810.40)
    assert v["emp"]["y1"] == pytest.approx(1200 * 5400 * 365 / 100000)      # one of the two left at nil
    # as it came out, in every period
    assert v["emp_m1"]["ly"] == pytest.approx(21000 / 120000)
    assert v["emp_m2"]["ly"] == pytest.approx(21000 * 100000 / (5000 * 365))
    assert v["emp_m2"]["cy"] == pytest.approx(22810.40 * 100000 / (5100 * 365))


@pytest.mark.parametrize("basis", ["opening", "average"])
def test_a_balance_times_a_rate_for_a_whole_year_and_for_part_of_one(basis):
    debt = Row("debt", "Borrowings", "money", {"past_year": FIGURE, "current": "debt[prev] - 1000", "forecast": "debt[prev] - 2000"})
    model = piece_model(DAYS, debt, *pieces.balance_rate("interest", "Interest", "rate", "Rate", balance="debt", basis=basis))
    v = evaluate(model, piece_company(days=DAY_VALUES, debt={"ly": 10000}, interest={"ly": 900, "ytd": 420},
                                      rate={"roy": 0.08, "y1": 0.09})).values
    base_roy = 10000 if basis == "opening" else (10000 + 9000) / 2
    base_y1 = 9000 if basis == "opening" else (9000 + 7000) / 2
    assert v["interest"]["roy"] == pytest.approx(0.08 * base_roy * 182 / 365)
    assert v["interest"]["cy"] == pytest.approx(420 + 0.08 * base_roy * 182 / 365)
    assert v["interest"]["y1"] == pytest.approx(0.09 * base_y1)
    # the rate as it came out: a part year scaled to a year on the opening balance
    assert v["rate"]["ly"] is None
    assert v["rate"]["ytd"] == pytest.approx(420 * 365 / 183 / 10000)
    assert v["rate"]["cy"] == pytest.approx(v["interest"]["cy"] / (10000 if basis == "opening" else 9500))


def test_a_balance_of_nil_gives_a_rate_of_nil_not_an_error():
    debt = Row("debt", "Borrowings", "money", {"past_year": FIGURE, "current": "debt[prev]", "forecast": "debt[prev]"})
    model = piece_model(DAYS, debt, *pieces.balance_rate("interest", "Interest", "rate", "Rate", balance="debt", basis="average"))
    v = evaluate(model, piece_company(days=DAY_VALUES, debt={"ly": 0}, interest={"ly": 0, "ytd": 0}, rate={"roy": 0.08, "y1": 0.09})).values
    assert v["rate"]["ytd"] == 0 and v["rate"]["cy"] == 0 and v["interest"]["y1"] == 0


def test_a_ratio_is_worked_out_in_the_past_typed_in_the_future_and_as_it_came_out_this_year():
    pat = Row("pat", "Profit", "money", {"past": FIGURE, "rest": ASSUME, "current": "pat[ytd] + pat[roy]", "forecast": ASSUME})
    mi = Row("mi", "Minorities", "money", {"past": FIGURE, "rest": "mi_share * pat", "current": "mi[ytd] + mi[roy]", "forecast": "mi_share * pat"})
    model = piece_model(pat, mi, pieces.ratio("mi_share", "Share", "div(mi, pat)"))
    v = evaluate(model, piece_company(pat={"ly": 1000, "ytd": 400, "roy": 600, "y1": 1200}, mi={"ly": 190, "ytd": 60},
                                      mi_share={"roy": 0.2, "y1": 0.25})).values
    assert v["mi_share"] == {"ly": 0.19, "ytd": 0.15, "roy": 0.2, "cy": pytest.approx(0.18), "y1": 0.25}
    assert v["mi"]["y1"] == 300
