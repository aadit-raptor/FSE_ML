"""Risk warnings computed, not written in (PLAN.md 2.8).

Every figure a warning shows must be either recomputed here from the deal's
own model run or read from the published table it names, and the words in the
web catalogue may hold no number of their own. Hand-checked against the
default deal: debt 600 on EBITDA 100 at close, year-one EBIT 88.85 against
interest 46.20, and a mezzanine bullet of 180 due in year 5 beside 21 of senior
amortisation, with 48.28 of cash flow to pay it.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.config import resolve_config
from core.deal import DEAL_MONEY_KEYS, DealInputs, in_millions, run_deal
from core.debt import TrancheSpec
from core.risk_sources import (
    COVERAGE_BANDS, CUMULATIVE_DEFAULT_PCT, LEVERAGE_GUIDANCE_X, SOURCES, cumulative_default_pct,
    is_speculative, rating_for_coverage,
)
from core.risk_warnings import WARNING_MONEY_KEYS, risk_warnings, unfunded_repayments

client = TestClient(app)
EN = Path(__file__).resolve().parents[1] / "web" / "messages" / "en.json"


def warnings_for(**overrides):
    d, cfg = in_millions(DealInputs(**overrides), resolve_config())
    result = run_deal(d, cfg)
    return {w["id"]: w for w in risk_warnings(d, result)}, result


# ---------------------------------------------------------------------------
# The default deal, by hand
# ---------------------------------------------------------------------------
def test_the_default_deal_raises_the_rating_and_unfunded_warnings_only():
    found, _ = warnings_for()
    # 600 / 100 is exactly 6.0x: the guidance flags debt *above* it
    assert set(found) == {"implied_rating", "unfunded_repayment"}


def test_year_one_coverage_maps_to_b_plus_and_reads_sp_five_year_default_rate():
    w = warnings_for()[0]["implied_rating"]
    f = w["figures"]
    assert f["coverage"] == pytest.approx(88.85 / 46.20)       # 1.923x
    assert w["labels"]["rating"] == "B+"                        # Damodaran: 1.75 to 2.00
    assert (f["band_low"], f["band_high"]) == (1.75, 2.00)
    assert f["years"] == 5
    assert f["default_pct"] == 12.88                            # S&P table 26, B+, Y5


def test_the_year_five_bullet_is_reported_as_unfunded_by_the_cash_it_lacks():
    f = warnings_for()[0]["unfunded_repayment"]["figures"]
    assert f["year"] == 5
    assert f["repayment_due"] == pytest.approx(201.0)
    assert f["unfunded"] == pytest.approx(201.0 - 48.28, abs=0.01)   # 152.72
    assert f["count"] == 1
    assert f["unfunded_total"] == pytest.approx(f["unfunded"])


def test_leverage_above_six_times_is_flagged_with_the_supervisors_line():
    found, _ = warnings_for(debt_pct=70.0)     # 70% of a 10x entry: 700 of debt on EBITDA 100
    w = found["leverage_above_guidance"]
    assert w["figures"]["leverage"] == pytest.approx(0.70 * 10.0)
    assert w["figures"]["threshold"] == LEVERAGE_GUIDANCE_X == 6.0
    assert {s["id"] for s in w["sources"]} == {"ecb_leveraged_2017", "us_leveraged_2013"}


def test_leases_on_the_post_view_count_as_debt_and_add_back_their_cost():
    found, result = warnings_for(debt_pct=50.0, accounting_standard="ifrs", lease_cost=10.0,
                                 lease_liability=200.0)
    debt = result.debt_schedule.total_beginning_debt[0]
    # IFRS EBITDA 100 is before lease costs: the valuation EBITDA is 100 again
    assert found["leverage_above_guidance"]["figures"]["leverage"] == pytest.approx((debt + 200.0) / 100.0)


def test_a_lightly_levered_deal_raises_nothing():
    found, _ = warnings_for(debt_pct=20.0)
    assert found == {}


def test_a_year_whose_ebitda_does_not_cover_interest_is_named_with_its_coverage():
    found, result = warnings_for(base_rate=18.0, mezz_spread=6.0)
    om = result.operating_model
    w = found["interest_exceeds_ebitda"]["figures"]
    covers = [e / i for e, i in zip(om.ebitda, om.interest_expense)]
    short = [t + 1 for t, c in enumerate(covers) if c < 1]
    assert short, "the case must have a short year"
    assert w["year"] == short[0]
    assert w["count"] == len(short)
    assert w["coverage"] == pytest.approx(min(covers))


# ---------------------------------------------------------------------------
# The unfunded amount is exactly the cash the model conjures (finding 11)
# ---------------------------------------------------------------------------
HEAVY = TrancheSpec(name="Term loan", kind="amortising_term_loan", amount=350.0, fixed_rate=6.0,
                    amort_pct=30.0, sweep=True, sweep_priority=1, maturity_years=7)


def rcf(amount):
    return TrancheSpec(name="RCF", kind="revolver", amount=amount, drawn_pct=0.0, fixed_rate=5.0,
                       allow_redraw=True, sweep=False, sweep_priority=2, maturity_years=6)


@pytest.mark.parametrize("deal", [
    {},
    {"mincash": 20.0},
    {"debt_pct": 75.0, "growth": -3.0},
    {"tranches": [HEAVY], "mincash": 20.0},
    {"tranches": [HEAVY, rcf(10.0)], "mincash": 20.0},
    {"tranches": [HEAVY, rcf(400.0)], "mincash": 20.0},
])
def test_cash_reconciles_once_the_unfunded_amount_is_counted_as_a_source(deal):
    """Cash at the end of each year = cash at the start + cash flow - repayments
    - sweep + revolver draws + the unfunded amount: nothing else moves it."""
    _, result = warnings_for(**deal)
    ds, cf = result.debt_schedule, result.cash_flow
    gaps = unfunded_repayments(result)
    cash = result.params.minimum_cash
    for t in range(len(ds.years)):
        drawn = sum(rows[t].redrawn for rows in ds.schedule.values())
        expected = (cash + cf.levered_fcf[t] - ds.total_mandatory_repayment[t] - ds.total_cash_sweep[t]
                    + drawn + gaps[t])
        assert ds.cash_balance[t] == pytest.approx(expected, abs=0.02), f"year {t + 1}"
        cash = ds.cash_balance[t]


def test_a_revolver_large_enough_leaves_nothing_unfunded():
    without, _ = warnings_for(tranches=[HEAVY], mincash=20.0)
    small, _ = warnings_for(tranches=[HEAVY, rcf(10.0)], mincash=20.0)
    large, _ = warnings_for(tranches=[HEAVY, rcf(400.0)], mincash=20.0)
    assert without["unfunded_repayment"]["figures"]["unfunded_total"] > \
        small["unfunded_repayment"]["figures"]["unfunded_total"] > 0
    assert "unfunded_repayment" not in large


# ---------------------------------------------------------------------------
# Sources: every warning names them, and the tables hang together
# ---------------------------------------------------------------------------
CASES = [{}, {"debt_pct": 70.0}, {"base_rate": 18.0, "mezz_spread": 6.0}, {"debt_pct": 85.0, "growth": -5.0}]


@pytest.mark.parametrize("deal", CASES)
def test_every_warning_shows_its_sources_in_full(deal):
    for w in warnings_for(**deal)[0].values():
        assert w["sources"], w["id"]
        for s in w["sources"]:
            assert SOURCES[s["id"]] == {k: v for k, v in s.items() if k != "id"}
            assert s["publisher"] and s["title"] and s["detail"]
            if s["id"] != "deal_model":
                assert s["url"].startswith("https://") and s["published"]


@pytest.mark.parametrize("deal", CASES)
def test_every_published_figure_is_the_one_in_its_table(deal):
    found, result = warnings_for(**deal)
    if "implied_rating" in found:
        w = found["implied_rating"]
        rating, low, high = rating_for_coverage(w["figures"]["coverage"])
        assert w["labels"]["rating"] == rating and is_speculative(rating)
        assert w["figures"]["default_pct"] == cumulative_default_pct(rating, result.params.holding_period)
    if "leverage_above_guidance" in found:
        assert found["leverage_above_guidance"]["figures"]["threshold"] == LEVERAGE_GUIDANCE_X


def test_default_rates_rise_with_the_horizon_and_with_weaker_ratings():
    rows = list(CUMULATIVE_DEFAULT_PCT.values())
    for row in rows:
        assert len(row) == 15
        assert all(a <= b for a, b in zip(row, row[1:]))
    # The study's rating levels are not strictly ordered at every horizon
    # (AA- defaults less than AA after year 3), so compare categories: each
    # speculative grade defaults more than every investment grade, and CCC/C most
    spec = ["BB+", "BB", "BB-", "B+", "B", "B-"]
    for horizon in range(15):
        best_spec = min(CUMULATIVE_DEFAULT_PCT[r][horizon] for r in spec)
        worst_ig = max(CUMULATIVE_DEFAULT_PCT[r][horizon] for r in list(CUMULATIVE_DEFAULT_PCT)[:10])
        assert best_spec > worst_ig
        assert CUMULATIVE_DEFAULT_PCT["CCC/C"][horizon] == max(r[horizon] for r in rows)


def test_coverage_bands_tile_the_line_in_rising_order():
    lows = [low for low, _ in COVERAGE_BANDS]
    assert lows == sorted(lows) and len(set(lows)) == len(lows)
    assert rating_for_coverage(-5.0)[0] == "D"
    assert rating_for_coverage(1.75)[0] == "B+" and rating_for_coverage(1.7499)[0] == "B"
    assert rating_for_coverage(50.0) == ("AAA", 8.50, None)
    assert not is_speculative("BBB") and is_speculative("BB+") and is_speculative("D")


# ---------------------------------------------------------------------------
# The API: units, and the catalogue holds no number of its own
# ---------------------------------------------------------------------------
def run_api(inputs):
    r = client.post("/api/deal/run", json={"inputs": inputs})
    assert r.status_code == 200, r.text
    return {w["id"]: w for w in r.json()["risk_warnings"]}


def test_the_api_answers_the_warnings_and_a_deal_in_thousands_scales_only_money():
    millions = run_api({})
    thousands = run_api({"unit": "thousands", "ebitda": 100_000.0, "mincash": 0.0})
    assert set(millions) == set(thousands) == {"implied_rating", "unfunded_repayment"}
    for wid, w in millions.items():
        for key, value in w["figures"].items():
            other = thousands[wid]["figures"][key]
            expected = value * 1000 if key in WARNING_MONEY_KEYS else value
            assert other == pytest.approx(expected, rel=1e-9), (wid, key)


def test_warning_money_keys_are_converted_with_the_rest_of_the_deal():
    assert WARNING_MONEY_KEYS <= DEAL_MONEY_KEYS


def _warning_messages():
    return json.loads(EN.read_text(encoding="utf-8"))["warnings"]


def test_every_warning_has_words_and_the_words_hold_no_number():
    """A number in a warning must come from its figures, never from the text."""
    messages = _warning_messages()
    ids = {"leverage_above_guidance", "implied_rating", "interest_exceeds_ebitda", "unfunded_repayment"}
    assert ids <= set(messages)
    for key, message in messages.items():
        text = re.sub(r"\{[a-zA-Z_]+\}", "", message)          # simple arguments
        text = re.sub(r"\{[a-zA-Z_]+, ?(plural|select),", "", text)   # plural/select heads
        text = re.sub(r"(=\d+|zero|one|two|few|many|other) ?\{", "{", text)  # branch selectors
        assert not re.search(r"\d", text), f"warnings.{key} writes a number in: {message}"


@pytest.mark.parametrize("deal", CASES)
def test_every_argument_a_warning_message_asks_for_is_a_figure_or_label_it_carries(deal):
    messages = _warning_messages()
    for wid, w in warnings_for(**deal)[0].items():
        asked = set(re.findall(r"\{([a-zA-Z_]+)[,}]", messages[wid]))
        assert asked <= set(w["figures"]) | set(w["labels"]), (wid, asked - set(w["figures"]) - set(w["labels"]))
