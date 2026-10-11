"""Live sliders for any deal (PLAN.md 5.8): core/live.py, POST /api/montecarlo/live.

The live estimate is the simulation itself on a small, fixed set of paths,
so every test here holds it to what Run Monte Carlo answers for the same
deal, the same number of paths and the same seed: a slider is an input, and
moving it must give exactly the run with that input changed.
"""
import dataclasses
import sys
import tracemalloc

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import limits
from api.auth import AuthUser, require_user
from api.main import app
from core import live
from core.config import resolve_config
from core.deal import DealInputs
from core.debt import equivalent_tranches, spec_from_kind
from core.montecarlo import MCInputs

client = TestClient(app)

TERM_LOAN = {"name": "Term loan", "kind": "institutional_term_loan", "amount": 420.0, "floating": True,
             "reference_level": 4.0, "margin": 3.5, "floor": 0.5, "amort_pct": 1.0, "sweep": True,
             "maturity_years": 7}
NOTES = {"name": "Notes", "kind": "senior_notes", "amount": 150.0, "floating": False, "fixed_rate": 9.0,
         "maturity_years": 8}
PIK = {"name": "PIK", "kind": "pik_notes", "amount": 60.0, "floating": False, "fixed_rate": 12.0,
       "pik_share": 100.0, "maturity_years": 8}
REVOLVER = {"name": "RCF", "kind": "revolver", "amount": 80.0, "drawn_pct": 0.0, "floating": True,
            "reference_level": 4.0, "margin": 3.0, "commitment_fee_pct": 0.5, "sweep": True,
            "allow_redraw": True, "maturity_years": 6}

# Deals of every kind the model knows: the live estimate has no trained range
DEALS = {
    "the default deal": {},
    "other terms": {"deal": {"entry_mult": 13.5, "hold": 7, "opex": 27.0, "tax": 30.0, "senior_pct": 55.0,
                             "mezz_spread": 6.0, "debt_pct": 48.0},
                    "mc": {"entry_mult": 13.5, "hold": 7, "rate_std": 2.5},
                    "settings": {"tx_fee_pct": 3.5, "fin_fee_pct": 1.0, "def_senior_amort": 2.0}},
    "listed facilities": {"deal": {"tranches": [REVOLVER, TERM_LOAN, NOTES, PIK]}},
    "tax rules": {"deal": {"tax_interest_limit": "ebitda_share", "tax_interest_limit_pct": 30.0,
                           "tax_loss_carryforward": True, "tax_loss_limit_pct": 60.0, "tax_minimum_pct": 15.0}},
    "leases": {"deal": {"accounting_standard": "ifrs", "lease_view": "post_ifrs16", "lease_cost": 10.0,
                        "lease_liability": 60.0}},
    "thousands": {"deal": {"unit": "thousands", "ebitda": 100_000.0, "mincash": 5_000.0},
                  "mc": {"ebitda": 100_000.0}},
}
SUMMARY = {"irr_p50": "median_irr", "irr_p5": "p5_irr", "irr_p95": "p95_irr", "irr_mean": "mean_irr",
           "p_above_hurdle": "p_above_hurdle", "p_wipeout": "wipeout_rate", "p_loss": "p_loss"}


def ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


def live_of(body: dict) -> dict:
    return ok(client.post("/api/montecarlo/live", json=body))


def run_of(body: dict, paths: int) -> dict:
    """Run Monte Carlo for the same deal on the live estimate's paths and seed."""
    request = {**body, "mc": {**body.get("mc", {}), "n": paths}, "seed": live.LIVE_SEED}
    return ok(client.post("/api/montecarlo/run", json=request))["summary"]


def assert_same(prediction: dict, summary: dict):
    for mine, theirs in SUMMARY.items():
        assert prediction[mine] == pytest.approx(summary[theirs], rel=1e-12, abs=1e-15), mine


# ---------------------------------------------------------------------------
# The estimate is the simulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(DEALS))
def test_the_live_estimate_is_the_simulation_on_the_same_paths(name):
    body = DEALS[name]
    got = live_of(body)
    assert got["paths"] == live.LIVE_PATHS and got["hurdle"] == 0.2
    assert_same(got["prediction"], run_of(body, got["paths"]))
    assert got["model"]["engine_version"]


def test_the_deals_differ_so_the_comparison_means_something():
    medians = {name: live_of(body)["prediction"]["irr_p50"] for name, body in DEALS.items()}
    # A thousand times the money, in thousands, is the default deal (the
    # simulation does not read minimum cash: CLAUDE.md finding 13)
    assert medians.pop("thousands") == pytest.approx(medians["the default deal"], rel=1e-9)
    assert len({round(m, 6) for m in medians.values()}) == len(medians)


def test_the_same_request_gets_the_same_answer():
    a, b = live_of({}), live_of({})
    assert a["prediction"] == b["prediction"] and a["median_interval"] == b["median_interval"]


def test_sliders_left_out_are_the_deals_own_assumptions():
    own = {"growth_mean": 5.0, "exit_mean": 10.0, "interest_mean": 6.5, "gross_margin_mean": 40.0,
           "debt_pct": 60.0, "exit_std": 1.5}
    assert live_of({"sliders": own})["prediction"] == live_of({})["prediction"]


# ---------------------------------------------------------------------------
# A slider is an input
# ---------------------------------------------------------------------------
OWN = {"growth_mean": 5.0, "exit_mean": 10.0, "interest_mean": 6.5, "gross_margin_mean": 40.0,
       "debt_pct": 60.0, "exit_std": 1.5}
MOVES = [
    ("growth_mean", 9.0, {"mc": {"growth_mean": 9.0}}),
    ("exit_mean", 12.5, {"mc": {"exit_mean": 12.5}}),
    ("exit_std", 3.0, {"mc": {"exit_std": 3.0}}),
    ("interest_mean", 9.0, {"mc": {"rate_mean": 9.0}}),
    ("gross_margin_mean", 46.0, {"mc": {"gm_mean": 46.0}}),
    ("debt_pct", 72.0, {"deal": {"debt_pct": 72.0}}),
]


@pytest.mark.parametrize("slider, value, as_input", MOVES, ids=[m[0] for m in MOVES])
def test_moving_a_slider_gives_the_run_with_that_input_changed(slider, value, as_input):
    moved = live_of({"sliders": {**OWN, slider: value}})
    assert_same(moved["prediction"], run_of(as_input, moved["paths"]))
    assert moved["prediction"]["irr_p50"] != pytest.approx(live_of({})["prediction"]["irr_p50"], abs=1e-4)


def written_out(**changes) -> list[dict]:
    """The default deal's percentages as a facility list, for the API."""
    deal = DealInputs(**changes)
    return [{k: v for k, v in dataclasses.asdict(t).items() if k != "currency"}
            for t in equivalent_tranches(deal, resolve_config({}))]


def test_the_debt_slider_scales_a_listed_structure():
    """The default deal written out as facilities is the same deal, so at 72%
    debt it must be the percentages deal at 72%: every facility scaled alike."""
    listed = {"deal": {"tranches": written_out()}}
    moved = live_of({**listed, "sliders": {**OWN, "debt_pct": 72.0}})
    assert_same(moved["prediction"], run_of({"deal": {"debt_pct": 72.0}}, moved["paths"]))
    # ... and left where it is, the structure is not touched at all
    assert live_of({**listed, "sliders": OWN})["prediction"] == live_of(listed)["prediction"]


def test_the_rate_slider_moves_floating_references_only():
    """A listed facility centres on its own reference, so the slider moves the
    floating ones by its distance from the rail's rate mean, as a scenario's
    rate stress does: +2.5 points is the run with each floating reference
    2.5 points higher and the rail's mean at the slider. Fixed notes do not
    move."""
    deal = {"tranches": [TERM_LOAN, NOTES]}
    moved = live_of({"deal": deal, "sliders": {**OWN, "interest_mean": 9.0, "debt_pct": 57.0}})
    by_hand = {"tranches": [{**TERM_LOAN, "reference_level": 6.5}, NOTES]}
    assert_same(moved["prediction"], run_of({"deal": by_hand, "mc": {"rate_mean": 9.0}}, moved["paths"]))
    assert moved["prediction"]["irr_p50"] < live_of({"deal": deal})["prediction"]["irr_p50"] - 0.002

    fixed = {"tranches": [NOTES, PIK]}
    share = live.debt_share(MCInputs(), DealInputs(tranches=fixed["tranches"]))
    still = live_of({"deal": fixed, "sliders": {**OWN, "interest_mean": 12.0, "debt_pct": share}})
    assert still["prediction"] == live_of({"deal": fixed})["prediction"]


def test_debt_share_counts_what_is_drawn_against_the_value_the_deal_is_priced_at():
    mc = MCInputs()
    assert live.debt_share(mc, DealInputs(debt_pct=42.0)) == 42.0
    # 420 + 150 drawn, the revolver's 80 undrawn, on 100 x 10
    listed = DealInputs(tranches=[REVOLVER, TERM_LOAN, NOTES])
    assert live.debt_share(mc, listed) == pytest.approx(57.0)
    # Post-IFRS 16 the deal is priced on EBITDA before the 10 of lease cost: 110 x 10
    leased = dataclasses.replace(listed, accounting_standard="ifrs", lease_view="post_ifrs16",
                                 lease_cost=10.0, lease_liability=60.0)
    assert live.debt_share(dataclasses.replace(mc, ebitda=110.0), leased) == pytest.approx(570 / 1100 * 100)


def test_a_structure_with_nothing_drawn_has_nothing_to_scale():
    deal = DealInputs(tranches=[REVOLVER])
    _, moved, _ = live.with_sliders(MCInputs(), deal, dataclasses.replace(live.sliders_of(MCInputs(), deal),
                                                                         debt_pct=50.0))
    assert moved.tranches == deal.tranches


def test_scaling_keeps_a_repayment_schedule_in_proportion():
    loan = spec_from_kind("amortising_term_loan", "A", amount=400.0, reference_level=4.0, margin=3.0,
                          amort_schedule=(40.0, 40.0, 80.0))
    deal = DealInputs(tranches=[loan])
    sliders = dataclasses.replace(live.sliders_of(MCInputs(), deal), debt_pct=20.0)
    _, moved, shift = live.with_sliders(MCInputs(), deal, sliders)
    assert moved.tranches[0].amount == pytest.approx(200.0)
    assert moved.tranches[0].amort_schedule == pytest.approx((20.0, 20.0, 40.0))
    assert shift == 0.0


def test_a_structure_the_deal_cannot_carry_is_refused_as_everywhere_else():
    too_much = {"deal": {"tranches": [{**TERM_LOAN, "amount": 900.0}, {**NOTES, "amount": 400.0}]}}
    resp = client.post("/api/montecarlo/live", json=too_much)
    assert resp.status_code == 422
    assert resp.json()["detail"] == client.post("/api/montecarlo/run", json=too_much).json()["detail"]


# ---------------------------------------------------------------------------
# Sampling error, said
# ---------------------------------------------------------------------------
def test_the_medians_interval_by_hand():
    """A hundred paths, IRRs 0.00 to 0.99: 1.96 x sqrt(100) / 2 = 9.8 ranks
    either side of the middle, so the 40th and the 60th."""
    assert live.median_interval(np.arange(100) / 100) == (0.40, 0.60)
    # Paths without an IRR are not paths the median is taken over
    assert live.median_interval(np.array([np.nan] * 5 + list(np.arange(100) / 100))) == (0.40, 0.60)
    assert all(np.isnan(v) for v in live.median_interval(np.array([np.nan])))


def test_the_interval_holds_the_median_and_the_long_runs_median():
    got = live_of({})
    low, high = got["median_interval"]
    assert low < got["prediction"]["irr_p50"] < high
    assert 0.002 < high - low < 0.01                       # about half a point wide at 8,000 paths
    long_run = ok(client.post("/api/montecarlo/run", json={"mc": {"n": 100_000}, "seed": 3}))["summary"]
    assert low < long_run["median_irr"] < high


# ---------------------------------------------------------------------------
# Lightweight
# ---------------------------------------------------------------------------
def test_paths_shrink_with_the_deals_size_and_never_below_the_floor():
    assert live.live_paths(5, 0, 2) == live.LIVE_PATHS                     # percentages count as two
    assert live.live_paths(10, 10, 1) == live.LIVE_PATHS                   # work 100
    assert live.live_paths(10, 10, 2) == live.LIVE_PATHS                   # work 200: the budget
    assert live.live_paths(10, 10, 4) == live.LIVE_PATHS // 2              # twice the budget
    assert live.live_paths(15, 12, 10) == live.MIN_LIVE_PATHS              # the API's caps
    sizes = [live.live_paths(h, f, 2) for h in range(1, 16) for f in (2, 6, 12)]
    assert min(sizes) >= live.MIN_LIVE_PATHS and max(sizes) == live.LIVE_PATHS


CAPS = {"mc": {"hold": 15}, "settings": {"mc_n_passes": 10},
        "deal": {"hold": 15, "tax_interest_limit": "ebitda_share", "tax_loss_carryforward": True,
                 "tax_minimum_pct": 15.0,
                 "tranches": [{**TERM_LOAN, "name": f"Loan {i}", "amount": 30.0} for i in range(6)]
                 + [{**NOTES, "name": f"Notes {i}", "amount": 30.0} for i in range(5)] + [REVOLVER]}}


def test_a_deal_at_the_caps_runs_on_fewer_paths_and_says_so():
    got = live_of(CAPS)
    assert got["paths"] == live.MIN_LIVE_PATHS
    assert_same(got["prediction"], run_of(CAPS, got["paths"]))


def test_a_live_estimate_at_the_caps_stays_small():
    """The free server has 512 MB and a live estimate holds no simulation
    slot, so several can run beside a full simulation: each must be small."""
    body = CAPS
    mc = MCInputs(hold=15)
    deal = DealInputs(**body["deal"])
    cfg = resolve_config(body["settings"])
    tracemalloc.start()
    try:
        live.live_estimate(mc, deal, cfg)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 24 * 1024 * 1024, f"{peak / 1e6:.1f} MB"


def test_live_needs_no_ml_packages():
    assert ok(client.get("/api/capabilities"))["live"] is True
    live_of({})
    assert not {"torch", "onnxruntime", "sklearn", "joblib"} & set(sys.modules)
    # The network's endpoint is gone with it
    assert client.post("/api/ml/surrogate", json={}).status_code == 404


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
def test_live_estimates_have_their_own_rate_and_are_not_model_runs(monkeypatch):
    assert not limits.is_run_request("POST", limits.LIVE_PATH)
    assert limits.LIVE_PATH not in limits.SIMULATION_PATHS
    monkeypatch.setattr(limits, "USER_LIVE", limits.Rule("user_live_per_minute", "live estimates", "user", 3, 60))
    monkeypatch.setattr(limits, "USER_RUNS", limits.Rule("user_runs_per_minute", "model runs", "user", 1, 60))
    app.dependency_overrides[require_user] = lambda: AuthUser(subject="dragger")
    for _ in range(3):                                     # more than the one model run allowed
        assert client.post("/api/montecarlo/live", json={}).status_code == 200
    resp = client.post("/api/montecarlo/live", json={})
    assert resp.status_code == 429
    assert resp.json()["limit"]["rule"] == "user_live_per_minute"
    assert "Too many live estimates" in resp.json()["detail"]
    # ... and none of them spent the model run
    assert client.post("/api/montecarlo/run", json={"mc": {"n": 1000}}).status_code == 200


def test_invalid_sliders_are_refused():
    assert client.post("/api/montecarlo/live", json={"sliders": {**OWN, "gross_margin_mean": 120}}).status_code == 422
    assert client.post("/api/montecarlo/live", json={"sliders": {"growth_mean": 5.0, "bogus": 1}}).status_code == 422
