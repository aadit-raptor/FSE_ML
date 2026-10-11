"""The model registry and the cards of the models the app loads (PLAN.md 5.1).

"Done when: the surrogate and anomaly detector have cards with per-region
results." The anomaly detector became the deal risk score in PLAN.md 5.2
and the surrogate network became the simulation on a small set of paths in
5.8 (its card keeps the id). These tests re-evaluate every model, so a card
can't be edited by hand or left behind by a changed model, and every
trained file in ml/ must be registered. No model needs ML packages any more.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import live
from ml.evaluation import card, deal_risk, surrogate
from ml.evaluation.__main__ import differences, model_card, stale, train
from ml.evaluation.harness import BEATS, NOT_ENOUGH, REGIONS

ARTIFACT_SUFFIXES = {".pkl", ".pt", ".onnx", ".joblib"}


@pytest.fixture(scope="module")
def fresh_cards() -> dict:
    """Each registered model's card as its committed files give it today."""
    return {m["id"]: model_card(m["id"]) for m in card.registry()["models"]}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_every_trained_file_in_ml_is_registered_and_every_registered_file_exists():
    registered = {a for m in card.registry()["models"] for a in m["artifacts"]}
    on_disk = {p.relative_to(card.ROOT).as_posix() for p in (card.ROOT / "ml").rglob("*")
               if p.suffix in ARTIFACT_SUFFIXES and "__pycache__" not in p.parts}
    assert on_disk <= registered, f"unregistered model files: {sorted(on_disk - registered)}"
    assert all((card.ROOT / a).exists() for a in registered)


def test_each_registered_model_names_its_evaluation_and_has_a_card():
    for m in card.registry()["models"]:
        json_path, md_path = card.paths(m["id"])
        assert json_path.exists() and md_path.exists()
        assert m["evaluation"] in {"ml.evaluation.deal_risk", "ml.evaluation.distress", "ml.evaluation.growth",
                                   "ml.evaluation.multiples", "ml.evaluation.regime",
                                   "ml.evaluation.surrogate"}


# ---------------------------------------------------------------------------
# The cards are what the committed files give
# ---------------------------------------------------------------------------
def test_every_card_is_current(fresh_cards, monkeypatch):
    import ml.evaluation.__main__ as cli
    monkeypatch.setattr(cli, "model_card", lambda model_id, base=None: fresh_cards[model_id])
    for model_id in fresh_cards:
        assert stale(model_id) == [], (
            f"{model_id}'s card is not current: run python -m ml.evaluation evaluate and commit the cards")


def test_a_card_records_the_hash_of_every_file_it_evaluated(fresh_cards):
    for model_id, c in fresh_cards.items():
        files = c["registry"]["artifacts"]
        assert [f["path"] for f in files] == card.entry(model_id)["artifacts"]
        for f in files:
            assert f["sha256"] == card.sha256(card.ROOT / f["path"])


def test_a_changed_model_file_makes_its_card_stale():
    """Retraining without re-evaluating fails CI: the hash no longer matches.
    (No registered model has a trained file today; the rule stays for the
    next one that does.)"""
    committed = {"registry": {"artifacts": [{"path": "ml/x/model.onnx", "bytes": 10, "sha256": "a" * 64}]}}
    changed = {"registry": {"artifacts": [{"path": "ml/x/model.onnx", "bytes": 10, "sha256": "0" * 64}]}}
    assert any("sha256" in p for p in differences(committed, changed))
    assert differences(committed, committed) == []


def test_a_changed_result_makes_its_card_stale(fresh_cards, monkeypatch):
    import ml.evaluation.__main__ as cli
    changed = json.loads(json.dumps(fresh_cards["surrogate"]))
    changed["evaluation"]["sets"]["varied_deals"]["overall"]["model"]["median_irr_error_pp"] += 0.01
    monkeypatch.setattr(cli, "model_card", lambda model_id, base=None: changed)
    assert any("median_irr_error_pp" in p for p in stale("surrogate"))


# ---------------------------------------------------------------------------
# Done when: both models have per-region results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", ["deal_risk", "distress", "surrogate"])
def test_the_card_has_results_for_every_region(model_id, fresh_cards):
    ev = fresh_cards[model_id]["evaluation"]
    headline = ev["sets"][ev["headline_set"]]
    assert list(headline["by_region"]) == list(REGIONS)
    with_results = [r for r, g in headline["by_region"].items() if g["verdict"] != NOT_ENOUGH]
    assert with_results, f"{model_id} has no region with enough data"
    for g in headline["by_region"].values():
        assert ("model" in g) == (g["verdict"] != NOT_ENOUGH)


def test_the_deal_risk_score_is_tested_on_the_reference_transactions_and_only_the_us_has_enough(fresh_cards):
    ev = fresh_cards["deal_risk"]["evaluation"]
    every = ev["sets"]["reference_deals"]
    # Ten transactions; D&B (15 US information services companies) and
    # Masonite (Canadian building materials) have no peer group in their region
    assert every["overall"]["cases"] == 8
    assert every["by_region"]["us"]["cases"] == 5
    assert all(every["by_region"][r] == {"cases": 1, "verdict": NOT_ENOUGH}
               for r in ("europe", "emerging", "other_developed"))
    assert "dun-bradstreet-2019, masonite-2005" in every["description"]


def test_the_live_sliders_are_tested_on_deals_from_every_region(fresh_cards):
    ev = fresh_cards["surrogate"]["evaluation"]
    for name, deals in (("all_regional_deals", surrogate.load_cases()), ("varied_deals", surrogate.load_varied())):
        every = ev["sets"][name]
        assert every["overall"]["cases"] == len(deals) == 230
        for region in REGIONS:
            assert every["by_region"][region]["cases"] == sum(d["region"] == region for d in deals) >= 5


def test_the_median_irr_stays_within_half_a_point_on_varied_deals_in_every_region(fresh_cards):
    """PLAN.md 5.8's "done when": median-IRR error under 0.5 points on varied
    held-out deals from several regions, and better than the cruder estimate."""
    ev = fresh_cards["surrogate"]["evaluation"]
    assert ev["headline_set"] == "varied_deals" and ev["headline_metric"] == "median_irr_error_pp"
    varied = ev["sets"]["varied_deals"]
    for group in [varied["overall"], *varied["by_region"].values()]:
        assert group["verdict"] == BEATS
        assert group["model"]["median_irr_error_pp"] < surrogate.MEDIAN_ERROR_LIMIT_PP == 0.5


def test_the_varied_deals_vary_what_the_task_names():
    """Multiples, holds, fees, debt types, rates and tax settings, in every region."""
    from tests.ml_regional_deals import STRUCTURES
    deals = surrogate.load_varied()
    for region in REGIONS:
        mine = [d for d in deals if d["region"] == region]
        assert {d["structure"] for d in mine} == set(STRUCTURES)
        assert len({d["mc"]["hold"] for d in mine}) >= 4
        assert len({d["mc"]["entry_mult"] for d in mine}) >= 8
        assert len({d["settings"]["tx_fee_pct"] for d in mine}) >= 8
        assert len({d["mc"]["rate_mean"] for d in mine}) >= 1
    assert len({d["mc"]["rate_mean"] for d in deals}) >= 15
    ruled = [d for d in deals if d["deal"].get("tax_preset")]
    assert len({d["deal"]["tax_preset"] for d in ruled}) >= 8 and len(ruled) < len(deals)
    assert any(d["deal"].get("lease_cost") for d in deals)
    kinds = {t["kind"] for d in deals for t in d["deal"].get("tranches", [])}
    assert kinds >= {"unitranche", "revolver", "amortising_term_loan", "institutional_term_loan", "second_lien",
                     "senior_notes", "pik_notes", "vendor_loan", "shareholder_loan"}


def test_a_varied_deal_is_estimated_as_the_app_estimates_it():
    """The card asks core.live.live_estimate, the endpoint's own function,
    and the API accepts every varied deal as written."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    for deal in surrogate.load_varied()[::23]:
        body = {k: deal[k] for k in ("mc", "deal", "settings")}
        resp = client.post("/api/montecarlo/live", json=body)
        assert resp.status_code == 200, resp.text
        got, mine = resp.json(), surrogate.live_on_deal(deal)
        assert got["prediction"]["irr_p50"] == mine["p50"] and got["prediction"]["p_wipeout"] == mine["wipeout"]
        assert got["paths"] == live.live_paths(deal["mc"]["hold"], len(deal["deal"].get("tranches", [])), 2)


def test_truth_is_the_same_deal_on_more_paths_with_its_own_draws():
    deal = surrogate.load_varied()[1]
    assert deal["structure"] != "percentages"
    small = surrogate.simulate_deal(deal, live.LIVE_PATHS, live.LIVE_SEED)
    assert small == surrogate.live_on_deal(deal)                    # same paths, same seed: the same answer
    other = surrogate.simulate_deal(deal, live.LIVE_PATHS, 1)
    assert other != small and abs(other["p50"] - small["p50"]) < 0.02


def test_the_network_deals_estimate_uses_the_live_paths_and_seed():
    f = surrogate.load_cases()[0]["features"]
    assert surrogate.live_on_network_deal(f) == surrogate.simulate(f, live.LIVE_PATHS, live.LIVE_SEED)


# ---------------------------------------------------------------------------
# The deal risk score's evaluation
# ---------------------------------------------------------------------------
def test_every_reference_transaction_has_an_industry_the_averages_know():
    tables = deal_risk.peer_tables()
    keys = {c.id for c in deal_risk.cases()}
    assert keys == set(deal_risk.INDUSTRY)
    assert set(deal_risk.INDUSTRY.values()) <= set(tables["debt.global"].rows)


def test_a_case_is_scored_as_the_app_scores_it():
    """The evaluation asks ml.anomaly_detector.compare, the app's own function."""
    from ml.anomaly_detector import DealShape, compare
    tables = deal_risk.peer_tables()
    toys = next(c for c in deal_risk.cases() if c.id == "toys-r-us-2005")
    found = compare(DealShape("US", "retail_special_lines", 6.69, 10.03), tables)
    assert deal_risk.predict(tables, toys) == {"score": found["score"], "flag": found["unusual"]}
    assert found["score"] > 0


def test_the_baseline_is_leverage_against_the_supervisory_limit():
    by_id = {c.id: c for c in deal_risk.cases()}
    assert deal_risk.baseline(by_id["toys-r-us-2005"]) == {"score": 6.69, "flag": True}
    assert deal_risk.baseline(by_id["hca-2006"])["flag"] is False


def test_the_peer_tables_file_is_current():
    from tests import ml_peer_tables
    assert deal_risk.PEER_TABLES.read_text(encoding="utf-8") == ml_peer_tables.text(), (
        "ml/evaluation/data/peer_tables.json is stale: run python -m tests.ml_peer_tables")


# ---------------------------------------------------------------------------
# The live sliders' evaluation
# ---------------------------------------------------------------------------
def test_in_range_is_the_retired_networks_training_ranges():
    inside = {k: (lo + hi) / 2 for k, lo, hi in zip(surrogate.X_COLS, surrogate.L_BOUNDS, surrogate.U_BOUNDS)}
    assert surrogate.in_range(inside)
    assert not surrogate.in_range({**inside, "exit_std": 4.01})
    assert surrogate.in_range({**inside, "debt_pct": 0.25})


def test_regional_features_come_from_the_sourced_settings_in_fractions():
    risk = {"mc_growth_mean": 11.51, "mc_growth_std": 3.88, "mc_exit_mean": 26.85, "mc_exit_std": 8.57,
            "mc_rate_mean": 5.8, "mc_gm_mean": 44.51, "mc_gm_std": 0.93}
    start = {"da": 2.57, "capex": 5.7, "nwc": 2.75}
    f = surrogate.features_from(risk, start, 60.0)
    assert f == {"growth_mean": 0.1151, "growth_std": 0.0388, "exit_mean": 26.85, "exit_std": 8.57,
                 "interest_mean": 0.058, "gross_margin_mean": 0.4451, "gross_margin_std": 0.0093,
                 "da_pct": 0.0257, "capex_pct": 0.057, "nwc_pct": 0.0275, "debt_pct": 0.6}
    assert surrogate.features_from({**risk, "mc_exit_std": None}, start, 60.0) is None


def test_the_regional_deals_file_is_current():
    from tests import ml_regional_deals
    assert surrogate.REGIONAL_DEALS.read_text(encoding="utf-8") == ml_regional_deals.text(), (
        "ml/evaluation/data/regional_deals.json is stale: run python -m tests.ml_regional_deals")


def test_truth_and_baseline_are_the_simulation_on_more_and_fewer_paths():
    f = surrogate.load_cases()[0]["features"]
    a = surrogate.simulate(f, surrogate.BASELINE_PATHS, 1)
    assert a == surrogate.simulate(f, surrogate.BASELINE_PATHS, 1)          # seeded
    assert set(a) == {"p5", "p50", "p95", "wipeout"} and a["p5"] <= a["p50"] <= a["p95"]


# ---------------------------------------------------------------------------
# Training (what ml.yml's train job runs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", ["deal_risk", "surrogate"])
def test_a_model_with_nothing_to_train_gets_its_card_written_where_the_workflow_reads_it(
        model_id, tmp_path: Path, fresh_cards, monkeypatch, capsys):
    """ml.yml's train job (which nobody but the owner can edit) still runs
    ``train surrogate --out DIR --samples N --epochs N`` and then reads
    DIR/docs/surrogate.md: with nothing to train that must succeed."""
    import ml.evaluation.__main__ as cli
    monkeypatch.setattr(cli, "model_card", lambda i, base=None: fresh_cards[i])
    before = card.paths(model_id)[0].read_bytes()
    assert cli.main(["train", model_id, "--out", str(tmp_path), "--samples", "300", "--epochs", "1"]) == 0
    assert "nothing to train" in capsys.readouterr().out
    stored = json.loads(card.to_json(fresh_cards[model_id]))        # rendered from the stored card, keys sorted
    assert (tmp_path / "docs" / f"{model_id}.md").read_text(encoding="utf-8") == card.render(stored)
    assert json.loads((tmp_path / "cards" / f"{model_id}.json").read_text(encoding="utf-8"))["id"] == model_id
    assert card.paths(model_id)[0].read_bytes() == before


def test_no_registered_model_has_anything_to_train():
    for m in card.registry()["models"]:
        assert m["train"] is None and m["artifacts"] == []
    assert train.__doc__ and "nothing to train" in train.__doc__
