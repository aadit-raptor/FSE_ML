"""The model registry and the cards of the models the app loads (PLAN.md 5.1).

"Done when: the surrogate and anomaly detector have cards with per-region
results." The anomaly detector became the deal risk score in PLAN.md 5.2.
These tests re-evaluate both from their committed files, so a
card can't be edited by hand or left behind by a retrained model, and every
trained file in ml/ must be registered. They need the ML packages (CI's
``ml`` job, which fails on a skip).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from ml.evaluation import card, deal_risk, surrogate              # noqa: E402
from ml.evaluation.__main__ import model_card, stale, train      # noqa: E402
from ml.evaluation.harness import NOT_ENOUGH, REGIONS              # noqa: E402

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
        assert m["evaluation"] in {"ml.evaluation.deal_risk", "ml.evaluation.surrogate"}


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


def test_a_changed_model_file_makes_its_card_stale(tmp_path, monkeypatch):
    """Retraining without re-evaluating fails CI: the hash no longer matches."""
    import ml.evaluation.__main__ as cli
    committed = json.loads(card.paths("surrogate")[0].read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(committed))
    changed["registry"]["artifacts"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(cli, "model_card", lambda model_id, base=None: changed)
    assert any("sha256" in p for p in stale("surrogate"))


# ---------------------------------------------------------------------------
# Done when: both models have per-region results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", ["deal_risk", "surrogate"])
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


def test_the_surrogate_is_tested_on_deals_from_every_region(fresh_cards):
    ev = fresh_cards["surrogate"]["evaluation"]
    every = ev["sets"]["all_regional_deals"]
    deals = surrogate.load_cases()
    assert every["overall"]["cases"] == len(deals)
    for region in REGIONS:
        assert every["by_region"][region]["cases"] == sum(d["region"] == region for d in deals) >= 5


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
# The surrogate's evaluation
# ---------------------------------------------------------------------------
def test_in_range_is_the_training_ranges():
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
def test_the_deal_risk_score_has_nothing_to_train(tmp_path: Path):
    with pytest.raises(SystemExit, match="nothing to train"):
        train("deal_risk", tmp_path, samples=0, epochs=0)
