"""The model registry and the cards of the models the app loads (PLAN.md 5.1).

"Done when: the surrogate and anomaly detector have cards with per-region
results." These tests re-evaluate both from their committed files, so a
card can't be edited by hand or left behind by a retrained model, and every
trained file in ml/ must be registered. They need the ML packages (CI's
``ml`` job, which fails on a skip).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from ml.evaluation import anomaly, card, surrogate                # noqa: E402
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
        assert m["evaluation"] in {"ml.evaluation.anomaly", "ml.evaluation.surrogate"}


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
    committed = json.loads(card.paths("anomaly_detector")[0].read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(committed))
    changed["registry"]["artifacts"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(cli, "model_card", lambda model_id, base=None: changed)
    assert any("sha256" in p for p in stale("anomaly_detector"))


# ---------------------------------------------------------------------------
# Done when: both models have per-region results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", ["anomaly_detector", "surrogate"])
def test_the_card_has_results_for_every_region(model_id, fresh_cards):
    ev = fresh_cards[model_id]["evaluation"]
    headline = ev["sets"][ev["headline_set"]]
    assert list(headline["by_region"]) == list(REGIONS)
    with_results = [r for r, g in headline["by_region"].items() if g["verdict"] != NOT_ENOUGH]
    assert with_results, f"{model_id} has no region with enough data"
    for g in headline["by_region"].values():
        assert ("model" in g) == (g["verdict"] != NOT_ENOUGH)


def test_the_anomaly_detector_is_tested_out_of_time_and_only_the_us_has_cases(fresh_cards):
    ev = fresh_cards["anomaly_detector"]["evaluation"]
    oot = ev["sets"]["out_of_time"]
    # Deals closed 2008 on: 14 cases, 4 of them distressed
    assert oot["overall"]["cases"] == 14
    assert oot["by_region"]["us"]["cases"] == 14
    assert all(oot["by_region"][r] == {"cases": 0, "verdict": NOT_ENOUGH}
               for r in ("europe", "emerging", "other_developed"))
    assert ev["sets"]["in_sample"]["overall"]["cases"] == 30


def test_the_surrogate_is_tested_on_deals_from_every_region(fresh_cards):
    ev = fresh_cards["surrogate"]["evaluation"]
    every = ev["sets"]["all_regional_deals"]
    deals = surrogate.load_cases()
    assert every["overall"]["cases"] == len(deals)
    for region in REGIONS:
        assert every["by_region"][region]["cases"] == sum(d["region"] == region for d in deals) >= 5


# ---------------------------------------------------------------------------
# The anomaly detector's evaluation
# ---------------------------------------------------------------------------
def test_refitting_on_every_deal_reproduces_the_committed_detector():
    """fit_detector is train_detector's fit: the harness refits the very model
    the app loads, not a look-alike."""
    from ml.anomaly_detector import fit_detector, load_detector
    detector, scaler, nn_model, raw_df, n = fit_detector()
    committed = load_detector()
    x = np.array([[c.features[k] for k in anomaly.FEATURES] for c in anomaly.cases()])
    assert n == 530
    assert np.allclose(scaler.transform(x), committed[1].transform(x), rtol=0, atol=1e-12)
    assert np.allclose(detector.score_samples(scaler.transform(x)),
                       committed[0].score_samples(committed[1].transform(x)), rtol=0, atol=1e-12)


def test_the_out_of_time_set_scores_each_deal_with_a_model_fitted_only_on_older_deals(monkeypatch):
    fitted_on = []
    real_fit = anomaly.fit

    def spy(train):
        fitted_on.append(max(c.year for c in train))
        return real_fit(train)
    monkeypatch.setattr(anomaly, "fit", spy)
    from ml.evaluation.harness import walk_forward
    out = walk_forward(anomaly.cases(), anomaly.CUTOFFS, anomaly.fit, anomaly.predict)
    assert fitted_on == [2007, 2009]
    assert sorted({c.year for c, _ in out}) == [2008, 2009, 2010, 2011, 2013, 2014, 2016]


def test_the_baseline_is_leverage_against_the_supervisory_limit():
    deal = next(c for c in anomaly.cases() if c.id == "Hilton 2007")
    assert anomaly.baseline(deal) == {"score": 14.6, "flag": True}
    assert anomaly.baseline(next(c for c in anomaly.cases() if c.id == "Dell 2013"))["flag"] is False


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
def test_training_the_anomaly_detector_gives_the_committed_files_card(tmp_path: Path, fresh_cards):
    """Deterministic training: the files trained into a scratch directory
    evaluate to the committed card, and nothing in the repository changes."""
    before = {a: card.sha256(card.ROOT / a) for a in card.entry("anomaly_detector")["artifacts"]}
    new = train("anomaly_detector", tmp_path, samples=0, epochs=0)
    assert {a: card.sha256(card.ROOT / a) for a in before} == before
    assert (tmp_path / "cards" / "anomaly_detector.json").exists()
    assert (tmp_path / "docs" / "anomaly_detector.md").exists()
    assert new["evaluation"] == fresh_cards["anomaly_detector"]["evaluation"]
