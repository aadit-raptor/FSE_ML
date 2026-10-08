"""The ML evaluation harness, the card template and the CI gate (PLAN.md 5.1).

None of this needs the ML packages: the harness is plain Python over cases,
the gate is standard library. tests/test_model_cards.py evaluates the real
models with them.
"""
from __future__ import annotations

import copy
import json

import pytest

from ml.evaluation import card
from ml.evaluation.__main__ import differences
from ml.evaluation.harness import (BEATS, DOES_NOT_BEAT, MIN_CASES, NOT_ENOUGH, REGIONS, Case, Metric, auc,
                                   evaluate_set, mean_abs, summarize, walk_forward)
from ops import model_gate


def case(i, year, region="us", truth=0.0):
    return Case(id=f"c{i}", region=region, year=year, features={"x": float(i)}, truth=truth)


ERROR = Metric("error", "lower", lambda cs, ps: mean_abs([p - c.truth for c, p in zip(cs, ps)]), 0.01, "error")


# ---------------------------------------------------------------------------
# Time-based splits
# ---------------------------------------------------------------------------
def test_walk_forward_never_predicts_with_a_model_that_saw_the_case_or_anything_later():
    cases = [case(i, y) for i, y in enumerate([2001, 2003, 2005, 2005, 2007, 2008, 2010, None])]
    seen = []

    def fit(train):
        seen.append(sorted(c.year for c in train))
        return max(c.year for c in train)

    out = walk_forward(cases, [2005, 2008], fit, lambda newest, c: newest)
    # Fold 1 trains on 2001-2003 and tests 2005-2007; fold 2 trains on everything
    # before 2008 and tests 2008 on. 2001, 2003 and the undated case are never tested.
    assert seen == [[2001, 2003], [2001, 2003, 2005, 2005, 2007]]
    assert [(c.year, newest) for c, newest in out] == [(2005, 2003), (2005, 2003), (2007, 2003),
                                                       (2008, 2007), (2010, 2007)]
    assert all(newest < c.year for c, newest in out)


def test_walk_forward_skips_a_fold_with_nothing_to_train_on_or_test():
    cases = [case(0, 2010), case(1, 2011)]
    fits = []
    out = walk_forward(cases, [2000, 2011, 2020], lambda t: fits.append(len(t)) or 0, lambda m, c: m)
    assert fits == [1] and [c.year for c, _ in out] == [2011]


def test_walk_forward_refuses_cutoffs_out_of_order():
    with pytest.raises(ValueError):
        walk_forward([], [2010, 2005], lambda t: 0, lambda m, c: 0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_auc_by_hand():
    # Positives 0.9 and 0.3, negatives 0.8 and 0.3: 0.9 beats both (2), 0.3
    # loses to 0.8 (0) and ties 0.3 (0.5) -> 2.5 of 4 pairs
    assert auc([0.9, 0.8, 0.3, 0.3], [True, False, True, False]) == 0.625
    assert auc([1, 2], [True, True]) is None
    assert auc([1, 2, 3], [False, True, True]) == 1.0


def test_mean_abs():
    assert mean_abs([1.0, -3.0]) == 2.0 and mean_abs([]) is None


# ---------------------------------------------------------------------------
# Baseline comparison and regions
# ---------------------------------------------------------------------------
def test_a_group_beats_the_baseline_only_when_strictly_better():
    cases = [case(i, None) for i in range(MIN_CASES)]
    better = summarize(cases, [0.1] * MIN_CASES, [0.2] * MIN_CASES, [ERROR], "error")
    assert better == {"cases": MIN_CASES, "model": {"error": 0.1}, "baseline": {"error": 0.2}, "verdict": BEATS}
    assert summarize(cases, [0.2] * MIN_CASES, [0.2] * MIN_CASES, [ERROR], "error")["verdict"] == DOES_NOT_BEAT
    assert summarize(cases, [0.3] * MIN_CASES, [0.2] * MIN_CASES, [ERROR], "error")["verdict"] == DOES_NOT_BEAT
    # Higher is better: a tie (the risk score's AUC 1.0 against leverage's) does not beat it
    score = Metric("score", "higher", lambda cs, ps: sum(ps) / len(ps), 0.0, "score")
    assert summarize(cases, [1.0] * MIN_CASES, [1.0] * MIN_CASES, [score], "score")["verdict"] == DOES_NOT_BEAT
    assert summarize(cases, [1.0] * MIN_CASES, [0.9] * MIN_CASES, [score], "score")["verdict"] == BEATS


def test_too_few_cases_is_not_enough_data():
    few = [case(i, None) for i in range(MIN_CASES - 1)]
    assert summarize(few, [0.0] * len(few), [1.0] * len(few), [ERROR], "error") == {
        "cases": MIN_CASES - 1, "verdict": NOT_ENOUGH}


def test_a_headline_that_cannot_be_computed_is_not_enough_data():
    one_class = [case(i, None, truth=False) for i in range(MIN_CASES)]
    area = Metric("auc", "higher", lambda cs, ps: auc(ps, [c.truth for c in cs]), 0.0, "auc")
    assert summarize(one_class, [1.0] * MIN_CASES, [0.0] * MIN_CASES, [area], "auc")["verdict"] == NOT_ENOUGH


def test_every_region_is_listed_and_judged_on_its_own_cases():
    rows = [(case(i, None, "us"), 0.0, 1.0) for i in range(6)] + [
        (case(10 + i, None, "europe"), 1.0, 0.5) for i in range(5)] + [(case(20, None, "emerging"), 0.0, 1.0)]
    out = evaluate_set(rows, [ERROR], "error")
    assert set(out["by_region"]) == set(REGIONS)
    assert out["overall"]["cases"] == 12
    assert out["by_region"]["us"]["verdict"] == BEATS
    assert out["by_region"]["europe"]["verdict"] == DOES_NOT_BEAT
    assert out["by_region"]["emerging"] == {"cases": 1, "verdict": NOT_ENOUGH}
    assert out["by_region"]["other_developed"] == {"cases": 0, "verdict": NOT_ENOUGH}


# ---------------------------------------------------------------------------
# The gate: CI fails if a model gets worse
# ---------------------------------------------------------------------------
def a_card(error=0.5, region_error=0.4, tolerance=0.02):
    group = {"cases": 9, "model": {"error": error}, "baseline": {"error": 0.9}, "verdict": BEATS}
    region = {"cases": 6, "model": {"error": region_error}, "baseline": {"error": 0.9}, "verdict": BEATS}
    return {"id": "m", "evaluation": {
        "metrics": [{"id": "error", "better": "lower", "tolerance": tolerance, "description": "e"}],
        "sets": {"test": {"overall": group, "by_region": {"us": region,
                                                          "europe": {"cases": 0, "verdict": NOT_ENOUGH}}}}}}


def test_the_gate_fails_a_card_worse_than_the_base_beyond_its_tolerance():
    assert model_gate.compare("m", a_card(), a_card()) == []
    assert model_gate.compare("m", a_card(), a_card(error=0.51)) == []          # within 0.02
    assert model_gate.compare("m", a_card(), a_card(error=0.3)) == []           # better
    assert model_gate.compare("m", a_card(), a_card(error=0.53)) == [
        "m test overall error: 0.5 -> 0.53 (lower is better, tolerance 0.02)"]
    # A region getting worse fails even when the overall figure improves
    assert model_gate.compare("m", a_card(), a_card(error=0.3, region_error=0.6)) == [
        "m test us error: 0.4 -> 0.6 (lower is better, tolerance 0.02)"]


def test_the_gate_uses_the_base_cards_tolerance():
    loose = a_card(error=0.6, tolerance=1.0)
    assert model_gate.compare("m", a_card(), loose) != []


def test_the_gate_fails_a_region_that_loses_its_results():
    new = a_card()
    new["evaluation"]["sets"]["test"]["by_region"]["us"] = {"cases": 2, "verdict": NOT_ENOUGH}
    assert model_gate.compare("m", a_card(), new) == ["m test us: had results on 6 cases, now none"]
    gone = a_card()
    del gone["evaluation"]["sets"]["test"]
    assert model_gate.compare("m", a_card(), gone) == ["m test: set removed"]


def test_higher_is_better_metrics_fail_when_they_fall():
    assert model_gate.worse("higher", 0.9, 0.88, 0.005) and not model_gate.worse("higher", 0.9, 0.897, 0.005)


def test_the_gate_reads_every_registered_card_against_the_base_ref():
    reg = lambda *ids: json.dumps({"models": [{"id": i} for i in ids]})
    base = {"ml/registry.json": reg("m", "old"), "ml/cards/m.json": json.dumps(a_card()),
            "ml/cards/old.json": json.dumps(a_card())}
    head = {"ml/registry.json": reg("m", "new"), "ml/cards/m.json": json.dumps(a_card(error=0.9)),
            "ml/cards/new.json": json.dumps(a_card())}

    def read(path):
        if path not in head:
            raise FileNotFoundError(path)
        return head[path]
    problems, notes = model_gate.gate("origin/main", show=lambda ref, p: base.get(p), read=read)
    assert problems == ["m test overall error: 0.5 -> 0.9 (lower is better, tolerance 0.02)"]
    assert notes == ["old: removed from the registry", "new: new model, nothing to compare with"]
    del head["ml/cards/m.json"]
    assert model_gate.gate("origin/main", show=lambda ref, p: base.get(p), read=read)[0] == [
        "m: registered but has no card"]


def test_the_gate_passes_the_committed_cards_against_themselves():
    same = lambda ref, path: (card.ROOT / path).read_text(encoding="utf-8")
    assert model_gate.gate("HEAD", show=same) == ([], [])


# ---------------------------------------------------------------------------
# The card template
# ---------------------------------------------------------------------------
def test_every_committed_card_renders_to_its_markdown_with_every_region():
    for model in card.registry()["models"]:
        json_path, md_path = card.paths(model["id"])
        c = json.loads(json_path.read_text(encoding="utf-8"))
        text = md_path.read_text(encoding="utf-8")
        assert text == card.render(c)
        for name in card.REGION_NAMES.values():
            assert f"| {name} |" in text
        assert "## Limitations" in text and "## How it is tested" in text


def test_the_template_says_not_enough_data_where_a_group_is_thin():
    c = {"id": "m", "name": "M", "module": "m.py", "used_by": "u", "estimates": "e", "min_cases": MIN_CASES,
         "training": {"data": "d", "command": "c", "deterministic": True}, "limitations": ["l"],
         "registry": {"artifacts": [{"path": "m.pkl", "bytes": 1200, "sha256": "ab" * 32}]},
         "evaluation": {**a_card()["evaluation"], "headline_set": "test", "headline_metric": "error",
                        "split": {"kind": "none", "why": "no dates"},
                        "baseline": {"id": "b", "description": "A constant"}}}
    c["evaluation"]["sets"]["test"]["description"] = "Held out"
    text = card.render(c)
    assert "| Europe | 0 | not enough data | – |" in text
    assert "| United States | 6 | beats the baseline | 0.4 / 0.9 |" in text
    assert "No time split: no dates." in text and "| `m.pkl` | 1,200 |" in text


def test_card_differences_allow_platform_noise_only():
    a = {"x": 1.0, "y": [1, "a"], "z": True}
    assert differences(a, copy.deepcopy(a)) == []
    assert differences(a, {**a, "x": 1.0004}) == []
    assert differences(a, {**a, "x": 1.01}) == [".x: 1.0 -> 1.01"]
    assert differences(a, {**a, "z": False}) == [".z: True -> False"]
    assert differences(a, {**a, "w": 1}) == [".w: only in the new card"]
