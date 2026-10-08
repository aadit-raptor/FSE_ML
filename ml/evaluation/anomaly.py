"""The deal risk score's evaluation (ml/anomaly_detector.py; PLAN.md 5.1).

Question: does the risk score rank the deals that ended in distress above
the ones that didn't, and does its flag catch them?

- **Cases**: the 30 real deals the detector learns from
  (``HISTORICAL_DEALS``), each dated by the year in its name and placed in
  its buyer's region. Every one is a US buyout, so every other region is
  ``not_enough_data`` until PLAN.md 5.2 rebuilds the score on global data.
- **Out of time** (the headline): ``walk_forward`` over ``CUTOFFS`` -- the
  detector is refitted (``fit_detector``) on the deals closed before each
  cutoff and scores the deals closed from it to the next, so no deal is
  scored by a model that saw it or anything later.
- **In sample**: the committed model scores all 30, which it learned from.
  Reported so the gap to out-of-time shows; never the headline.
- **Baseline**: leverage alone -- debt / EBITDA as the score, and a flag
  above the supervisory 6.0x that the deal's risk warnings use
  (``core.risk_sources.LEVERAGE_GUIDANCE_X``, as warning ``leverage_above_guidance``).
"""
from __future__ import annotations

from typing import Optional, Sequence

from core.risk_sources import LEVERAGE_GUIDANCE_X
from library.base_rates import sp_region
from ml.anomaly_detector import BASE, HISTORICAL_DEALS, assess, deal_records, fit_detector, load_detector
from ml.evaluation.harness import Case, Metric, auc, evaluate_set, metric_specs, share, walk_forward

MODEL_ID = "anomaly_detector"
# Test years start where the list has deals on both sides: closed before 2008
# (16 deals, six distressed), 2008-2009 (five), 2010 on (nine)
CUTOFFS = (2008, 2010)
FEATURES = ("entry_mult", "leverage", "growth", "margin", "interest")
# Each deal's buyer's home: every deal in HISTORICAL_DEALS is a US buyout
COUNTRY = "US"
HEADLINE_SET = "out_of_time"


def cases() -> list[Case]:
    """The 30 real deals as cases; truth = ended in distress."""
    out = []
    for row in HISTORICAL_DEALS:
        *values, success, name = row
        year = int(name.rsplit(" ", 1)[-1])
        out.append(Case(id=name, region=sp_region(COUNTRY), year=year,
                        features=dict(zip(FEATURES, values)), truth=not bool(success)))
    return out


def _row(case: Case) -> list:
    f = case.features
    return [*(f[k] for k in FEATURES), 0 if case.truth else 1, case.id]


def fit(train: Sequence[Case]) -> tuple:
    """The detector refitted on ``train`` only (the harness's ``fit``)."""
    rows = [_row(c) for c in train]
    detector, scaler, nn_model, raw_df, _ = fit_detector(rows)
    return detector, scaler, nn_model, deal_records(raw_df)


def predict(models: tuple, case: Case) -> dict:
    """The risk score and flag the app would show for ``case``."""
    f = case.features
    r = assess(models, f["entry_mult"], f["leverage"], f["growth"], f["margin"], f["interest"])
    return {"score": r.risk_score, "flag": bool(r.is_anomalous)}


def baseline(case: Case) -> dict:
    """Leverage alone."""
    lev = case.features["leverage"]
    return {"score": lev, "flag": lev > LEVERAGE_GUIDANCE_X}


def _auc(cases, preds) -> Optional[float]:
    return auc([p["score"] for p in preds], [c.truth for c in cases])


def _caught(cases, preds) -> Optional[float]:
    return share([p["flag"] for c, p in zip(cases, preds) if c.truth])


def _false_alarms(cases, preds) -> Optional[float]:
    return share([p["flag"] for c, p in zip(cases, preds) if not c.truth])


METRICS = (
    Metric("auc", "higher", _auc, 0.005,
           "Chance a distressed deal scores above a deal that wasn't (0.5 = no better than chance)"),
    Metric("distress_flagged", "higher", _caught, 0.005, "Share of distressed deals flagged"),
    Metric("false_alarms", "lower", _false_alarms, 0.005, "Share of deals without distress flagged"),
)
HEADLINE = "auc"


def evaluate(base: str = BASE) -> dict:
    """Both sets for the card: out of time (refitted per fold) and in sample
    (the trained files in ``base``)."""
    all_cases = cases()
    oot = walk_forward(all_cases, CUTOFFS, fit, predict)
    trained = load_detector(base)
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "walk_forward", "cutoffs": list(CUTOFFS), "dated_by": "year the deal closed"},
        "baseline": {"id": "leverage", "description":
                     f"Debt / EBITDA as the score; flagged above {LEVERAGE_GUIDANCE_X:g}x (the supervisory "
                     "leverage limit the deal's risk warnings use)"},
        "sets": {
            "out_of_time": {
                "description": "Refitted on deals closed before each cutoff; scores the deals closed after it",
                **evaluate_set([(c, p, baseline(c)) for c, p in oot], METRICS, HEADLINE)},
            "in_sample": {
                "description": "The committed model on the 30 deals it learned from (not a test)",
                **evaluate_set([(c, predict(trained, c), baseline(c)) for c in all_cases], METRICS, HEADLINE)},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Deal risk score (anomaly detector)",
    "module": "ml/anomaly_detector.py",
    "used_by": "Deal -> Inputs risk score (POST /api/ml/deal-risk)",
    "estimates": "A 1-10 risk score and an unusual-deal flag from entry multiple, leverage, growth, "
                 "EBITDA margin and interest rate",
    "training": {
        "data": "30 hand-entered US LBOs closed 1989-2016 (20 without distress, 10 distressed), plus 500 "
                "synthetic deals jittered around the 20; the figures are unsourced approximations",
        "command": "python -m ml.evaluation train anomaly_detector",
        "deterministic": True,
    },
    "limitations": [
        "Every training deal is a US buyout: Europe, emerging markets and other developed markets have "
        "no cases, so the card shows not enough data there.",
        "The deal figures were typed in without sources; PLAN.md 5.2 rebuilds the score on sourced, "
        "regional market data.",
        "Fourteen out-of-time cases, four of them distressed: each case moves the statistics by several "
        "points.",
        "The score adds fixed rule-of-thumb terms (leverage, multiple, coverage) to the forest's anomaly "
        "severity; the card evaluates the score as the app shows it.",
    ],
}
