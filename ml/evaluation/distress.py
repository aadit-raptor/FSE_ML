"""The distress predictor's evaluation (ml/distress_model.py; PLAN.md 5.1, 5.3).

Two questions, two sets:

- **reference_deals** (the headline, which decides where the app shows the
  probabilities): does the predictor's chance of default over the hold rank
  the sourced buyouts that ended in distress above the ones that didn't,
  better than the deal's year-one default risk
  (``core.risk_warnings.credit_view``: coverage alone, S&P's global table,
  held flat)? Each repository transaction (``library/reference_deals.json``,
  every figure read from a filing) is entered with its filed EBITDA, multiple
  and debt share and everything else at the defaults, as the validation
  report's default check does (``validation.cases.predicted_default``), in
  its own country. Truth: distress at any time after closing (a missed
  payment, a bankruptcy or a restructuring), as the deal risk score's card
  reads it. Nothing is fitted to outcomes, so there is no time split.
- **calibration**: does a deal that stays in one band default as often as S&P
  says companies in that band do, in its region, at every horizon the study
  prints? The predictor against the region's table; the baseline reads the
  global table whatever the region, as the year-one default risk does.
  PLAN.md 5.3's "implied default rates match base rates for comparable
  bands within the card's tolerance" is this set's ``max_abs_gap_pp`` within
  ``CALIBRATION_TOLERANCE_PP``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from core.config import resolve_config
from core.deal import DealInputs, build_lbo_params
from core.risk_warnings import credit_view
from lbo_engine.model import run_lbo
from library import base_rates, references
from ml import distress_model as dm
from ml.evaluation.harness import Case, Metric, auc, evaluate_set, metric_specs

MODEL_ID = "distress"
HEADLINE_SET = "reference_deals"
# How far (percentage points) an implied cumulative default rate may sit from
# the table's for the same band, region and horizon
CALIBRATION_TOLERANCE_PP = 0.01
# Each table's region in the card (Table 24, global, is what the other
# developed markets read)
TABLE_REGION = {"us": "us", "europe": "europe", "emerging": "emerging", "global": "other_developed"}


def deal_inputs(d: dict) -> DealInputs:
    value = references.value(d, "transaction_value")
    ebitda = references.value(d, "ebitda")
    debt = references.value(d, "debt")
    return DealInputs(ebitda=ebitda, entry_mult=value / ebitda, debt_pct=debt / value * 100, country=d["country"])


def cases() -> list[Case]:
    return [Case(id=d["key"], region=base_rates.sp_region(d["country"]), year=references.closed_year(d),
                 features={"deal": d}, truth=d["outcome"]["kind"] == "distress")
            for d in references.repository_deals()]


def run(d: dict):
    inputs = deal_inputs(d)
    return inputs, run_lbo(replace(build_lbo_params(inputs, resolve_config()), compute_sensitivity=False))


def predict(case: Case) -> dict:
    """The predictor's chance of default over the hold, whatever the card says."""
    inputs, result = run(case.features["deal"])
    view = dm.deal_view(inputs, result, card=_ALWAYS_SHOWN)
    return {"score": view["years"][-1]["cumulative"]}


def baseline(case: Case) -> dict:
    """The deal's year-one default risk over the hold."""
    _, result = run(case.features["deal"])
    pct = credit_view(result)["default_pct"]
    return {"score": 0.0 if pct is None else pct / 100}


# A card under which every region shows the probabilities, so the evaluation
# reads them (the app's card decides what users see)
_ALWAYS_SHOWN = {"evaluation": {"headline_set": "x", "headline_metric": "auc", "sets": {"x": {"by_region": {
    r: {"verdict": "beats_baseline", "cases": 0} for r in base_rates.SP_REGIONS}}}}}


def _auc(cases_, preds) -> Optional[float]:
    return auc([p["score"] for p in preds], [c.truth for c in cases_])


def _brier(cases_, preds) -> Optional[float]:
    if not cases_:
        return None
    return sum((p["score"] - (1.0 if c.truth else 0.0)) ** 2 for c, p in zip(cases_, preds)) / len(cases_)


DEAL_METRICS = (
    Metric("auc", "higher", _auc, 0.005,
           "Chance a distressed deal is given a higher chance of default than a deal that wasn't "
           "(0.5 = no better than chance)"),
    Metric("brier", "lower", _brier, 0.005,
           "Mean squared gap between the chance of default over the hold and what happened (distress = 1)"),
)


def calibration_cases() -> list[Case]:
    """One case per table, band and horizon the study prints."""
    out = []
    for table, region in TABLE_REGION.items():
        rows = base_rates.CUMULATIVE[table]
        for band in dm.BANDS:
            for years, pct in enumerate(rows[band], start=1):
                out.append(Case(id=f"{table}:{band}:{years}", region=region, year=None,
                                features={"table": table, "band": band, "years": years}, truth=pct))
    return out


def implied_pct(table: str, band: str, years: int) -> float:
    """The predictor's cumulative default rate (%) for a deal that stays in
    ``band`` for ``years``, reading ``table``."""
    _, cumulative = dm.probabilities([dm.BANDS.index(band)] * years, table)
    return float(cumulative[-1]) * 100


def _gaps(cases_, preds):
    return [abs(p["pct"] - c.truth) for c, p in zip(cases_, preds)]


def _mean_gap(cases_, preds) -> Optional[float]:
    gaps = _gaps(cases_, preds)
    return sum(gaps) / len(gaps) if gaps else None


def _max_gap(cases_, preds) -> Optional[float]:
    gaps = _gaps(cases_, preds)
    return max(gaps) if gaps else None


CALIBRATION_METRICS = (
    Metric("mean_abs_gap_pp", "lower", _mean_gap, 0.005,
           "Mean gap (percentage points) between the implied cumulative default rate and S&P's for the same "
           "band, region and horizon"),
    Metric("max_abs_gap_pp", "lower", _max_gap, 0.005, "The largest of those gaps"),
)


def calibration_rows() -> list[tuple[Case, dict, dict]]:
    global_rows = base_rates.CUMULATIVE["global"]
    rows = []
    for c in calibration_cases():
        f = c.features
        rows.append((c, {"pct": implied_pct(f["table"], f["band"], f["years"])},
                     {"pct": float(global_rows[f["band"]][f["years"] - 1])}))
    return rows


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    deal_rows = [(c, predict(c), baseline(c)) for c in cases()]
    cal = calibration_rows()
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": "auc",
        "metrics": metric_specs(DEAL_METRICS + CALIBRATION_METRICS),
        "split": {"kind": "none", "why": "the predictor fits nothing to outcomes, so no case can leak into it; "
                                         "every case reads the tables of S&P's 2024 study and its January "
                                         "2024 methodology"},
        "baseline": {"id": "year_one_default_risk", "description":
                     "The deal summary's default risk: year-one EBIT / interest through Damodaran's coverage "
                     "table, and S&P's global cumulative default rate for that rating over the hold"},
        "calibration_tolerance_pp": CALIBRATION_TOLERANCE_PP,
        "sets": {
            HEADLINE_SET: {
                "description": f"The sourced reference transactions ({len(deal_rows)}), each entered with its "
                               "filed EBITDA, multiple and debt share in its own country, everything else at the "
                               "defaults, business risk 4 (fair); truth: distress at any time after closing",
                **evaluate_set(deal_rows, DEAL_METRICS, "auc")},
            "calibration": {
                "description": f"A deal held in one band, against S&P's cumulative default rate for that band "
                               f"in its region at every horizon printed ({len(cal)} cases: Table 25 for the "
                               "US, Europe and emerging markets, Table 24 for the other developed markets); "
                               "the baseline reads Table 24 everywhere",
                **evaluate_set(cal, CALIBRATION_METRICS, "max_abs_gap_pp")},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Distress predictor (default probability by year)",
    "module": "ml/distress_model.py",
    "used_by": "Deal -> Debt and Monte Carlo -> Risk, distress by year (POST /api/deal/run, "
               "POST /api/montecarlo/run); probabilities shown only in regions where this card says it beats "
               "the baseline",
    "estimates": "Each year's chance of default: the year's rating band (the weaker of EBIT / interest through "
                 "Damodaran's coverage table and debt / EBITDA through S&P's Corporate Methodology Tables 17 "
                 "and 3) and S&P's forward default rate for that band and age in the deal's region",
    "training": {
        "data": "Nothing is trained: the predictor reads published tables (S&P's 2024 default study, Tables 24 "
                "and 25; S&P's January 2024 Corporate Methodology, Tables 3 and 17; Damodaran's January 2026 "
                "coverage table)",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "Ten sourced transactions: only the United States has enough (six, two distressed); every other "
        "region shows not enough data.",
        "On those ten deals it does not rank distress better than the year-one default risk: Dollar General "
        "(10x leverage) and Avago (13x) exited well while Toys \"R\" Us and Gymboree (about 6.5x) failed years "
        "later. The app therefore shows the bands but not the probabilities anywhere today.",
        "Truth is distress at any time after closing; the predictor's horizon is the hold, so a failure years "
        "after an exit counts against it.",
        "S&P's regional tables print letter grades only and ten years for Europe and emerging markets; past "
        "the last year the last year's rate is held.",
        "The business risk profile is the user's choice (4, fair, unless changed); where Table 3 gives two "
        "anchors the weaker is read.",
        "Calibration is exact by construction for a deal that stays in one band; a deal that moves between "
        "bands reads each band's rate at its age, which S&P's tables don't test.",
    ],
}
