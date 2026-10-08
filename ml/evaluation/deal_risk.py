"""The deal risk score's evaluation (ml/anomaly_detector.py; PLAN.md 5.1, 5.2).

Question: does how far a deal's leverage and price sit above its industry's
in its region rank the buyouts that ended in distress above the ones that
didn't, better than leverage alone?

- **Cases**: the sourced reference transactions in the repository
  (``library/reference_deals.json``, PLAN.md 4.5b: every figure read from a
  filing), each placed in its S&P region and given the Damodaran industry
  its business belongs to (``INDUSTRY``; the transactions record a GICS
  sector, which is coarser). Truth: distress (a missed payment, a
  bankruptcy or a restructuring). Whether a transaction is approved in a
  deployed library doesn't matter here: the evaluation reads the filings'
  figures, not a user's library, and the score works with the library off.
- **Peers**: Damodaran's January 2026 averages
  (``data/peer_tables.json``, written by ``python -m tests.ml_peer_tables``),
  exactly what the app reads.
- **No time split**: the score fits nothing to outcomes -- it compares a
  deal with published averages -- so no case can leak into it. Its weakness
  is the other way round: the averages are later than every deal (no
  earlier edition is stored for debt), as the limitations say.
- **Abstaining**: a deal whose industry has too few companies in its
  region gets no score in the app ("not enough data") and is left out of
  the set, as is its baseline.
- **Baseline**: leverage alone, flagged above the supervisory 6.0x
  (``core.risk_sources.LEVERAGE_GUIDANCE_X``), as the 5.1 card's.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from benchmarks.damodaran import Table
from core.risk_sources import LEVERAGE_GUIDANCE_X
from library import base_rates, references
from ml.anomaly_detector import DealShape, compare
from ml.evaluation.harness import Case, Metric, auc, evaluate_set, metric_specs, share

MODEL_ID = "deal_risk"
PEER_TABLES = Path(__file__).parent / "data" / "peer_tables.json"
HEADLINE_SET = "reference_deals"
# The Damodaran industry (benchmarks, January 2026 ids) of each reference
# transaction's business, read from its filings' description of it
INDUSTRY = {
    "hca-2006": "hospitals_healthcare_facilities",
    "toys-r-us-2005": "retail_special_lines",
    "dollar-general-2007": "retail_general",
    "dominos-1998": "restaurant_dining",
    "gymboree-2010": "retail_special_lines",
    "dun-bradstreet-2019": "information_services",
    "nxp-2006": "semiconductor",
    "masonite-2005": "building_materials",
    "avago-2005": "semiconductor",
    "focus-media-2013": "advertising",
}


def peer_tables() -> dict[str, Table]:
    data = json.loads(PEER_TABLES.read_text(encoding="utf-8"))["tables"]
    return {name: Table(name, date.fromisoformat(t["published"]) if t["published"] else None, t["url"], t["rows"])
            for name, t in data.items()}


def cases() -> list[Case]:
    """Every repository transaction with the figures the score reads."""
    out = []
    for d in references.repository_deals():
        derived = references.derived(d)
        out.append(Case(id=d["key"], region=base_rates.sp_region(d["country"]), year=references.closed_year(d),
                        features={"country": d["country"], "industry": INDUSTRY[d["key"]],
                                  "leverage": derived["leverage"], "entry_multiple": derived["entry_multiple"]},
                        truth=d["outcome"]["kind"] == "distress"))
    return out


def predict(tables, case: Case) -> Optional[dict]:
    """The score and flag the app would give ``case``; None where it says
    "not enough data"."""
    f = case.features
    found = compare(DealShape(f["country"], f["industry"], f["leverage"], f["entry_multiple"]), tables)
    if found["score"] is None:
        return None
    return {"score": found["score"], "flag": found["unusual"]}


def baseline(case: Case) -> dict:
    """Leverage alone."""
    lev = case.features["leverage"]
    return {"score": lev, "flag": lev > LEVERAGE_GUIDANCE_X}


def _auc(cases_, preds) -> Optional[float]:
    return auc([p["score"] for p in preds], [c.truth for c in cases_])


def _caught(cases_, preds) -> Optional[float]:
    return share([p["flag"] for c, p in zip(cases_, preds) if c.truth])


def _false_alarms(cases_, preds) -> Optional[float]:
    return share([p["flag"] for c, p in zip(cases_, preds) if not c.truth])


METRICS = (
    Metric("auc", "higher", _auc, 0.005,
           "Chance a distressed deal scores above a deal that wasn't (0.5 = no better than chance)"),
    Metric("distress_flagged", "higher", _caught, 0.005, "Share of distressed deals flagged"),
    Metric("false_alarms", "lower", _false_alarms, 0.005, "Share of deals without distress flagged"),
)
HEADLINE = "auc"


def scored(tables=None) -> list[tuple[Case, dict, dict]]:
    """``(case, prediction, baseline)`` for every case the score answers."""
    tables = tables if tables is not None else peer_tables()
    rows = [(c, predict(tables, c), baseline(c)) for c in cases()]
    return [r for r in rows if r[1] is not None]


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    """The card's one set: every reference transaction the score answers."""
    rows = scored()
    left_out = sorted(c.id for c in cases() if c.id not in {r[0].id for r in rows})
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "none", "why": "the score fits nothing to outcomes, so no case can leak into it; "
                                         "every case is scored against the January 2026 averages"},
        "baseline": {"id": "leverage", "description":
                     f"Debt / EBITDA as the score; flagged above {LEVERAGE_GUIDANCE_X:g}x (the supervisory "
                     "leverage limit the deal's risk warnings use)"},
        "sets": {
            HEADLINE_SET: {
                "description": "The sourced reference transactions whose industry has enough companies in "
                               f"their region ({len(rows)} of {len(cases())}; left out: {', '.join(left_out)})",
                **evaluate_set(rows, METRICS, HEADLINE)},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Deal risk score (against companies like it)",
    "module": "ml/anomaly_detector.py",
    "used_by": "Deal -> Inputs, Deal risk (POST /api/ml/deal-risk); shown only in regions where this card "
               "says it beats the baseline",
    "estimates": "How far a deal's leverage and entry multiple sit above its industry's listed companies "
                 "in its region (the sum of the positive gaps, each over how much the region's industries "
                 "differ), and an unusual-deal flag",
    "training": {
        "data": "Nothing is trained: the score reads Damodaran's industry averages for the deal's region "
                "(PLAN.md 4.3), refreshed nightly",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "Ten sourced transactions, most of them US buyouts: only the United States has five scored cases, "
        "two of them distressed; every other region shows not enough data, so the app shows no score "
        "there.",
        "Each case moves the statistics by many points: the verdict says the score ranked these deals "
        "better than leverage alone, not that it will in general.",
        "The averages are January 2026's, later than every deal; industries' leverage and prices in the "
        "year each deal closed were different.",
        "The averages are listed companies' and are not split by size.",
        "The EBITDA margin is compared on screen but not scored: the transactions' filings don't all "
        "give revenue.",
    ],
}
