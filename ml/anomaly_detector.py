"""The deal risk score: a deal against companies and deals like it (PLAN.md 5.2).

The deal's leverage, entry multiple and EBITDA margin are compared with the
**listed companies of its industry in its own region**: Damodaran's industry
averages (PLAN.md 4.3, ``benchmarks/``), which say how many companies stand
behind each figure. Nothing is trained; the comparison reads the stored
averages, so it follows every refresh.

- **The peer group** of a figure is the closest group in the country's chain
  (``benchmarks.catalogue.chain``) whose industry row has at least
  ``MIN_FIRMS`` companies: the country's own file (the US, Japan, China,
  India), else its region. **Never the global group**: a deal is compared
  with companies in its own region or not at all, and a region with too few
  companies in the industry says "not enough data".
- **How far off** the deal is, ``z``: its figure less the industry's, over
  how much industries in that group differ from each other -- the robust
  spread (1.4826 x the median absolute deviation) across the group's
  industries with ``MIN_FIRMS`` companies, at least ``MIN_INDUSTRIES`` of
  them. ``z`` is signed so that positive is riskier: higher leverage, a
  higher price, a lower margin.
- **The score** is how far the deal's leverage and price sit above its
  industry's, the sum of their positive ``z``; the margin is shown beside
  them but not scored, because the deals the score is tested on
  (``ml/evaluation/deal_risk.py``) don't all publish revenue. The deal is
  **unusual** when any figure is ``WELL_Z`` or more on the risky side.
- **The score is shown only where its card says it beats the baseline**
  (leverage alone) in the deal's S&P region; elsewhere the screen says "not
  enough data", as PLAN.md phase 5 requires. The comparison itself is
  published data, shown wherever the peer group is large enough.
- **Deals like it**, when the reference library is on (PLAN.md 4.5): the
  approved reference transactions in the same S&P region and GICS sector,
  marked when they are also the same size. With the library off the score
  and the comparison are unchanged; only this list is left out.

The industry averages are not split by company size (no free source does),
which the answer's ``notes`` say.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional

from benchmarks.catalogue import ALL_INDUSTRIES_ID, MIN_FIRMS, REGIONS, SOURCE, chain
from benchmarks.damodaran import Table
from library import base_rates, coverage, references

CARD = Path(__file__).parent / "cards" / "deal_risk.json"
# A group needs this many industries (each with MIN_FIRMS companies) to say
# how much industries differ
MIN_INDUSTRIES = 10
# Robust spread: the median absolute deviation scaled to a normal's standard deviation
MAD_TO_SD = 1.4826
# |z| from which a figure is above (below) its industry's, and well above (below)
ABOVE_Z = 1.0
WELL_Z = 2.0
NOTES = ("size_not_split", "listed_company_figures")


@dataclass(frozen=True)
class Metric:
    id: str
    dataset: str           # the benchmarks data set (benchmarks.catalogue.DATASETS)
    column: str            # its column
    direction: int         # +1: higher is riskier; -1: lower is riskier
    scored: bool           # part of the score (and so of its card)
    percent: bool          # stored as a fraction, compared in per cent


METRICS = (
    Metric("leverage", "debt", "debt_ebitda", 1, True, False),
    Metric("entry_multiple", "multiples", "ev_ebitda", 1, True, False),
    Metric("ebitda_margin", "margins", "ebitda_margin", -1, False, True),
)
DATASETS_READ = tuple(dict.fromkeys(m.dataset for m in METRICS))


@dataclass(frozen=True)
class DealShape:
    """What the score reads of a deal."""
    country: str               # ISO 3166-1 alpha-2; "" when not chosen
    industry: str              # a Damodaran industry id; "" for the whole market
    leverage: float            # debt at close / EBITDA, x
    entry_multiple: float      # EV / EBITDA, x
    ebitda_margin: Optional[float] = None   # per cent; None when unknown


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def peer_group(tables: Mapping[str, Table], metric: Metric, country: str,
               industry: str) -> tuple[Optional[str], Optional[dict], list[dict]]:
    """The closest group in the country's own region with enough companies
    and a figure: ``(group, row, skipped)``; ``(None, None, skipped)`` when
    none has. The global group is never used."""
    skipped = []
    for area in chain(country):
        if REGIONS[area].level == "global":
            break
        table = tables.get(f"{metric.dataset}.{area}")
        row = table.rows.get(industry) if table else None
        if row is None:
            skipped.append({"area": area, "reason": "missing", "sample": None})
        elif row.get("firms", 0) < MIN_FIRMS:
            skipped.append({"area": area, "reason": "thin", "sample": row.get("firms")})
        elif not _number(row.get(metric.column)):
            skipped.append({"area": area, "reason": "unusable", "sample": row.get("firms")})
        else:
            return area, row, skipped
    return None, None, skipped


def spread(table: Table, column: str) -> tuple[Optional[float], int]:
    """How much the group's industries differ on ``column``: the robust
    spread over its industries with MIN_FIRMS companies, and how many there
    were; None under MIN_INDUSTRIES or with no spread at all."""
    values = [r[column] for k, r in table.rows.items()
              if k != ALL_INDUSTRIES_ID and r.get("firms", 0) >= MIN_FIRMS and _number(r.get(column))]
    if len(values) < MIN_INDUSTRIES:
        return None, len(values)
    middle = statistics.median(values)
    mad = statistics.median(abs(v - middle) for v in values) * MAD_TO_SD
    return (mad if mad > 0 else None), len(values)


def position(z: float) -> str:
    """Where the deal's figure sits against its industry's (signed so that
    ``above`` is a higher figure, whichever way is riskier)."""
    if z >= WELL_Z:
        return "well_above"
    if z >= ABOVE_Z:
        return "above"
    if z <= -WELL_Z:
        return "well_below"
    if z <= -ABOVE_Z:
        return "below"
    return "in_line"


def _value(deal: DealShape, metric: Metric) -> Optional[float]:
    return {"leverage": deal.leverage, "entry_multiple": deal.entry_multiple,
            "ebitda_margin": deal.ebitda_margin}[metric.id]


def compare_one(tables: Mapping[str, Table], deal: DealShape, metric: Metric) -> dict:
    """One figure of the deal against its industry's in its region."""
    value = _value(deal, metric)
    out = {"metric": metric.id, "scored": metric.scored, "deal": value, "status": "not_enough_data",
           "peer": None, "spread": None, "z": None, "risk_z": None, "position": None, "group": None,
           "level": None, "firms": None, "industries": None, "published": None, "url": None, "skipped": []}
    group, row, skipped = peer_group(tables, metric, deal.country, deal.industry)
    out["skipped"] = skipped
    if group is None or value is None:
        return out
    table = tables[f"{metric.dataset}.{group}"]
    s, n = spread(table, metric.column)
    out.update(group=group, level=REGIONS[group].level, firms=row["firms"], industries=n,
               published=table.published, url=table.url)
    if s is None:
        out["skipped"] = [*skipped, {"area": group, "reason": "few_industries", "sample": n}]
        return out
    scale = 100.0 if metric.percent else 1.0
    peer, s = row[metric.column] * scale, s * scale
    z = (value - peer) / s
    out.update(status="ok", peer=round(peer, 4), spread=round(s, 4), z=round(z, 4),
               risk_z=round(z * metric.direction, 4), position=position(z))
    return out


def score_of(comparisons: Iterable[dict]) -> Optional[float]:
    """The sum of the scored figures' positive risk ``z``; None unless every
    scored figure has a peer group."""
    scored = [c for c in comparisons if c["scored"]]
    if not scored or any(c["status"] != "ok" for c in scored):
        return None
    return round(sum(max(c["risk_z"], 0.0) for c in scored), 4)


def unusual(comparisons: Iterable[dict]) -> bool:
    """Whether any figure sits WELL_Z or more on the risky side of its industry's."""
    return any(c["status"] == "ok" and c["risk_z"] >= WELL_Z for c in comparisons)


def compare(deal: DealShape, tables: Mapping[str, Table]) -> dict:
    """Every figure against the industry's in the deal's region, the score and the flag."""
    region = base_rates.sp_region(deal.country)
    deal = replace(deal, industry=deal.industry or ALL_INDUSTRIES_ID)
    if not deal.country:
        return {"status": "not_enough_data", "reason": "no_country", "region": None, "comparisons": [],
                "sample": None, "score": None, "unusual": False}
    comparisons = [compare_one(tables, deal, m) for m in METRICS]
    ok = [c for c in comparisons if c["status"] == "ok"]
    if not ok:
        return {"status": "not_enough_data", "reason": "no_peers", "region": region, "comparisons": comparisons,
                "sample": None, "score": None, "unusual": False}
    # "Based on N companies in [group, industry]": the smallest group a figure came from
    smallest = min(ok, key=lambda c: c["firms"])
    return {"status": "ok", "reason": None, "region": region, "comparisons": comparisons,
            "sample": {"firms": smallest["firms"], "group": smallest["group"]},
            "score": score_of(comparisons), "unusual": unusual(comparisons)}


# ---------------------------------------------------------------------------
# The card decides where the score is shown
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _card() -> dict:
    return json.loads(CARD.read_text(encoding="utf-8"))


def card_result(region: Optional[str], card: Optional[Mapping] = None) -> dict:
    """The card's headline verdict for ``region`` (S&P's), with the cases and
    the headline statistic for the score and for the baseline."""
    ev = (card or _card())["evaluation"]
    group = ev["sets"][ev["headline_set"]]["by_region"].get(region) if region else None
    if not group:
        return {"verdict": "not_enough_data", "cases": 0, "model": None, "baseline": None}
    head = ev["headline_metric"]
    return {"verdict": group["verdict"], "cases": group["cases"],
            "model": (group.get("model") or {}).get(head), "baseline": (group.get("baseline") or {}).get(head)}


def shown_score(found: Mapping, card: Optional[Mapping] = None) -> dict:
    """The score as the screen may show it: its value only where the card
    says it beats the baseline in the deal's region."""
    result = card_result(found["region"], card)
    shown = result["verdict"] == "beats_baseline" and found["score"] is not None
    return {"shown": shown, "value": found["score"] if shown else None, "region": found["region"], **result}


# ---------------------------------------------------------------------------
# Deals like it (the reference library, when it is on)
# ---------------------------------------------------------------------------
def sector_of(industry: str) -> Optional[str]:
    from validation.tags import INDUSTRY_SECTOR
    return INDUSTRY_SECTOR.get(industry)


def similar_deals(library: Iterable[dict], country: str, industry: str, ev_usd_m: Optional[float]) -> dict:
    """The approved reference transactions in the deal's S&P region and GICS
    sector, the same size first, then the newest."""
    region, sector, size = base_rates.sp_region(country), sector_of(industry), coverage.size(ev_usd_m)
    found = []
    for d in library:
        tags = references.tags(d)
        if not region or not sector or tags["region"] != region or tags["sector"] != sector:
            continue
        derived = references.derived(d)
        found.append({"key": d["key"], "target": d["target"], "country": d["country"], "sector": tags["sector"],
                      "size": tags["size"], "year": references.closed_year(d), "outcome": d["outcome"]["kind"],
                      "event": d["outcome"]["event"], "entry_multiple": derived["entry_multiple"],
                      "leverage": derived["leverage"], "same_size": size is not None and tags["size"] == size})
    found.sort(key=lambda x: (not x["same_size"], -x["year"], x["key"]))
    return {"enabled": True, "region": region, "sector": sector, "size": size, "deals": found}


def assess(deal: DealShape, tables: Mapping[str, Table], library: Optional[Iterable[dict]] = None,
           ev_usd_m: Optional[float] = None, card: Optional[Mapping] = None) -> dict:
    """The whole answer: the comparison, the score as it may be shown and,
    when the library is on (``library`` not None), the deals like it."""
    found = compare(deal, tables)
    world = tables.get("margins.global")
    industry = deal.industry or ALL_INDUSTRIES_ID
    deals = (similar_deals(library, deal.country, industry, ev_usd_m) if library is not None
             else {"enabled": False, "region": None, "sector": None, "size": None, "deals": []})
    return {
        "status": found["status"], "reason": found["reason"], "country": deal.country, "industry": industry,
        "industry_name": world.rows[industry]["name"] if world and industry in world.rows else None,
        "region": found["region"], "min_firms": MIN_FIRMS, "sample": found["sample"],
        "comparisons": found["comparisons"], "unusual": found["unusual"],
        "score": shown_score(found, card), "deals": deals, "notes": list(NOTES), "source": dict(SOURCE),
    }
