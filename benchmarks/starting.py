"""A new deal's starting figures, each with its source, sample and date (PLAN.md 4.3).

A deal is started from its country, its industry, its size and its
currency. Every starting figure comes from published data on many
companies or economies, never from a few deals:

- **margins, D&A, multiples, capex, working capital, leverage and interest
  coverage** from Damodaran's industry averages for the deal's region
  (``benchmarks.catalogue``), the closest group with enough companies;
- **revenue growth** is the country's nominal GDP growth this year, the
  IMF's projection of real growth and inflation (PLAN.md 4.2's stored
  series); the free industry growth figures are averages of single
  companies' growth and run far above what a whole industry grows
  (decided with the user, 2026-10-07);
- **tax** is the country's marginal corporate rate (the Tax Foundation's
  survey, as Damodaran publishes it);
- **the interest rate** is the currency's benchmark at today's level (4.2)
  plus Damodaran's default spread for the rating the industry's interest
  coverage implies.

Choices stated on screen: the exit multiple starts equal to the entry
multiple (no expansion assumed); debt starts at the industry's
**listed-company** leverage, all senior, because no free source publishes
buyout leverage (decided with the user, 2026-10-07); the figures are the
same for every size, because no free source splits them by size.

A figure that can't be found is left out of ``inputs`` and listed in
``missing``; the deal keeps what it has for it.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Mapping, Optional

from benchmarks.catalogue import (
    ALL_INDUSTRIES_ID, COUNTRY_TAX, DEFAULT_SPREAD_PCT, MIN_FIRMS, REGIONS, SOURCE, SPREAD_SOURCE_ID, chain,
    region_of,
)
from benchmarks.damodaran import Table
from core.risk_sources import SOURCES as RISK_SOURCES, rating_for_coverage
from economy import views
from economy.catalogue import COUNTRIES, CURRENCY_BENCHMARKS, SOURCES as ECONOMY_SOURCES, series_key
from economy.model import Series

DAYS_A_YEAR = 365
MAX_MULTIPLE = 100.0          # an EV/EBITDA above this is not a usable starting point
MAX_DEBT_PCT = 99.0           # DealInputsIn's bound on debt / EV
MAX_CAPEX_TO_DA = 10.0        # capex ten times D&A is a building phase, not a starting point
SENIOR_ONLY_PCT = 100.0
NOTES = ("size_not_split", "exit_equals_entry", "listed_company_leverage")


@dataclass(frozen=True)
class Skipped:
    """A group passed over on the way to the one used, and why."""

    area: str
    reason: str                 # "thin" (fewer than MIN_FIRMS), "unusable", "missing"
    sample: Optional[int] = None


@dataclass(frozen=True)
class Figure:
    """Where one starting figure came from."""

    field: str                  # the deal input it fills
    value: float
    source: str                 # damodaran, imf, tax_foundation, benchmark, choice
    dataset: str                # the table or series read
    area: str                   # the group or country read: "europe", "DE" ...
    level: str                  # country, region or global
    sample: Optional[int]       # how many companies, economies or countries stand behind it
    sample_kind: Optional[str]  # companies, economies, countries
    as_of: Optional[str]
    url: Optional[str]
    skipped: tuple = ()         # Skipped groups, closest first
    detail: Mapping = field(default_factory=dict)


@dataclass
class _Pick:
    row: dict
    area: str
    table: Table
    skipped: list


def _pick(tables: Mapping[str, Table], dataset: str, country: str, industry: str, usable) -> tuple[Optional[_Pick], list]:
    """The closest group in the country's chain with enough companies and a
    usable row; and the groups passed over."""
    skipped = []
    for area in chain(country):
        table = tables.get(f"{dataset}.{area}")
        row = table.rows.get(industry) if table else None
        if row is None:
            skipped.append(Skipped(area, "missing"))
        elif row.get("firms", 0) < MIN_FIRMS:
            skipped.append(Skipped(area, "thin", row.get("firms")))
        elif not usable(row):
            skipped.append(Skipped(area, "unusable", row.get("firms")))
        else:
            return _Pick(row, area, table, list(skipped)), skipped
    return None, skipped


def _damodaran(name: str, value: float, pick: _Pick, dataset: str, **detail) -> Figure:
    return Figure(name, value, SOURCE["id"], f"{dataset}.{pick.area}", pick.area, REGIONS[pick.area].level,
                  pick.row["firms"], "companies", pick.table.published.isoformat() if pick.table.published else None,
                  pick.table.url, tuple(pick.skipped), detail)


def _num(row: Mapping, *metrics: str) -> bool:
    return all(isinstance(row.get(m), (int, float)) for m in metrics)


def _margins_usable(r: Mapping) -> bool:
    return (_num(r, "gross_margin", "operating_margin", "ebitda_margin")
            and 0 < r["gross_margin"] <= 1 and r["ebitda_margin"] > 0
            and r["operating_margin"] <= r["ebitda_margin"] <= r["gross_margin"])


def _multiples_usable(r: Mapping) -> bool:
    return _num(r, "ev_ebitda") and 0 < r["ev_ebitda"] <= MAX_MULTIPLE


def _capex_usable(r: Mapping) -> bool:
    return _num(r, "capex_to_da") and 0 <= r["capex_to_da"] <= MAX_CAPEX_TO_DA


def _wc_usable(r: Mapping) -> bool:
    keys = ("receivables_sales", "inventory_sales", "payables_sales", "noncash_wc_sales")
    return _num(r, *keys) and all(0 <= r[k] < 1 for k in keys[:3]) and -1 < r["noncash_wc_sales"] < 1


def _debt_usable(r: Mapping) -> bool:
    return _num(r, "debt_ebitda", "interest_coverage") and r["debt_ebitda"] >= 0


def _pct(x: float) -> float:
    return round(x * 100, 2)


# ---------------------------------------------------------------------------
# Economy-wide figures: growth and tax, pooled by region when a country has none
# ---------------------------------------------------------------------------
def _nominal_growth(economy: Mapping[str, Series], area: str, today: date) -> Optional[tuple[float, dict]]:
    """The IMF's projection for this year: (1 + real growth)(1 + inflation) - 1, in %."""
    figures = views.country_figures(economy, area, today)
    real, inflation = figures.get("gdp_growth"), figures.get("inflation")
    if real is None or inflation is None:
        return None
    value = ((1 + real.value / 100) * (1 + inflation.value / 100) - 1) * 100
    return value, {"real_growth": real.value, "inflation": inflation.value, "period": real.period, "url": real.url}


def _growth(economy: Mapping[str, Series], country: str, today: date) -> tuple[Optional[Figure], list]:
    own = _nominal_growth(economy, country, today) if country in COUNTRIES else None
    weo = ECONOMY_SOURCES["imf"]
    if own:
        value, detail = own
        return Figure("growth", round(value, 2), "imf", "NGDP_RPCH+PCPIPCH", country, "country", 1, "economies",
                      detail["period"], detail["url"], (), detail), []
    skipped = [Skipped(country, "missing")]
    region = region_of(country)
    for area, members in ((region, [c for c in COUNTRIES if region_of(c) == region]), ("global", list(COUNTRIES))):
        found = [g[0] for c in members if (g := _nominal_growth(economy, c, today))]
        if found:
            return Figure("growth", round(statistics.median(found), 2), "imf", "NGDP_RPCH+PCPIPCH", area,
                          REGIONS[area].level, len(found), "economies", str(today.year),
                          "https://www.imf.org/external/datamapper/datasets/WEO", tuple(skipped),
                          {"statistic": "median", "source_name": weo.name}), skipped
        skipped.append(Skipped(area, "missing"))
    return None, skipped


def growth_figure(economy: Mapping[str, Series], country: str, today: date) -> Optional[dict]:
    """The country's nominal growth this year as the starting figures give
    it (its own, else its region's median), or None when none is stored."""
    found, _ = _growth(economy, country, today)
    return _plain(found) if found else None


def _tax(tables: Mapping[str, Table], country: str) -> tuple[Optional[Figure], list]:
    table = tables.get(COUNTRY_TAX["id"])
    if table is None:
        return None, [Skipped(country, "missing")]
    as_of = table.published.isoformat() if table.published else None
    own = table.rows.get(country)
    if own:
        return Figure("tax", _pct(own["rate"]), "tax_foundation", COUNTRY_TAX["id"], country, "country", 1,
                      "countries", as_of, table.url), []
    skipped = [Skipped(country, "missing")]
    region = region_of(country)
    for area, members in ((region, [c for c in table.rows if region_of(c) == region]), ("global", list(table.rows))):
        rates = [table.rows[c]["rate"] for c in members]
        if rates:
            return Figure("tax", _pct(statistics.median(rates)), "tax_foundation", COUNTRY_TAX["id"], area,
                          REGIONS[area].level, len(rates), "countries", as_of, table.url, tuple(skipped),
                          {"statistic": "median"}), skipped
        skipped.append(Skipped(area, "missing"))
    return None, skipped


def _benchmark_level(economy: Mapping[str, Series], country: str, currency: str, today: date) -> Optional[tuple[str, views.Figure]]:
    """The currency's benchmark at today's level, else the country's policy rate."""
    code = CURRENCY_BENCHMARKS.get(currency)
    if code:
        fig = views.reference_rates(economy, today).get(code)
        if fig is not None:
            return code, fig
    policy = views.country_figures(economy, country, today).get("policy_rate") if country in COUNTRIES else None
    if policy is not None and policy.current:
        return "policy_rate", policy
    return None


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def starting_assumptions(country: str, industry: str, currency: str, tables: Mapping[str, Table],
                         economy: Mapping[str, Series], today: date) -> dict:
    """The starting figures for a deal in ``country`` and ``industry``
    (an industry id, ``all`` for every industry) in ``currency``."""
    figures: list[Figure] = []
    missing: dict[str, list] = {}

    def add(fig: Optional[Figure], skipped: list, *names: str) -> None:
        if fig is not None:
            figures.append(fig)
        else:
            for n in names:
                missing[n] = skipped

    growth, skipped = _growth(economy, country, today)
    add(growth, skipped, "growth")
    tax, skipped = _tax(tables, country)
    add(tax, skipped, "tax")

    margins, skipped = _pick(tables, "margins", country, industry, _margins_usable)
    gm = da = None
    if margins:
        r = margins.row
        gm, da = r["gross_margin"], r["ebitda_margin"] - r["operating_margin"]
        add(_damodaran("gross_margin", _pct(gm), margins, "margins"), [])
        add(_damodaran("opex", _pct(gm - r["operating_margin"]), margins, "margins",
                       operating_margin=r["operating_margin"]), [])
        add(_damodaran("da", _pct(da), margins, "margins", ebitda_margin=r["ebitda_margin"],
                       operating_margin=r["operating_margin"]), [])
    else:
        add(None, skipped, "gross_margin", "opex", "da", "capex", "ar_days", "inv_days", "ap_days")

    multiples, skipped = _pick(tables, "multiples", country, industry, _multiples_usable)
    multiple = multiples.row["ev_ebitda"] if multiples else None
    if multiples:
        add(_damodaran("entry_mult", round(multiple, 2), multiples, "multiples"), [])
        add(_damodaran("exit_mult", round(multiple, 2), multiples, "multiples", note="exit_equals_entry"), [])
    else:
        add(None, skipped, "entry_mult", "exit_mult", "debt_pct")

    if margins:
        capex, skipped = _pick(tables, "capex", country, industry, _capex_usable)
        if capex:
            add(_damodaran("capex", _pct(da * capex.row["capex_to_da"]), capex, "capex",
                           capex_to_da=capex.row["capex_to_da"], da=round(da, 6)), [])
        else:
            add(None, skipped, "capex")

    wc, skipped = _pick(tables, "working_capital", country, industry, _wc_usable)
    if wc:
        r = wc.row
        if gm is not None and gm < 1:
            days = {"ar_days": r["receivables_sales"] * DAYS_A_YEAR,
                    "inv_days": r["inventory_sales"] * DAYS_A_YEAR / (1 - gm),
                    "ap_days": r["payables_sales"] * DAYS_A_YEAR / (1 - gm)}
            for name, value in days.items():
                if value <= DAYS_A_YEAR:
                    add(_damodaran(name, round(value, 1), wc, "working_capital"), [])
                else:
                    add(None, [Skipped(wc.area, "unusable", r["firms"])], name)
        else:                                   # no gross margin to put the days on cost of sales
            add(None, [Skipped(wc.area, "unusable", r["firms"])], "ar_days", "inv_days", "ap_days")
        if growth is not None:
            g = growth.value / 100
            add(_damodaran("nwc", _pct(r["noncash_wc_sales"] * g / (1 + g)), wc, "working_capital",
                           noncash_wc_sales=r["noncash_wc_sales"], growth=growth.value), [])
        else:
            add(None, missing.get("growth", []), "nwc")
    else:
        add(None, skipped, "ar_days", "inv_days", "ap_days", "nwc")

    debt, skipped = _pick(tables, "debt", country, industry, _debt_usable)
    if debt and multiple:
        debt_pct = debt.row["debt_ebitda"] / multiple * 100
        if debt_pct <= MAX_DEBT_PCT:
            add(_damodaran("debt_pct", round(debt_pct, 2), debt, "debt", debt_ebitda=debt.row["debt_ebitda"],
                           ev_ebitda=multiple, note="listed_company_leverage"), [])
            figures.append(Figure("senior_pct", SENIOR_ONLY_PCT, "choice", "listed_company_leverage", debt.area,
                                  REGIONS[debt.area].level, None, None, None, None))
        else:
            add(None, [*skipped, Skipped(debt.area, "unusable", debt.row["firms"])], "debt_pct")
    elif not debt:
        add(None, skipped, "debt_pct")

    benchmark = _benchmark_level(economy, country, currency, today)
    if debt and benchmark:
        rating, _, _ = rating_for_coverage(debt.row["interest_coverage"])
        spread = DEFAULT_SPREAD_PCT[rating]
        code, level = benchmark
        add(Figure("base_rate", round(level.value + spread, 2), "benchmark", code, debt.area,
                   REGIONS[debt.area].level, debt.row["firms"], "companies", level.period, level.url,
                   tuple(debt.skipped),
                   {"benchmark": code, "benchmark_level": level.value, "benchmark_source": level.source,
                    "benchmark_basis": level.basis, "interest_coverage": round(debt.row["interest_coverage"], 2),
                    "rating": rating, "spread": spread, "spread_source": SPREAD_SOURCE_ID,
                    "spread_url": RISK_SOURCES[SPREAD_SOURCE_ID]["url"]}), [])
    else:
        add(None, skipped if not debt else [Skipped(currency, "missing")], "base_rate")

    inputs = {f.field: f.value for f in figures}
    world = tables.get("margins.global")
    return {
        "country": country, "industry": industry, "currency": currency,
        "industry_name": world.rows[industry]["name"] if world and industry in world.rows else None,
        "region": region_of(country), "chain": list(chain(country)), "min_firms": MIN_FIRMS,
        "inputs": inputs,
        "figures": [_plain(f) for f in figures],
        "missing": [{"field": k, "skipped": [asdict(s) for s in v]} for k, v in missing.items() if k not in inputs],
        "notes": list(NOTES),
        "source": dict(SOURCE),
    }


def _plain(f: Figure) -> dict:
    out = asdict(f)
    out["skipped"] = [asdict(s) for s in f.skipped]
    out["detail"] = dict(f.detail)
    return out


def industries(tables: Mapping[str, Table]) -> list[dict]:
    """Every industry the stored averages cover, by name, with the
    companies in the global group (``all`` first)."""
    table = tables.get("margins.global")
    if table is None:
        return []
    rows = sorted(table.rows.items(), key=lambda kv: (kv[0] != ALL_INDUSTRIES_ID, kv[1]["name"]))
    return [{"id": k, "name": r["name"], "firms": r["firms"]} for k, r in rows]
