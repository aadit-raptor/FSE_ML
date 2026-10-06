"""From reported facts to summary figures per fiscal year, for every source.

A connector reads its filings into ``Fact``s -- one reported number each,
with its concept, currency, period and the filing it came from -- and
``summarize`` turns them into ``YearFigures``:

- **Annual and consolidated only.** A flow counts only over a period of a
  year (``ANNUAL_DAYS``: 52 or 53 weeks too); a fact with any dimension
  (a segment, a parent-only column) never becomes a Fact at all.
- **Named by the year the period ends in**, and a 52/53-week year that ends
  in a month's first week (``FIRST_WEEK``) belongs to the month before, as
  ``ml/edgar_extractor.py`` names them.
- **Each year from the highest-priority concept that has it** (the lists in
  ``companies.items``), and from the newest filing that reported it, so a
  restatement wins.
- Money in **millions** of the filing's currency.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

from companies.items import ABSOLUTE, DERIVED, FLOWS, SUMMARY_FIELDS, ConceptMap, Sum, concepts_of
from companies.model import FilingLink, Warning, YearFigures

ANNUAL_DAYS = (340, 380)
# A fiscal year is one some filing reports any of these for; equity alone
# only when neither is reported (a micro-entity's balance sheet)
ANCHORS = ("revenue", "total_assets")
LAST_ANCHOR = "total_equity"
FIRST_WEEK = 7
MILLION = 1_000_000.0


@dataclass(frozen=True)
class Fact:
    concept: str                 # "prefix:Name", prefixes canonical (companies.items)
    value: float                 # in units of the currency, as reported
    currency: str
    end: date                    # an instant's date, or a period's last day
    start: Optional[date]        # None for an instant
    filing: FilingLink


def fiscal_year_of(end: date) -> tuple[int, int]:
    """(year, month) of the fiscal year a period ending on ``end`` closes."""
    if end.day <= FIRST_WEEK:
        end = end.replace(day=1) - timedelta(days=1)
    return end.year, end.month


def is_annual(fact: Fact) -> bool:
    if fact.start is None:
        return True
    days = (fact.end - fact.start).days + 1
    return ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]


def main_currency(facts: Iterable[Fact], concepts: Iterable[str]) -> Optional[str]:
    """The currency the statements are in: the commonest among these concepts."""
    wanted = set(concepts)
    counts = Counter(f.currency for f in facts if f.concept in wanted and f.currency)
    return counts.most_common(1)[0][0] if counts else None


def _by_year(facts: list[Fact], concept: str, flow: Optional[bool]) -> dict[int, Fact]:
    """The newest-filed annual fact of ``concept`` per fiscal year: over a
    year when ``flow``, at a year end when not, either when None."""
    out: dict[int, Fact] = {}
    for f in facts:
        if f.concept != concept or (flow is not None and (f.start is None) == flow) or not is_annual(f):
            continue
        year = fiscal_year_of(f.end)[0]
        held = out.get(year)
        if held is None or (f.filing.filed_on or date.min) > (held.filing.filed_on or date.min):
            out[year] = f
    return out


def _alternative(facts: list[Fact], alt, flow: Optional[bool]) -> dict[int, tuple[float, Fact]]:
    """One alternative's value per year: a concept, or a ``Sum``."""
    parts = (alt,) if isinstance(alt, str) else alt.parts
    out: dict[int, tuple[float, Fact]] = {}
    seen: dict[int, int] = {}
    for part in parts:
        sign = -1.0 if part.startswith("-") else 1.0
        for year, f in _by_year(facts, part.lstrip("-"), flow).items():
            total, first = out.get(year, (0.0, f))
            out[year] = (total + sign * f.value, first)
            seen[year] = seen.get(year, 0) + 1
    if isinstance(alt, Sum) and alt.every:
        out = {y: v for y, v in out.items() if seen[y] == len(parts)}
    return out


def field_values(facts: list[Fact], alternatives: list, flow: Optional[bool],
                 years: list[int]) -> tuple[list[Optional[float]], list[Optional[Fact]]]:
    """Per year, the value of the highest-priority alternative reported for
    that year (millions), and the fact it came from."""
    values: list[Optional[float]] = [None] * len(years)
    sources: list[Optional[Fact]] = [None] * len(years)
    for alt in alternatives:
        found = _alternative(facts, alt, flow)
        for j, year in enumerate(years):
            if values[j] is None and year in found:
                total, fact = found[year]
                values[j], sources[j] = total / MILLION, fact
        if all(v is not None for v in values):
            break
    return values, sources


def single_concept(facts: list[Fact], alternatives: list, flow: Optional[bool],
                   years: list[int]) -> list[Optional[float]]:
    """Values from one alternative for every year: the first reported for the
    latest year, else the first reported at all (a lease cost must not mix a
    past cost with a forward payment, core/accounting.py)."""
    found = [field_values(facts, [alt], flow, years)[0] for alt in alternatives]
    found = [v for v in found if any(x is not None for x in v)]
    if not found:
        return [None] * len(years)
    return next((v for v in found if v[-1] is not None), found[0])


def fiscal_years(facts: list[Fact], anchors: list[str], n_years: int) -> list[tuple[int, date]]:
    """The latest ``n_years`` fiscal years with an anchor fact, oldest first,
    each with its period end."""
    ends: dict[int, date] = {}
    for f in facts:
        if f.concept in anchors and is_annual(f):
            year = fiscal_year_of(f.end)[0]
            ends[year] = max(ends.get(year, f.end), f.end)
    return sorted(ends.items())[-n_years:]


def _debt(debt_total, noncurrent, current):
    """Funded debt, current maturities included. Some filers' "total" equals
    the non-current figure (MCD), so the current part is added back then,
    as ml/edgar_extractor.py does."""
    if debt_total:
        if current and noncurrent and abs(debt_total - noncurrent) < 0.5:
            return debt_total + current
        return debt_total
    if noncurrent is None and current is None:
        return None
    return (noncurrent or 0.0) + (current or 0.0)


def summarize(facts: list[Fact], items: ConceptMap, n_years: int) -> tuple[list[YearFigures], list[Warning]]:
    """Summary figures for the latest ``n_years`` fiscal years in ``facts``."""
    def reported(names):
        """Years in which one of these fields has a value, with their ends."""
        concepts = [c for name in names for alt in items.fields[name] for c in concepts_of(alt)]
        candidates = fiscal_years(facts, concepts, 100)
        years = [y for y, _ in candidates]
        has = [any(v is not None for v in vals) for vals in zip(*(
            field_values(facts, items.fields[name], name in FLOWS, years)[0] for name in names))]
        return [c for c, ok in zip(candidates, has) if ok][-n_years:]

    picked = reported(ANCHORS) or reported((LAST_ANCHOR,))
    if not picked:
        return [], [Warning("no_annual_figures")]
    years = [y for y, _ in picked]

    raw: dict[str, list[Optional[float]]] = {}
    filings: list[Optional[Fact]] = [None] * len(years)
    for name, alternatives in items.fields.items():
        if name == "lease_cost":
            # A filer with no lease cost may tag next year's payments due, a balance
            raw[name] = single_concept(facts, alternatives, None, years)
            continue
        raw[name], sources = field_values(facts, alternatives, name in FLOWS, years)
        if name in ANCHORS + (LAST_ANCHOR,):
            filings = [filings[j] or sources[j] for j in range(len(years))]

    out, missing = [], set()
    for j, (year, period_end) in enumerate(picked):
        v = {name: raw[name][j] for name in raw}
        figures = {name: v.get(name) for name in SUMMARY_FIELDS if name not in DERIVED}
        figures["total_debt"] = _debt(v.get("debt_total"), v.get("debt_noncurrent"), v.get("debt_current"))
        if figures.get("total_equity") is None:
            figures["total_equity"] = v.get("equity_parent")
        lease_interest = v.get("lease_interest")
        if figures.get("lease_cost") is not None and lease_interest:
            figures["lease_cost"] += abs(lease_interest)
        liability = v.get("lease_liability")
        if liability is None and (v.get("lease_liability_current") is not None
                                  or v.get("lease_liability_noncurrent") is not None):
            liability = (v.get("lease_liability_current") or 0.0) + (v.get("lease_liability_noncurrent") or 0.0)
        figures["lease_liability"] = liability
        for name in ABSOLUTE:
            if figures.get(name) is not None:
                figures[name] = abs(figures[name])
        op, da = figures.get("operating_income"), figures.get("depreciation_amortization")
        figures["ebitda"] = None if op is None or da is None else op + da
        figures = {k: (None if figures.get(k) is None else round(figures[k], 6)) for k in SUMMARY_FIELDS}
        if j == len(picked) - 1:            # gaps in the latest year are the ones that matter
            missing = {k for k, val in figures.items() if val is None and k not in items.optional}
        source = filings[j]
        out.append(YearFigures(fiscal_year=year, period_end=period_end, figures=figures,
                               filing=source.filing if source else FilingLink("", None)))
    warnings = [Warning("missing_figure", name) for name in SUMMARY_FIELDS if name in missing]
    return out, warnings
