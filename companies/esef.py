"""ESEF annual reports, as collected by filings.xbrl.org.

EU-listed companies (and UK-listed ones, which file the same way) publish
their annual report as inline XBRL under the European Single Electronic
Format: IFRS consolidated statements tagged with the ``ifrs-full``
taxonomy. filings.xbrl.org, run by XBRL International, gathers them, keyed
by the issuer's LEI, and publishes each as xBRL-JSON. Free and keyless; it
asks for restraint, so calls are paced (companies/http.py).

Each report holds the year and the one before, so ``fetch`` reads the newest
``n_years - 1`` reports. ``filed_on`` is the day filings.xbrl.org added the
report (it publishes no filing date); the link opens its viewer.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Optional

from companies import http
from companies.facts import Fact, fiscal_year_of, main_currency, summarize
from companies.items import IFRS_MAP, canonical_prefix
from companies.model import IFRS, CompanyData, CompanyRef, FilingLink

SOURCE = "esef"
BASE = "https://filings.xbrl.org"
ENTITIES_URL = f"{BASE}/api/entities"
FILINGS_URL = f"{BASE}/api/entities/{{lei}}/filings"
FORM = "ESEF"
MAX_REPORTS = 4
# A fact's own dimensions; any other (a segment, a member) is not the group total
PLAIN_DIMENSIONS = frozenset({"concept", "entity", "period", "unit", "language"})


def configured() -> bool:
    return True


def _filter(name: str, op: str, value: str) -> str:
    return json.dumps([{"name": name, "op": op, "val": value}])


def _refs(data: dict, limit: int) -> list[CompanyRef]:
    out = []
    for row in (data or {}).get("data", [])[:limit]:
        attrs = row.get("attributes", {})
        lei = attrs.get("identifier", "")
        if lei:
            out.append(CompanyRef(SOURCE, lei, attrs.get("name") or lei, None, {"lei": lei}))
    return out


def search(query: str, limit: int = 10) -> list[CompanyRef]:
    """By a part of the name."""
    q = query.strip().replace("%", "").replace("_", " ")
    if len(q) < 2:
        return []
    data = http.get_json(SOURCE, ENTITIES_URL, params={
        "filter": _filter("name", "ilike", f"%{q}%"), "page[size]": str(limit)})
    return _refs(data, limit)


def by_lei(lei: str) -> list[CompanyRef]:
    data = http.get_json(SOURCE, ENTITIES_URL, params={"filter": _filter("identifier", "eq", lei)})
    return _refs(data, 1)


def _reports(lei: str) -> list[dict]:
    """The newest report per period end, newest period first."""
    data = http.get_json(SOURCE, FILINGS_URL.format(lei=lei), params={"page[size]": "100"}, allow_404=True)
    best: dict[str, dict] = {}
    for row in (data or {}).get("data", []):
        a = row.get("attributes", {})
        # Paths on filings.xbrl.org only: a value that names another host is refused
        if not str(a.get("json_url") or "").startswith("/") or not a.get("period_end"):
            continue
        key = a["period_end"]
        held = best.get(key)
        rank = ((a.get("error_count") or 0) == 0, a.get("date_added") or "")
        if held is None or rank > ((held.get("error_count") or 0) == 0, held.get("date_added") or ""):
            best[key] = {**a, "id": row.get("id", "")}
    return [best[k] for k in sorted(best, reverse=True)]


def _date(text: str, *, instant_end: bool) -> date:
    """An xBRL-JSON date-time. Periods end at midnight *after* their last
    day, so a date-time at 00:00:00 belongs to the day before."""
    moment = datetime.fromisoformat(text)
    if instant_end and moment.time() == datetime.min.time() and "T" in text:
        return moment.date() - timedelta(days=1)
    return moment.date()


def facts_from_xbrl_json(doc: dict, link: FilingLink) -> list[Fact]:
    """The plain (undimensioned) ifrs-full money facts of one report."""
    prefixes = {p: canonical_prefix(uri) for p, uri in
                (doc.get("documentInfo", {}).get("namespaces") or {}).items()}
    out = []
    for fact in (doc.get("facts") or {}).values():
        dims = fact.get("dimensions") or {}
        if set(dims) - PLAIN_DIMENSIONS:
            continue
        prefix, _, name = (dims.get("concept") or "").partition(":")
        unit = dims.get("unit") or ""
        if prefixes.get(prefix) != "ifrs-full" or not unit.startswith("iso4217:") or "/" in unit:
            continue
        try:
            value = float(fact.get("value"))
        except (TypeError, ValueError):
            continue
        period = dims.get("period") or ""
        if "/" in period:
            start_text, end_text = period.split("/", 1)
            start, end = _date(start_text, instant_end=False), _date(end_text, instant_end=True)
        elif period:
            start, end = None, _date(period, instant_end=True)
        else:
            continue
        out.append(Fact(f"ifrs-full:{name}", value, unit.split(":", 1)[1], end, start, link))
    return out


def fetch(source_id: str, n_years: int = 3) -> CompanyData:
    lei = source_id.strip().upper()
    reports = _reports(lei)
    if not reports:
        raise http.NotFound(SOURCE)
    facts: list[Fact] = []
    country: Optional[str] = None
    for report in reports[:max(1, min(MAX_REPORTS, n_years - 1))]:
        page = report.get("viewer_url") or report.get("report_url") or ""
        link = FilingLink(url=BASE + page if page.startswith("/") else "",
                          filed_on=date.fromisoformat(report["date_added"][:10]) if report.get("date_added") else None,
                          form=FORM, id=report.get("fxo_id") or str(report.get("id", "")))
        doc = http.get_json(SOURCE, BASE + report["json_url"])
        facts += facts_from_xbrl_json(doc or {}, link)
        country = country or report.get("country")
    currency = main_currency(facts, ("ifrs-full:Assets", "ifrs-full:Revenue", "ifrs-full:EquityAndLiabilities"))
    if currency is None:                   # no statement totals to say what the money is in
        raise http.SourceError(SOURCE, "unreadable")
    facts = [f for f in facts if f.currency == currency]
    years, warnings = summarize(facts, IFRS_MAP, n_years)
    try:                                   # the name is a nicety: never fail a load over it
        names = by_lei(lei)
    except http.SourceError:
        names = []
    name = names[0].name if names else lei
    month = fiscal_year_of(years[-1].period_end)[1] if years else None
    ref = CompanyRef(SOURCE, lei, name, country, {"lei": lei})
    return CompanyData(ref, currency, IFRS, month, years, warnings)
