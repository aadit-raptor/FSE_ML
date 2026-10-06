"""SEC EDGAR: US filers' 10-K (US GAAP) and foreign filers' 20-F/40-F (IFRS).

Free and keyless; the SEC asks for at most 10 requests a second and a
User-Agent naming the caller (companies/http.py).

- Search: the SEC's list of tickers and names (``company_tickers.json``),
  kept in memory for a day.
- Figures: the company facts API, which holds every number the filer has
  tagged in every filing. Each fact names its filing (accession number), so
  each year links to the filing it came from.
"""
from __future__ import annotations

import time
from datetime import date
from typing import Optional

from companies import http
from companies.facts import Fact, main_currency, summarize
from companies.items import IFRS_MAP, US_GAAP_MAP
from companies.model import IFRS, US_GAAP, CompanyData, CompanyRef, FilingLink

SOURCE = "sec"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accn}-index.htm"
LIST_TTL_S = 24 * 3600

US_FORMS = ("10-K", "10-K/A")
FOREIGN_FORMS = ("20-F", "20-F/A", "40-F", "40-F/A", "10-K", "10-K/A")
US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
    "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split())

_tickers: dict = {"at": None, "rows": []}


def configured() -> bool:
    return True


def _cik(value: str) -> str:
    return str(int(value)).zfill(10)


def _ticker_rows() -> list[dict]:
    if _tickers["at"] is None or time.monotonic() - _tickers["at"] > LIST_TTL_S:
        data = http.get_json(SOURCE, TICKERS_URL) or {}
        _tickers["rows"] = [{"cik": _cik(r["cik_str"]), "ticker": r.get("ticker", ""), "name": r.get("title", "")}
                            for r in data.values()]
        _tickers["at"] = time.monotonic()
    return _tickers["rows"]


def reset_cache() -> None:
    _tickers.update(at=None, rows=[])


def _ref(row: dict) -> CompanyRef:
    ids = {"cik": row["cik"]}
    if row.get("ticker"):
        ids["ticker"] = row["ticker"]
    return CompanyRef(SOURCE, row["cik"], row["name"], None, ids)


def search(query: str, limit: int = 10) -> list[CompanyRef]:
    """By ticker (exact), CIK, or a part of the name."""
    q = query.strip().upper()
    rows = _ticker_rows()
    if q.isdigit():
        cik = _cik(q)
        return [_ref(r) for r in rows if r["cik"] == cik][:1]
    exact = [r for r in rows if r["ticker"].upper() == q]
    named = [r for r in rows if q in r["name"].upper() and r not in exact]
    seen, out = set(), []
    for r in exact + named:
        if r["cik"] not in seen:        # a company lists one row per share class
            seen.add(r["cik"])
            out.append(_ref(r))
    return out[:limit]


def search_by_name(name: str, limit: int = 3) -> list[CompanyRef]:
    """A legal name from another register (GLEIF): matched on the name
    without punctuation, so "Apple Inc." finds "Apple Inc."."""
    want = _plain(name)
    return [_ref(r) for r in _ticker_rows() if _plain(r["name"]) == want][:limit]


def _plain(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def facts_from_companyfacts(data: dict, cik: str) -> tuple[list[Fact], str, str]:
    """(facts, standard, currency): US GAAP from 10-Ks when the filer has
    them, else IFRS from its 20-F or 40-F in its statements' currency."""
    concepts = data.get("facts", {})
    us = _facts(concepts.get("us-gaap", {}), "us-gaap", US_FORMS, cik)
    if any(f.concept == "us-gaap:Assets" and f.currency == "USD" for f in us):
        return [f for f in us if f.currency == "USD"], US_GAAP, "USD"
    ifrs = _facts(concepts.get("ifrs-full", {}), "ifrs-full", FOREIGN_FORMS, cik)
    currency = main_currency(ifrs, ("ifrs-full:Assets", "ifrs-full:Revenue", "ifrs-full:EquityAndLiabilities"))
    if currency is None:
        return us, US_GAAP, "USD"
    return [f for f in ifrs if f.currency == currency], IFRS, currency


def _facts(namespace: dict, prefix: str, forms: tuple, cik: str) -> list[Fact]:
    out = []
    for concept, body in namespace.items():
        for unit, rows in body.get("units", {}).items():
            if len(unit) != 3 or not unit.isupper():
                continue
            for r in rows:
                if r.get("form") not in forms or r.get("fp") != "FY" or "end" not in r:
                    continue
                accn = r.get("accn", "")
                link = FilingLink(
                    url=FILING_URL.format(cik=int(cik), folder=accn.replace("-", ""), accn=accn) if accn else "",
                    filed_on=date.fromisoformat(r["filed"]) if r.get("filed") else None,
                    form=r.get("form", ""), id=accn)
                out.append(Fact(f"{prefix}:{concept}", float(r["val"]), unit, date.fromisoformat(r["end"]),
                                date.fromisoformat(r["start"]) if r.get("start") else None, link))
    return out


def _country(submissions: dict) -> Optional[str]:
    business = (submissions.get("addresses") or {}).get("business") or {}
    return "US" if business.get("stateOrCountry") in US_STATES else None


def fetch(source_id: str, n_years: int = 3) -> CompanyData:
    cik = _cik(source_id)
    submissions = http.get_json(SOURCE, SUBMISSIONS_URL.format(cik=cik))
    data = http.get_json(SOURCE, FACTS_URL.format(cik=cik))
    facts, standard, currency = facts_from_companyfacts(data, cik)
    years, warnings = summarize(facts, US_GAAP_MAP if standard == US_GAAP else IFRS_MAP, n_years)
    ids = {"cik": cik}
    tickers = submissions.get("tickers") or []
    if tickers:
        ids["ticker"] = tickers[0]
    fye = submissions.get("fiscalYearEnd") or ""
    month = int(fye[:2]) if len(fye) == 4 and fye.isdigit() else None
    if years and month is None:
        month = years[-1].period_end.month
    ref = CompanyRef(SOURCE, cik, submissions.get("name") or data.get("entityName", cik),
                     _country(submissions), ids)
    return CompanyData(ref, currency, standard, _year_end_month(years, month), years, warnings)


def _year_end_month(years, declared: Optional[int]) -> Optional[int]:
    """The month the latest fiscal year really ends in (a 52/53-week year
    ending in a month's first week counts as the month before)."""
    from companies.facts import fiscal_year_of
    if years:
        return fiscal_year_of(years[-1].period_end)[1]
    return declared
