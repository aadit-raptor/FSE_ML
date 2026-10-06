"""The one interface to company data, whatever the country.

    sources.search("Tesco")                -> companies from every source
    sources.search("2138002P5RNKC5W2JZ46") -> the company with that LEI
    sources.search("GB00BLGZ9862")         -> the company with that ISIN
    sources.fetch("esef", "2138002P5RNKC5W2JZ46") -> its summary figures

A query is read as an ISIN or an LEI when its check digits say so: GLEIF
gives the company's name, country and home-register number, and each source
is asked by those. Anything else goes to every source as a name, ticker or
number. A source that fails or has no key doesn't fail the search: it is
listed in ``unavailable`` with a reason code.

For a company no source covers, the answer is document upload (PLAN.md 6.2):
``FALLBACK`` names it, and the screen offers it.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Optional

from companies import companies_house, edinet, esef, gleif, http, identifiers, sec
from companies.model import CompanyData, CompanyRef

FALLBACK = "document_upload"
MAX_YEARS = 5
DEFAULT_YEARS = 3


@dataclass(frozen=True)
class Source:
    id: str
    module: ModuleType
    # Where its companies are, as ISO country codes or "EU" (any member state)
    coverage: tuple
    # The identifiers it can be searched by
    searchable_by: tuple
    # The environment variable holding its key, if it needs one
    key_env: Optional[str]
    licence: str
    id_pattern: re.Pattern

    def configured(self) -> bool:
        return self.module.configured()


SOURCES: dict[str, Source] = {s.id: s for s in (
    Source("sec", sec, ("US", "*"), ("name", "ticker", "cik", "lei", "isin"), None,
           "U.S. public domain (SEC EDGAR)", re.compile(r"^[0-9]{1,10}\Z")),
    Source("esef", esef, ("EU", "GB", "NO", "IS", "LI", "UA"), ("name", "lei", "isin"), None,
           "filings.xbrl.org, XBRL International (issuers' public reports)", re.compile(r"^[A-Z0-9]{20}\Z")),
    Source("companies_house", companies_house, ("GB",), ("name", "company_number", "lei", "isin"),
           companies_house.KEY_ENV, "UK Open Government Licence (Companies House)",
           re.compile(r"^[A-Z0-9]{8}\Z")),
    Source("edinet", edinet, ("JP",), ("name", "ticker", "edinet_code", "jcn", "lei", "isin"), edinet.KEY_ENV,
           "Financial Services Agency of Japan, EDINET (public disclosure)", re.compile(r"^E[0-9]{5}\Z")),
)}


class UnknownSource(ValueError):
    pass


@dataclass
class SearchResult:
    results: list = field(default_factory=list)            # CompanyRef
    unavailable: list = field(default_factory=list)        # (source, reason)
    matched: Optional[str] = None                          # "lei", "isin" or None


def _try(out: SearchResult, source: str, call: Callable[[], list], limit: int,
         extra_ids: Optional[dict] = None) -> None:
    try:
        found = call()
    except http.SourceError as exc:
        out.unavailable.append((source, exc.reason))
        return
    except Exception:  # noqa: BLE001 - a source that changed its format must not break the search
        out.unavailable.append((source, "unreadable"))
        return
    have = {(r.source, r.source_id) for r in out.results}
    for ref in found[:limit]:
        if (ref.source, ref.source_id) in have:
            continue
        if extra_ids:
            ref = CompanyRef(ref.source, ref.source_id, ref.name, ref.country,
                             {**ref.identifiers, **extra_ids}, ref.local_name)
        out.results.append(ref)


def _by_lei_record(out: SearchResult, rec: gleif.LeiRecord, extra: dict, limit: int) -> None:
    ids = {"lei": rec.lei, **extra}
    _try(out, "esef", lambda: esef.by_lei(rec.lei), limit, ids)
    if rec.country == "GB" and rec.registered_as and identifiers.is_uk_company_number(rec.registered_as):
        if companies_house.configured():
            _try(out, "companies_house", lambda: companies_house.by_number(rec.registered_as), limit, ids)
        else:
            out.unavailable.append(("companies_house", "not_configured"))
    if rec.country == "JP" and rec.registered_as:
        _try(out, "edinet", lambda: edinet.by_jcn(rec.registered_as), limit, ids)
    if rec.name:
        _try(out, "sec", lambda: sec.search_by_name(rec.name), limit, ids)


def search(query: str, limit: int = 10) -> SearchResult:
    q = query.strip()
    if not 2 <= len(q) <= 100:
        raise ValueError("A search needs 2 to 100 characters.")
    qu = q.upper()
    out = SearchResult()
    if identifiers.is_isin(qu):
        out.matched = "isin"
        try:
            records = gleif.by_isin(qu)
        except http.SourceError as exc:
            out.unavailable.append(("gleif", exc.reason))
            records = []
        for rec in records:
            _by_lei_record(out, rec, {"isin": qu}, limit)
        return out
    if identifiers.is_lei(qu):
        out.matched = "lei"
        try:
            rec = gleif.by_lei(qu)
        except http.SourceError as exc:
            out.unavailable.append(("gleif", exc.reason))
            rec = None
        _by_lei_record(out, rec or gleif.LeiRecord(qu, "", None, None), {}, limit)
        return out
    _try(out, "sec", lambda: sec.search(q, limit), limit)
    _try(out, "esef", lambda: esef.search(q, limit), limit)
    if companies_house.configured():
        if identifiers.is_uk_company_number(qu):
            _try(out, "companies_house", lambda: companies_house.by_number(qu), limit)
        else:
            _try(out, "companies_house", lambda: companies_house.search(q, limit), limit)
    else:
        out.unavailable.append(("companies_house", "not_configured"))
    _try(out, "edinet", lambda: edinet.search(q, limit), limit)
    return out


def source(source_id: str) -> Source:
    found = SOURCES.get(source_id)
    if found is None:
        raise UnknownSource(f"No source named {source_id!r}. Sources: {', '.join(SOURCES)}.")
    return found


def normalise_id(source_id: str, company_id: str) -> str:
    """The company's id as its source writes it, or ValueError: ids go into
    the sources' URLs, so only their own shapes are accepted."""
    s = source(source_id)
    v = company_id.strip().upper()
    if source_id == "companies_house":
        v = companies_house.normalise_number(v)
    if not s.id_pattern.match(v):
        raise ValueError(f"That is not a {source_id} company id.")
    return v


def fetch(source_id: str, company_id: str, n_years: int = DEFAULT_YEARS) -> CompanyData:
    s = source(source_id)
    company_id = normalise_id(source_id, company_id)
    if not s.configured():
        raise http.NotConfigured(source_id)
    with _loads:
        return s.module.fetch(company_id, max(2, min(MAX_YEARS, n_years)))


# A load holds a filing or two in memory (tens of MB for a large one): at
# most two at once on the free 512 MB
_loads = threading.BoundedSemaphore(2)
