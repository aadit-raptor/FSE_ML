"""UK Companies House: every UK company's register entry and filed accounts.

Free, with a key (``COMPANIES_HOUSE_API_KEY``, sent as the HTTP basic-auth
user name); the limit is 600 requests every five minutes, and calls are
paced well under it (companies/http.py).

Figures come from accounts filed electronically, which are inline XBRL
(companies/ixbrl.py). Accounts filed on paper are scanned PDFs with no
figures to read: such a company loads with its register details, no years
and a ``scanned_accounts`` warning, and the answer for it is document upload
(PLAN.md 6.2). Each set of accounts holds the year and the one before, so
``fetch`` reads the newest ``n_years - 1``.

The standard is what the accounts say they follow (the FRC taxonomy's
accounting-standards members): FRS 102 or FRS 101 is UK GAAP, IFRS is IFRS.
"""
from __future__ import annotations

import os
import re
from datetime import date
from typing import Optional

from companies import http, ixbrl
from companies.facts import Fact, fiscal_year_of, main_currency, summarize
from companies.items import FRC_MAP, IFRS_MAP
from companies.model import IFRS, UK_GAAP, CompanyData, CompanyRef, FilingLink, Warning

SOURCE = "companies_house"
KEY_ENV = "COMPANIES_HOUSE_API_KEY"
API = "https://api.company-information.service.gov.uk"
DOCUMENT_API = "https://document-api.company-information.service.gov.uk/document/{id}"
FILING_PAGE = ("https://find-and-update.company-information.service.gov.uk/company/{number}"
               "/filing-history/{transaction}/document?format=xhtml&download=0")
XHTML = "application/xhtml+xml"
MAX_ACCOUNTS = 4
IFRS_MEMBERS = frozenset({"InternationalReportingStandards", "IFRS", "EU-IFRS"})
UK_GAAP_MEMBERS = frozenset({"FRS102", "FRS101", "SmallEntities", "Micro-entities", "MicroEntities", "FRSSE"})


def _key() -> Optional[str]:
    return os.environ.get(KEY_ENV) or None


def configured() -> bool:
    return _key() is not None


def _get_json(url: str, **kwargs):
    key = _key()
    if key is None:
        raise http.NotConfigured(SOURCE)
    return http.get_json(SOURCE, url, auth=(key, ""), **kwargs)


def normalise_number(value: str) -> str:
    """A company number as the register writes it: eight characters, digits
    zero-padded (``445790`` is ``00445790``)."""
    v = value.strip().upper().replace(" ", "")
    return v.zfill(8) if v.isdigit() else v


def _ref(number: str, name: str) -> CompanyRef:
    return CompanyRef(SOURCE, number, name, "GB", {"company_number": number})


def search(query: str, limit: int = 10) -> list[CompanyRef]:
    data = _get_json(f"{API}/search/companies", params={"q": query.strip(), "items_per_page": str(limit)})
    return [_ref(i["company_number"], i.get("title") or i["company_number"])
            for i in (data or {}).get("items", []) if i.get("company_number")][:limit]


def by_number(number: str) -> list[CompanyRef]:
    profile = _get_json(f"{API}/company/{normalise_number(number)}", allow_404=True)
    if not profile:
        return []
    return [_ref(profile["company_number"], profile.get("company_name") or profile["company_number"])]


def _accounts(number: str) -> list[dict]:
    data = _get_json(f"{API}/company/{number}/filing-history",
                     params={"category": "accounts", "items_per_page": "25"}) or {}
    return [i for i in data.get("items", []) if i.get("type") == "AA" and
            (i.get("links") or {}).get("document_metadata")]


def _document(item: dict) -> Optional[bytes]:
    """The accounts as inline XBRL, or None when only a PDF was filed."""
    doc_id = item["links"]["document_metadata"].rstrip("/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", doc_id):
        return None
    meta = _get_json(DOCUMENT_API.format(id=doc_id)) or {}
    if XHTML not in (meta.get("resources") or {}):
        return None
    key = _key()
    resp = http.get(SOURCE, DOCUMENT_API.format(id=doc_id) + "/content",
                    headers={"Accept": XHTML}, auth=(key, ""))
    return resp.content if resp else None


def standard_of(members: set[str], facts: list[Fact]) -> str:
    if members & IFRS_MEMBERS:
        return IFRS
    if members & UK_GAAP_MEMBERS:
        return UK_GAAP
    return IFRS if any(f.concept.startswith("ifrs-full:") for f in facts) else UK_GAAP


def fetch(source_id: str, n_years: int = 3) -> CompanyData:
    number = normalise_number(source_id)
    profile = _get_json(f"{API}/company/{number}")
    facts: list[Fact] = []
    members: set[str] = set()
    read = 0
    for item in _accounts(number):
        if read >= max(1, min(MAX_ACCOUNTS, n_years - 1)):
            break
        content = _document(item)
        if content is None:
            continue
        link = FilingLink(url=FILING_PAGE.format(number=number, transaction=item.get("transaction_id", "")),
                          filed_on=date.fromisoformat(item["date"]) if item.get("date") else None,
                          form="AA", id=item.get("transaction_id", ""))
        try:
            found, found_members = ixbrl.read(content, link)
        except ixbrl.Unreadable:
            continue
        facts += found
        members |= found_members
        read += 1
    ref = _ref(number, profile.get("company_name") or number)
    reference = (profile.get("accounts") or {}).get("accounting_reference_date") or {}
    declared_month = int(reference["month"]) if str(reference.get("month", "")).isdigit() else None
    if not facts:
        return CompanyData(ref, "GBP", UK_GAAP, declared_month, [], [Warning("scanned_accounts")])
    standard = standard_of(members, facts)
    ifrs_tagged = any(f.concept.startswith("ifrs-full:") for f in facts)
    currency = main_currency(facts, [f.concept for f in facts]) or "GBP"
    facts = [f for f in facts if f.currency == currency]
    years, warnings = summarize(facts, IFRS_MAP if ifrs_tagged else FRC_MAP, n_years)
    month = fiscal_year_of(years[-1].period_end)[1] if years else declared_month
    return CompanyData(ref, currency, standard, month, years, warnings)
