"""GLEIF: the global LEI register, free and keyless (60 requests a minute).

Used to turn an LEI or an ISIN into the company behind it: its legal name,
its country and the number its home register gives it (``registeredAs``:
a UK company number, a Japanese corporate number). The sources then find the
company by those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from companies import http

SOURCE = "gleif"
API = "https://api.gleif.org/api/v1/lei-records"


@dataclass(frozen=True)
class LeiRecord:
    lei: str
    name: str
    country: Optional[str]
    registered_as: Optional[str]


def _records(data: dict) -> list[LeiRecord]:
    out = []
    for row in (data or {}).get("data") or []:
        attrs = row.get("attributes") or {}
        entity = attrs.get("entity") or {}
        out.append(LeiRecord(
            lei=attrs.get("lei") or row.get("id", ""),
            name=((entity.get("legalName") or {}).get("name") or "").strip(),
            country=(entity.get("legalAddress") or {}).get("country"),
            registered_as=(entity.get("registeredAs") or None)))
    return out


def by_lei(lei: str) -> Optional[LeiRecord]:
    data = http.get_json(SOURCE, f"{API}/{lei.strip().upper()}", allow_404=True)
    if not data:
        return None
    found = _records({"data": [data.get("data")]} if data.get("data") else {})
    return found[0] if found else None


def by_isin(isin: str) -> list[LeiRecord]:
    return _records(http.get_json(SOURCE, API, params={"filter[isin]": isin.strip().upper()}))
