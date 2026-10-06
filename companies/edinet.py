"""Japan's EDINET: annual securities reports (有価証券報告書).

Free, with a key (``EDINET_API_KEY``). EDINET takes the key as a query
parameter, so it is marked secret: never recorded, never in a message
(companies/http.py). Calls are paced at one a second.

- **Search** reads EDINET's public code list (every filer's EDINET code,
  Japanese and English name, securities code and corporate number), kept in
  memory for a day. Individuals who file (large shareholders) are left out.
- **Finding a company's reports**: EDINET lists documents by the day they
  were filed, not by company. The scheduled refresh (companies/refresh.py)
  reads each day's list once and keeps the annual reports in an index
  (``ReportIndex``: a table in the database, or memory without one), so a
  load is two calls, not a search through a year of days.
- **Figures** come from each report's XBRL-to-CSV file: the consolidated,
  undimensioned facts of the current and prior year (the contexts EDINET
  names ``CurrentYear...`` and ``Prior1Year...``). The report says which
  standard it follows (``AccountingStandardsDEI``): Japanese GAAP
  (``jppfs``), IFRS (``jpigp``) or US GAAP.
"""
from __future__ import annotations

import csv
import io
import os
import re
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol

from companies import http
from companies.facts import Fact, fiscal_year_of, main_currency, summarize
from companies.items import JPIGP_MAP, JPPFS_MAP
from companies.model import IFRS, JGAAP, US_GAAP, CompanyData, CompanyRef, FilingLink, Warning

SOURCE = "edinet"
KEY_ENV = "EDINET_API_KEY"
KEY_PARAM = "Subscription-Key"
API = "https://api.edinet-fsa.go.jp/api/v2"
CODE_LIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
PDF_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf/{doc_id}.pdf"
FORM = "有価証券報告書"
LIST_TTL_S = 24 * 3600
MAX_REPORTS = 4
# Annual securities reports and their amendments, under the Cabinet Office
# ordinance on disclosure of corporate affairs
ANNUAL_REPORT_TYPES = frozenset({"120", "130"})
DISCLOSURE_ORDINANCE = "010"
INDIVIDUAL = "個人"
STANDARD_DEI = {"Japan GAAP": JGAAP, "IFRS": IFRS, "US GAAP": US_GAAP}
PREFIXES = {"jppfs_cor": "jppfs", "jpigp_cor": "jpigp", "jpcrp_cor": "jpcrp", "jpdei_cor": "jpdei"}
# Consolidated, undimensioned contexts: CurrentYearDuration, Prior1YearInstant ...
CONTEXT = re.compile(r"^(CurrentYear|Prior([1-4])Year)(Duration|Instant)$")


@dataclass(frozen=True)
class IndexedReport:
    edinet_code: str
    doc_id: str
    period_start: Optional[date]
    period_end: date
    submitted_at: datetime


class ReportIndex(Protocol):
    def reports(self, edinet_code: str) -> list[IndexedReport]: ...      # newest period first
    def add(self, reports: list[IndexedReport]) -> int: ...
    def scanned_through(self) -> Optional[date]: ...
    def set_scanned_through(self, day: date) -> None: ...
    def prune(self, today: date) -> int: ...


class MemoryIndex:
    """The index for a process without a database."""

    def __init__(self):
        self._rows: dict[tuple[str, date], IndexedReport] = {}
        self._through: Optional[date] = None

    def reports(self, edinet_code: str) -> list[IndexedReport]:
        rows = [r for (code, _), r in self._rows.items() if code == edinet_code]
        return sorted(rows, key=lambda r: r.period_end, reverse=True)

    def add(self, reports: list[IndexedReport]) -> int:
        added = 0
        for r in reports:
            held = self._rows.get((r.edinet_code, r.period_end))
            if held is None or r.submitted_at > held.submitted_at:
                self._rows[(r.edinet_code, r.period_end)] = r
                added += 1
        return added

    def scanned_through(self) -> Optional[date]:
        return self._through

    def set_scanned_through(self, day: date) -> None:
        self._through = day

    def prune(self, today: date) -> int:
        return 0


_index: list = [MemoryIndex()]
_codes: dict = {"at": None, "rows": []}


def use_index(index: Optional[ReportIndex]) -> None:
    _index[0] = index or MemoryIndex()


def index() -> ReportIndex:
    return _index[0]


def _key() -> Optional[str]:
    return os.environ.get(KEY_ENV) or None


def configured() -> bool:
    return _key() is not None


def _api(path: str, **params) -> http.Response:
    key = _key()
    if key is None:
        raise http.NotConfigured(SOURCE)
    return http.get(SOURCE, f"{API}/{path}", params={**params, KEY_PARAM: key}, secret_params=(KEY_PARAM,))


# ---------------------------------------------------------------------------
# The code list and search
# ---------------------------------------------------------------------------
def _plain(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").upper()


def parse_code_list(content: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        text = z.read(z.namelist()[0]).decode("cp932")
    lines = text.splitlines()[1:]          # the first line says when it was made
    rows = []
    for r in csv.reader(lines[1:]):
        if len(r) < 13 or r[1].startswith(INDIVIDUAL):
            continue
        rows.append({"edinet_code": r[0], "kind": r[1], "listed": r[2] == "上場", "year_end": r[5],
                     "name": r[6], "name_en": r[7], "sec_code": r[11], "jcn": r[12]})
    return rows


def _code_rows() -> list[dict]:
    if _codes["at"] is None or time.monotonic() - _codes["at"] > LIST_TTL_S:
        resp = http.get(SOURCE, CODE_LIST_URL)
        try:
            _codes["rows"] = parse_code_list(resp.content)
        except (zipfile.BadZipFile, UnicodeDecodeError, IndexError):
            raise http.SourceError(SOURCE, "unreadable") from None
        _codes["at"] = time.monotonic()
    return _codes["rows"]


def reset_cache() -> None:
    _codes.update(at=None, rows=[])


def _ref(row: dict) -> CompanyRef:
    ids = {"edinet_code": row["edinet_code"]}
    if row["sec_code"]:
        ids["sec_code"] = row["sec_code"][:4]
        ids["ticker"] = row["sec_code"][:4]
    if row["jcn"]:
        ids["jcn"] = row["jcn"]
    country = "JP" if row["kind"].startswith("内国") else None
    return CompanyRef(SOURCE, row["edinet_code"], row["name_en"] or row["name"], country, ids,
                      local_name=row["name"] if row["name_en"] else None)


def search(query: str, limit: int = 10) -> list[CompanyRef]:
    """By EDINET code, securities code (``7203``), corporate number or a
    part of the Japanese or English name; listed companies first."""
    q = _plain(query.strip())
    rows = _code_rows()
    if re.fullmatch(r"E\d{5}", q):
        found = [r for r in rows if r["edinet_code"] == q]
    elif re.fullmatch(r"\d{13}", q):
        found = [r for r in rows if r["jcn"] == q]
    elif re.fullmatch(r"\d{3}[0-9A-Z]", q):
        found = [r for r in rows if r["sec_code"][:4] == q]
    elif len(q) >= 2:
        found = [r for r in rows if q in _plain(r["name"]) or q in _plain(r["name_en"])]
        found.sort(key=lambda r: not r["listed"])
    else:
        found = []
    return [_ref(r) for r in found[:limit]]


def by_jcn(jcn: str) -> list[CompanyRef]:
    return search(jcn, 1) if re.fullmatch(r"\d{13}", jcn or "") else []


# ---------------------------------------------------------------------------
# The index of annual reports
# ---------------------------------------------------------------------------
def _date(text: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(text[:10]) if text else None
    except ValueError:
        return None


def annual_reports(listing: dict) -> list[IndexedReport]:
    """The annual reports in one day's document list."""
    out = []
    for r in (listing or {}).get("results") or []:
        if (r.get("docTypeCode") not in ANNUAL_REPORT_TYPES or r.get("ordinanceCode") != DISCLOSURE_ORDINANCE
                or r.get("csvFlag") != "1" or r.get("withdrawalStatus") not in (None, "0")
                or not r.get("edinetCode") or not _date(r.get("periodEnd"))):
            continue
        out.append(IndexedReport(r["edinetCode"], r["docID"], _date(r.get("periodStart")),
                                 _date(r["periodEnd"]),
                                 datetime.fromisoformat(r.get("submitDateTime") or f"{r['periodEnd']} 00:00")))
    return out


def scan_day(day: date) -> int:
    """Add one day's annual reports to the index; how many were new."""
    resp = _api("documents.json", date=day.isoformat(), type="2")
    try:
        import json
        listing = json.loads(resp.content)
    except ValueError:
        raise http.SourceError(SOURCE, "unreadable") from None
    return index().add(annual_reports(listing))


# ---------------------------------------------------------------------------
# Reading a report
# ---------------------------------------------------------------------------
def _shift(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:                     # 29 February
        return day.replace(year=day.year - years, day=28)


def read_csv(content: bytes, report: IndexedReport, link: FilingLink) -> tuple[list[Fact], dict[str, str]]:
    """(money facts, DEI labels) from a report's XBRL-to-CSV archive."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise http.SourceError(SOURCE, "unreadable") from None
    facts, labels = [], {}
    names = [n for n in archive.namelist() if re.search(r"XBRL_TO_CSV/jpcrp[^/]*\.csv$", n)]
    for name in names:
        text = archive.read(name).decode("utf-16")
        for row in csv.reader(io.StringIO(text), delimiter="\t"):
            if len(row) < 9 or ":" not in row[0]:
                continue
            element, context, unit, value = row[0], row[2], row[6], row[8]
            prefix, _, local = element.partition(":")
            canonical = PREFIXES.get(prefix)
            if canonical is None:
                continue
            if canonical == "jpdei":
                labels.setdefault(f"jpdei:{local}", value)
                continue
            m = CONTEXT.match(context)
            if m is None or not re.fullmatch(r"[A-Z]{3}", unit or ""):
                continue
            try:
                amount = float(value)
            except ValueError:
                continue
            back = int(m.group(2) or 0)
            end = _shift(report.period_end, back)
            start = None
            if m.group(3) == "Duration":
                start = _shift(report.period_start, back) if report.period_start else _shift(end, 1)
            facts.append(Fact(f"{canonical}:{local}", amount, unit, end, start, link))
    return facts, labels


def fetch(source_id: str, n_years: int = 3) -> CompanyData:
    code = source_id.strip().upper()
    found = search(code, 1)
    if not found:
        raise http.NotFound(SOURCE)
    ref = found[0]
    reports = index().reports(code)
    if not reports:
        return CompanyData(ref, "JPY", JGAAP, None, [], [Warning("not_indexed_yet")])
    facts: list[Fact] = []
    labels: dict[str, str] = {}
    for report in reports[:max(1, min(MAX_REPORTS, n_years - 1))]:
        link = FilingLink(PDF_URL.format(doc_id=report.doc_id), report.submitted_at.date(), FORM, report.doc_id)
        resp = _api(f"documents/{report.doc_id}", type="5")
        found_facts, found_labels = read_csv(resp.content, report, link)
        facts += found_facts
        for k, v in found_labels.items():
            labels.setdefault(k, v)
    standard = STANDARD_DEI.get(labels.get("jpdei:AccountingStandardsDEI", ""), JGAAP)
    currency = main_currency(facts, [f.concept for f in facts]) or "JPY"
    facts = [f for f in facts if f.currency == currency]
    years, warnings = summarize(facts, JPIGP_MAP if standard == IFRS else JPPFS_MAP, n_years)
    if standard == US_GAAP:
        warnings.append(Warning("summary_only"))
    month = fiscal_year_of(years[-1].period_end)[1] if years else None
    return CompanyData(ref, currency, standard, month, years, warnings)
