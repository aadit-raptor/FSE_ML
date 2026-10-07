"""Year-by-year history the risk ranges are measured on (PLAN.md 4.4).

Two kinds, both stored as compact tables beside the current averages
(db/benchmarks.py) by the scheduled ``benchmarks-refresh``:

- **Industry history**, ``history.<group>``: each industry's EV/EBITDA, gross
  margin and EBITDA margin, year by year, from **Damodaran's archive** of
  his past January editions. The archive's file ``margin<group>16.xls`` is
  the January 2017 edition, so it describes **2016**; the current edition
  (January 2026, already stored as ``margins.<group>``) describes 2025, and
  ``industry_years`` joins the two. The archive goes back to the 2011 data
  for every group; older editions print fewer columns (gross margin only
  since 2017, EBITDA/sales since 2013), and a few years are missing or
  unreadable for some groups, so each figure has the years it has.
  Archived editions never change, so a year is read once: ``history.read``
  records what each file gave, and only a recent file still missing (not
  yet archived) is asked again.
- **Economic history**, ``history.macro``: real GDP growth and inflation by
  year from the IMF's World Economic Outlook, and each central bank's policy
  rate averaged over each year from the BIS's monthly series, for the 24
  economies and the euro area of economy/catalogue.py, from ``WINDOW_START``
  to the last full year.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date
from typing import Callable, Mapping, Optional

from benchmarks.catalogue import REGIONS, Dataset, Region
from benchmarks.damodaran import SOURCE_NAME, Table, Unreadable, parse_industries
from companies import http
from economy.catalogue import AREAS, COUNTRIES, EURO_MEMBERS
from economy.connectors import BIS_API, IMF_API, IMF_INDICATORS, _number

ARCHIVE_BASE = "https://pages.stern.nyu.edu/~adamodar/pc/archives"
ARCHIVE_PAGE = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/dataarchived.html"
# The first year of industry history read: the first one archived for every group
FIRST_YEAR = 2011
# The economic history's first year. Earlier policy rates include the 1990s
# hyperinflations (Brazil's monthly rate ran in the thousands of per cent),
# which would swamp every spread measured on them
WINDOW_START = 2000
# Workbooks one refresh call reads at most: paced at the host's rate, a few
# seconds each, well inside the scheduler's two minutes a call. The task
# says ``more`` until the backfill is done (about 20 calls the first time)
BATCH = 12

HISTORY_DATASETS: Mapping[str, Dataset] = {d.id: d for d in (
    Dataset("margins", "margin", "Operating and net margins", {
        "grossmargin": "gross_margin",
        "ebitda/sales": "ebitda_margin",
    }),
    Dataset("multiples", "vebitda", "Enterprise value multiples", {
        "ev/ebitda": "ev_ebitda",
    }),
)}
METRICS = ("gross_margin", "ebitda_margin", "ev_ebitda")
READ_TABLE = "history.read"
MACRO_TABLE = "history.macro"
IMF_PAGE = "https://www.imf.org/external/datamapper/datasets/WEO"
BIS_PAGE = "https://data.bis.org/topics/CBPOL"


def table_name(group: str) -> str:
    return f"history.{group}"


def archive_url(dataset: Dataset, region: Region, year: int) -> str:
    return f"{ARCHIVE_BASE}/{dataset.stem}{region.suffix}{year % 100:02d}.xls"


def archive_years(today: date) -> range:
    """Years the archive holds: the current edition (January of this year)
    describes last year, so the newest archived one describes the year before."""
    return range(FIRST_YEAR, today.year - 1)


def read_key(dataset: str, group: str, year: int) -> str:
    return f"{dataset}.{group}.{year}"


def to_read(read: Mapping[str, dict], today: date, retry: bool = True) -> list[tuple[str, str, int]]:
    """The archive files not yet read, oldest first: (dataset, group, year).
    A file read before is final, except one of the newest archived year that
    was missing: it may not be archived yet, so it is asked again (unless
    ``retry`` is off: what is left of the backfill)."""
    newest = archive_years(today)[-1]
    out = []
    for year in archive_years(today):
        for group in REGIONS:
            for dataset in HISTORY_DATASETS:
                status = (read.get(read_key(dataset, group, year)) or {}).get("status")
                if status is None or (retry and status == "missing" and year == newest):
                    out.append((dataset, group, year))
    return out


def read_archive(dataset: str, group: str, year: int) -> tuple[str, dict]:
    """One archived workbook: its status (``ok``, ``missing``, ``unreadable``)
    and its rows by industry id."""
    d = HISTORY_DATASETS[dataset]
    resp = http.get(SOURCE_NAME, archive_url(d, REGIONS[group], year), allow_404=True)
    if resp is None:
        return "missing", {}
    try:
        _, rows = parse_industries(resp.content, d, every_column=False)
    except (Unreadable, IndexError, TypeError, ValueError):
        return "unreadable", {}
    return "ok", rows


def merge(history: dict, year: int, rows: Mapping[str, dict]) -> dict:
    """Industry rows of one edition merged into a group's history. Two data
    sets count their companies separately; a year keeps the smaller count,
    so a figure is never credited with more companies than it had."""
    for industry, row in rows.items():
        entry = history.setdefault(industry, {"name": row["name"], "years": {}})
        entry["name"] = row["name"] if year >= max(map(int, entry["years"]), default=0) else entry["name"]
        figures = entry["years"].setdefault(str(year), {})
        figures["firms"] = min(figures.get("firms", row["firms"]), row["firms"])
        figures.update({m: row[m] for m in METRICS if m in row})
    return history


def industry_years(tables: Mapping[str, Table], group: str, industry: str) -> dict[int, dict]:
    """An industry's figures by year in ``group``: the archive's years and the
    current edition's (its year is the one before it was published)."""
    years = {int(y): dict(f) for y, f in
             ((tables.get(table_name(group)).rows.get(industry) or {}).get("years", {}).items()
              if tables.get(table_name(group)) else ())}
    current = [tables.get(f"{d}.{group}") for d in ("margins", "multiples")]
    for table in current:
        row = table.rows.get(industry) if table and table.published else None
        if row:
            figures = years.setdefault(table.published.year - 1, {})
            figures["firms"] = min(figures.get("firms", row["firms"]), row["firms"])
            figures.update({m: row[m] for m in METRICS if m in row})
    return dict(sorted(years.items()))


def industries_in(tables: Mapping[str, Table], group: str) -> set[str]:
    stored = tables.get(table_name(group))
    current = tables.get(f"margins.{group}")
    return set(stored.rows if stored else ()) | set(current.rows if current else ())


# ---------------------------------------------------------------------------
# Economic history
# ---------------------------------------------------------------------------
def _imf(today: date) -> dict[str, dict]:
    by_iso3 = {AREAS[c].iso3: c for c in COUNTRIES}
    years = range(WINDOW_START, today.year)
    out: dict[str, dict] = {}
    for indicator, code in IMF_INDICATORS.items():
        name = {"gdp_growth": "real_growth", "inflation": "inflation"}[indicator]
        data = http.get_json("imf", f"{IMF_API}/{code}") or {}
        values = (data.get("values") or {}).get(code) or {}
        for iso3, country in by_iso3.items():
            row = values.get(iso3) or {}
            found = {str(y): v for y in years if (v := _number(row.get(str(y)))) is not None}
            if found:
                out.setdefault(country, {})[name] = found
    return out


def _bis(today: date) -> dict[str, dict]:
    """Each year's average of the monthly policy rate, for complete years
    only (a euro member takes the euro area's)."""
    areas = [c for c in AREAS if c not in EURO_MEMBERS]
    resp = http.get("bis", f"{BIS_API}/WS_CBPOL/M.{'+'.join(areas)}/all",
                    params={"startPeriod": f"{WINDOW_START - 1}-01", "format": "csv", "detail": "dataonly"})
    try:
        rows = list(csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error):
        raise http.SourceError("bis", "unreadable") from None
    months: dict[str, dict[str, list]] = {}
    for r in rows:
        area, period, value = r.get("REF_AREA", ""), r.get("TIME_PERIOD", ""), _number(r.get("OBS_VALUE"))
        if area in AREAS and value is not None and len(period) == 7 and period[:4].isdigit():
            months.setdefault(area, {}).setdefault(period[:4], []).append(value)
    return {area: {"policy_rate": {y: round(sum(v) / len(v), 4) for y, v in sorted(years.items())
                                   if len(v) == 12 and int(y) < today.year}}
            for area, years in months.items()}


def read_macro(today: date) -> Table:
    """The economic history table: ``{"GB": {"real_growth": {"2000": 4.3,
    ...}, "inflation": {...}, "policy_rate": {...}}, "XM": {...}}``."""
    rows: dict[str, dict] = {}
    for part in (_imf(today), _bis(today)):
        for area, series in part.items():
            rows.setdefault(area, {}).update(series)
    if not rows:
        raise Unreadable("no economic history")
    return Table(MACRO_TABLE, today, IMF_PAGE, rows)


def macro_series(tables: Mapping[str, Table], area: str, name: str) -> dict[int, float]:
    table = tables.get(MACRO_TABLE)
    found = ((table.rows.get(area) or {}).get(name) or {}) if table else {}
    return {int(y): v for y, v in sorted(found.items())}


def policy_area(country: str) -> str:
    """Whose policy rate a country has: a euro member's is the euro area's."""
    return "XM" if country in EURO_MEMBERS else country



# ---------------------------------------------------------------------------
# What a refresh reads
# ---------------------------------------------------------------------------
def fill(stored: Mapping[str, Table], today: date, log: Optional[Callable] = None,
         batch: int = BATCH) -> tuple[list[Table], dict]:
    """Read up to ``batch`` archive files not read yet, merged into their
    groups' history; the tables to store and a summary (``left``: files
    still to read after this call)."""
    read = dict(stored[READ_TABLE].rows) if READ_TABLE in stored else {}
    pending = to_read(read, today)
    groups: dict[str, dict] = {}
    problems: dict[str, str] = {}
    for dataset, group, year in pending[:batch]:
        key = read_key(dataset, group, year)
        try:
            status, rows = read_archive(dataset, group, year)
        except http.SourceError as exc:          # a failed call is asked again next time
            problems[key] = exc.reason
            if log:
                log("benchmark_history_failed", logging.WARNING, table=key, reason=exc.reason)
            continue
        read[key] = {"status": status, "industries": len(rows)}
        if rows:
            history = groups.setdefault(group, dict(stored[table_name(group)].rows)
                                        if table_name(group) in stored else {})
            merge(history, year, rows)
    tables = [Table(table_name(g), None, ARCHIVE_PAGE, rows) for g, rows in groups.items()]
    if read:
        tables.append(Table(READ_TABLE, None, ARCHIVE_PAGE, read))
    left = len(to_read(read, today, retry=False))
    return tables, {"left": left, "problems": problems}
