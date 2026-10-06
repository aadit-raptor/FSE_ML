"""Reading Damodaran's workbooks into compact tables (PLAN.md 4.3).

Each industry-average workbook has a sheet whose first column holds a row
starting "Industry Name" (row 8 or 9, depending on the file); the rows below
it are one industry each, then the totals. The date the data was updated is
an Excel serial day beside "Date updated:" in the first row. The sheet's
name differs from file to file, so the sheet is found by that header row.

The workbooks are untrusted input: a value that isn't a finite number is
left out (Damodaran prints "NA" where a group has none), and a workbook
with no header row is ``Unreadable``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping, Optional

from benchmarks.catalogue import (
    ALL_INDUSTRIES, ALL_INDUSTRIES_ID, COUNTRY_TAX, DATASETS, FINANCIAL_INDUSTRIES, REGIONS, TOTAL_ROWS,
    Dataset, Region, workbook_url,
)
from benchmarks.countries import NAME_TO_ISO
from companies import http

SOURCE_NAME = "damodaran"
EXCEL_EPOCH = date(1899, 12, 30)
DECIMALS = 6
MAX_HEADER_ROW = 20


class Unreadable(ValueError):
    """A workbook of a shape this reader doesn't know."""


@dataclass(frozen=True)
class Table:
    """One stored table: a data set for one region (``margins.europe``), or
    the country tax rates (``country_tax``). ``rows`` maps an industry id
    (or an ISO country code) to its figures."""

    name: str
    published: Optional[date]
    url: str
    rows: Mapping[str, dict] = field(default_factory=dict)


def industry_id(name: str) -> str:
    """``Oil/Gas (Integrated)`` -> ``oil_gas_integrated``; the all-industry row is ``all``."""
    if name == ALL_INDUSTRIES:
        return ALL_INDUSTRIES_ID
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return "_".join(part for part in out.split("_") if part)


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), DECIMALS) if math.isfinite(value) else None


def _header(text) -> str:
    return "".join(str(text).split()).lower()


def _book(content: bytes):
    import xlrd
    try:
        return xlrd.open_workbook(file_contents=content)
    except Exception as exc:        # xlrd raises its own errors and struct/compound-file ones
        raise Unreadable("not a workbook") from exc


def _published(sheet) -> Optional[date]:
    for r in range(min(sheet.nrows, MAX_HEADER_ROW)):
        if _header(sheet.cell_value(r, 0)).startswith("dateupdated") and sheet.ncols > 1:
            serial = _number(sheet.cell_value(r, 1))
            return EXCEL_EPOCH + timedelta(days=int(serial)) if serial else None
    return None


def _find_sheet(book, first_header: str):
    for sheet in book.sheets():
        for r in range(min(sheet.nrows, MAX_HEADER_ROW)):
            if _header(sheet.cell_value(r, 0)) == first_header:
                return sheet, r
    raise Unreadable(f"no {first_header!r} row")


def parse_industries(content: bytes, dataset: Dataset) -> tuple[Optional[date], dict[str, dict]]:
    """The figures by industry id: ``{"machinery": {"name": "Machinery",
    "firms": 210, "gross_margin": 0.31, ...}}``, fractions as printed."""
    sheet, header_row = _find_sheet(_book(content), "industryname")
    headers = [_header(h) for h in sheet.row_values(header_row)]
    firms_col = next((i for i, h in enumerate(headers) if h == "numberoffirms"), None)
    if firms_col is None:
        raise Unreadable("no number of firms")
    columns = {}
    for metric_header, metric in dataset.columns.items():
        col = next((i for i, h in enumerate(headers) if h == metric_header), None)
        if col is None:
            raise Unreadable(f"no {metric_header!r} column")
        columns[metric] = col
    rows = {}
    for r in range(header_row + 1, sheet.nrows):
        values = sheet.row_values(r)
        name = str(values[0]).strip()
        firms = _number(values[firms_col]) if len(values) > firms_col else None
        if not name or firms is None or name in TOTAL_ROWS or name in FINANCIAL_INDUSTRIES:
            continue
        figures = {"name": name, "firms": int(firms)}
        for metric, col in columns.items():
            value = _number(values[col]) if col < len(values) else None
            if value is not None:
                figures[metric] = value
        rows[industry_id(name)] = figures
    if ALL_INDUSTRIES_ID not in rows:
        raise Unreadable("no all-industry row")
    return _published(sheet), rows


def parse_country_tax(content: bytes) -> tuple[Optional[date], dict[str, dict]]:
    """Each country's marginal corporate tax rate, ``{"DE": {"rate": 0.2993}}``.

    The workbook repeats the end of its list further down with older rates
    (the UK at 19%, before its 2023 rise), so a country's first row wins.
    A name this reader can't place (``benchmarks.countries``) is skipped.
    """
    sheet, header_row = _find_sheet(_book(content), "country")
    rows: dict[str, dict] = {}
    for r in range(header_row + 1, sheet.nrows):
        values = sheet.row_values(r)
        code = NAME_TO_ISO.get(str(values[0]).strip())
        rate = _number(values[1]) if len(values) > 1 else None
        if code and code not in rows and rate is not None and 0 <= rate < 1:
            rows[code] = {"rate": rate}
    if not rows:
        raise Unreadable("no country rates")
    return _published(sheet), rows


def fetch_table(dataset: Dataset, region: Region) -> Table:
    url = workbook_url(dataset.stem, region)
    published, rows = parse_industries(http.get(SOURCE_NAME, url).content, dataset)
    return Table(f"{dataset.id}.{region.id}", published, url, rows)


def fetch_country_tax() -> Table:
    url = workbook_url(COUNTRY_TAX["stem"])
    published, rows = parse_country_tax(http.get(SOURCE_NAME, url).content)
    return Table(COUNTRY_TAX["id"], published, url, rows)


def every_table() -> list[tuple[str, callable]]:
    """Each table a refresh reads, by name, with the call that reads it."""
    calls = [(f"{d.id}.{r.id}", (lambda d=d, r=r: fetch_table(d, r)))
             for d in DATASETS.values() for r in REGIONS.values()]
    return calls + [(COUNTRY_TAX["id"], fetch_country_tax)]
