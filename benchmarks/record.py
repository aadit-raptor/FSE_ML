"""Record Damodaran's real workbooks as test fixtures, trimmed (PLAN.md 4.3, 4.4).

    python -m benchmarks.record                 # writes tests/fixtures/benchmarks/damodaran
    python -m benchmarks.record --history       # and tests/fixtures/benchmarks/history

The same calls the scheduled refresh makes, through companies/record.py's
recording transport. Each workbook is written again (with ``xlwt``, a
development dependency) holding only what the reader uses: the "Date
updated" row, the header row and, for every industry, its name, its number
of companies and the columns read. That keeps 41 workbooks of about 70 KB
each to a few kilobytes, and recording a fixture again from itself gives the
same bytes (tested). The country tax workbook keeps every country row,
including the stale repeat at its end that the reader must skip.

The history case (PLAN.md 4.4) records every archived edition the refresh
reads, trimmed further to ``HISTORY_INDUSTRIES`` (a few industries every
group has, enough for the correlation panel), and the IMF's and the BIS's
history kept for the catalogue's economies from ``history.WINDOW_START``.
A file the archive doesn't have is not recorded: replayed, it is a 404, as
the archive answers. An edition the reader can't open is recorded as it
came (its first ``UNREADABLE_KEPT`` bytes), so the test sees the same
``unreadable``.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import damodaran, history
from benchmarks.catalogue import COUNTRY_TAX, DATASETS, REGIONS
from companies import http
from companies.record import INDEX, Recorder, _name
from economy import connectors
from economy.catalogue import AREAS, COUNTRIES

CASE = "damodaran"
HISTORY_CASE = "history"
UNREADABLE_KEPT = 4096
# Industries kept in the history fixture: in every group, most years
HISTORY_INDUSTRIES = frozenset({
    "Machinery", "Chemical (Basic)", "Steel", "Food Processing", "Auto Parts", "Building Materials",
    "Electrical Equipment", "Retail (General)", "Telecom. Services", "Utility (General)",
})


def _columns_for(url: str) -> list[str] | None:
    """The headers to keep for a workbook (None: the country tax one)."""
    stem = url.rsplit("/", 1)[-1].removesuffix(".xls")
    if stem == COUNTRY_TAX["stem"]:
        return None
    for d in DATASETS.values():
        suffix = stem[len(d.stem):]
        if stem.startswith(d.stem) and (suffix == "" or suffix[0].isupper() or suffix == "emerg"):
            return ["industryname", "numberoffirms", *d.columns]
    raise ValueError(f"unknown workbook {stem}")


def trim(url: str, content: bytes) -> bytes:
    import xlwt

    book = damodaran._book(content)
    keep = _columns_for(url)
    sheet, header_row = damodaran._find_sheet(book, "country" if keep is None else "industryname")
    headers = [damodaran._header(h) for h in sheet.row_values(header_row)]
    cols = [0, 1, 2] if keep is None else [headers.index(h) for h in keep]
    date_row = next(r for r in range(header_row) if damodaran._header(sheet.cell_value(r, 0)).startswith("dateupdated"))

    out = xlwt.Workbook()
    target = out.add_sheet(sheet.name)
    target.write(0, 0, sheet.cell_value(date_row, 0))
    target.write(0, 1, sheet.cell_value(date_row, 1))
    for i, r in enumerate(range(header_row, sheet.nrows), start=1):
        values = sheet.row_values(r)
        for j, c in enumerate(cols):
            if c < len(values) and values[c] != "":
                target.write(i, j, values[c])
    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue()


def trim_history(content: bytes, dataset: str) -> bytes:
    """An archived edition with only the history's columns and industries;
    one the reader can't open is kept as it came."""
    import xlwt

    try:
        book = damodaran._book(content)
        sheet, header_row = damodaran._find_sheet(book, "industryname")
    except damodaran.Unreadable:
        return content[:UNREADABLE_KEPT]
    headers = [damodaran._header(h) for h in sheet.row_values(header_row)]
    wanted = ["industryname", "numberoffirms", *history.HISTORY_DATASETS[dataset].columns]
    cols = [headers.index(h) for h in wanted if h in headers]
    out = xlwt.Workbook()
    target = out.add_sheet(sheet.name)
    target.write(0, 0, "Date updated:")
    row = 1
    for r in range(header_row, sheet.nrows):
        values = sheet.row_values(r)
        if r > header_row and str(values[0]).strip() not in HISTORY_INDUSTRIES:
            continue
        for j, c in enumerate(cols):
            if c < len(values) and values[c] != "":
                target.write(row, j, values[c])
        row += 1
    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue()


def trim_macro(req: http.Request, content: bytes) -> bytes:
    if req.url.startswith(connectors.IMF_API + "/"):
        data = json.loads(content)
        wanted = {AREAS[c].iso3 for c in COUNTRIES}
        values = {code: {iso3: {y: v for y, v in row.items() if y.isdigit() and int(y) >= history.WINDOW_START}
                         for iso3, row in (rows or {}).items() if iso3 in wanted and row}
                  for code, rows in (data.get("values") or {}).items()}
        return json.dumps({"values": values}, sort_keys=True).encode()
    return content


def record_history(out: Path, today, real: http.Transport = http._requests_transport) -> Path:
    directory = out / HISTORY_CASE
    directory.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(real, paced=real is http._requests_transport)
    http.use_transport(recorder, paced=recorder.paced)
    try:
        history.read_macro(today)
        for dataset, group, year in history.to_read({}, today):
            for attempt in range(3):            # a university server: a dropped call is tried again
                try:
                    history.read_archive(dataset, group, year)
                    break
                except http.SourceError:
                    if attempt == 2:
                        raise
                    time.sleep(10)
    finally:
        http.use_transport(None)
    for old in directory.iterdir():
        old.unlink()
    index = {}
    for key, (req, resp) in recorder.responses.items():
        if resp.status != 200:
            continue
        if req.url.startswith(history.ARCHIVE_BASE):
            dataset = next(d for d, ds in history.HISTORY_DATASETS.items()
                           if req.url.rsplit("/", 1)[-1].startswith(ds.stem))
            body, name, kind = trim_history(resp.content, dataset), _name(key, "")[:-4] + ".xls", "application/vnd.ms-excel"
        else:
            body, name, kind = trim_macro(req, resp.content), _name(key, resp.content_type or ""), resp.content_type
            if name.endswith(".bin") and "csv" in (kind or ""):
                name = name[:-4] + ".csv"
        (directory / name).write_bytes(body)
        index[key] = {"file": name, "status": resp.status, "type": kind}
    index["_recorded_on"] = today.isoformat()
    (directory / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return directory


def record(out: Path, real: http.Transport = http._requests_transport) -> Path:
    directory = out / CASE
    directory.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(real, paced=real is http._requests_transport)
    http.use_transport(recorder, paced=recorder.paced)
    try:
        for _, call in damodaran.every_table():
            call()
    finally:
        http.use_transport(None)
    for old in directory.iterdir():
        old.unlink()
    index = {}
    for key, (req, resp) in recorder.responses.items():
        body = trim(req.url, resp.content) if resp.status == 200 else resp.content
        name = _name(key, "")[:-4] + ".xls"
        (directory / name).write_bytes(body)
        index[key] = {"file": name, "status": resp.status, "type": "application/vnd.ms-excel"}
    index["_recorded_on"] = datetime.now(timezone.utc).date().isoformat()
    (directory / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return directory


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="tests/fixtures/benchmarks")
    ap.add_argument("--history", action="store_true", help="record the history case (PLAN.md 4.4) instead")
    args = ap.parse_args(argv)
    if args.history:
        today = datetime.now(timezone.utc).date()
        print(f"{HISTORY_CASE}: {record_history(Path(args.out), today)}")
    else:
        print(f"{CASE}: {record(Path(args.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
