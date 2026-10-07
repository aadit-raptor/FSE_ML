"""Record Damodaran's real workbooks as test fixtures, trimmed (PLAN.md 4.3).

    python -m benchmarks.record                 # writes tests/fixtures/benchmarks/damodaran

The same calls the scheduled refresh makes, through companies/record.py's
recording transport. Each workbook is written again (with ``xlwt``, a
development dependency) holding only what the reader uses: the "Date
updated" row, the header row and, for every industry, its name, its number
of companies and the columns read. That keeps 41 workbooks of about 70 KB
each to a few kilobytes, and recording a fixture again from itself gives the
same bytes (tested). The country tax workbook keeps every country row,
including the stale repeat at its end that the reader must skip.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import damodaran
from benchmarks.catalogue import COUNTRY_TAX, DATASETS
from companies import http
from companies.record import INDEX, Recorder, _name

CASE = "damodaran"


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
    args = ap.parse_args(argv)
    print(f"{CASE}: {record(Path(args.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
