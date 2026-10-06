"""Record real responses from the economic data sources as test fixtures.

    python -m economy.record --out tests/fixtures/economy economy_open
    python -m economy.record --list

The same calls the nightly refresh makes, through companies/record.py's
recording transport, trimmed to what the code reads: the IMF's DataMapper
answers every country since 1980 in one response, kept here for this
catalogue's countries and years only; the ECB's year of exchange rates is
kept for its last ``FX_DAYS_KEPT`` publication days.

``economy_fred`` needs ``FRED_API_KEY``: the ``record-economy`` workflow runs
it with the repository secret. Keys are never recorded (requests are named
without them) and the run fails if a key's value appears in anything written.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from companies import http
from companies.record import INDEX, Recorder, _name
from db.economy import KEEP_FX_DAYS
from economy import connectors
from economy.catalogue import AREAS, COUNTRIES

FX_DAYS_KEPT = 10


def trim(req: http.Request, content: bytes, today: date) -> bytes:
    if req.url.startswith(connectors.IMF_API + "/"):
        data = json.loads(content)
        years = {str(y) for y in range(today.year - connectors.IMF_YEARS_BACK,
                                       today.year + connectors.IMF_YEARS_AHEAD + 1)}
        wanted = {AREAS[c].iso3 for c in COUNTRIES}
        values = {code: {iso3: {y: v for y, v in row.items() if y in years}
                         for iso3, row in (rows or {}).items() if iso3 in wanted and row}
                  for code, rows in (data.get("values") or {}).items()}
        return json.dumps({"values": values}, sort_keys=True).encode()
    if req.url.endswith("/EXR/D..EUR.SP00.A"):
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        days = sorted({r["TIME_PERIOD"] for r in rows})[-FX_DAYS_KEPT:]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["KEY"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in rows if r["TIME_PERIOD"] in days)
        return buf.getvalue().encode()
    return content


@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[[date], None]
    needs: tuple = ()


def _open(today: date) -> None:
    connectors.imf(today)
    connectors.worldbank()
    connectors.bis()
    connectors.oecd()
    connectors.ecb()
    connectors.ecb_fx(today - timedelta(days=KEEP_FX_DAYS))


CASES = {c.name: c for c in (
    Case("economy_open", _open),
    Case("economy_fred", lambda today: connectors.fred(), (connectors.FRED_KEY_ENV,)),
)}


def record(case: Case, out: Path, today: date, real: http.Transport = http._requests_transport) -> Path:
    directory = out / case.name
    directory.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(real, paced=real is http._requests_transport)
    http.use_transport(recorder, paced=recorder.paced)
    try:
        case.run(today)
    finally:
        http.use_transport(None)
    secrets = [os.environ[k].encode() for k in (connectors.FRED_KEY_ENV,) if os.environ.get(k)]
    index = {}
    for old in directory.iterdir():
        old.unlink()
    for key, (req, resp) in recorder.responses.items():
        body = trim(req, resp.content, today) if resp.status == 200 else resp.content
        if any(s in body or s in resp.content or s in key.encode() for s in secrets):
            raise SystemExit(f"{case.name}: a key appeared in a recorded response; nothing written")
        name = _name(key, resp.content_type or "")
        if name.endswith(".bin") and "csv" in (resp.content_type or ""):
            name = name[:-4] + ".csv"
        (directory / name).write_bytes(body)
        index[key] = {"file": name, "status": resp.status, "type": resp.content_type}
    index["_recorded_on"] = today.isoformat()
    (directory / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return directory


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--out", default="tests/fixtures/economy")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list or not args.cases:
        for c in CASES.values():
            print(c.name, "(needs " + ", ".join(c.needs) + ")" if c.needs else "")
        return 0
    today = datetime.now(timezone.utc).date()
    for name in args.cases:
        case = CASES[name]
        missing = [k for k in case.needs if not os.environ.get(k)]
        if missing:
            print(f"{name}: skipped, set {', '.join(missing)}", file=sys.stderr)
            continue
        print(f"{name}: {record(case, Path(args.out), today)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
