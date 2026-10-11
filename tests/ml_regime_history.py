"""What the economic regime's card is tested on, and its recorded figures (PLAN.md 5.7).

    python -m tests.ml_regime_history --cache DIR

Writes two things from the IMF's own files:

- ``ml/evaluation/data/regime_history.json``: for each of the seven regions
  (ml/macro_regime.py ``REGIONS``), real GDP growth and inflation by year as
  the World Economic Outlook gives them today (the DataMapper), and what
  each spring and autumn report since 1990 said about its own year and the
  year before, from the IMF's **historical forecasts database**
  (``WEOhistorical.xlsx``, about 9 MB; kept in ``DIR`` between runs). That
  file ends with the reports it was last rebuilt with, so the card tests
  the years it covers. Figures are rounded to one decimal, as the
  DataMapper publishes them.
- ``tests/fixtures/regime/imf``: the two DataMapper answers the monthly
  refit reads, trimmed to the seven regions, for the tests to replay.

Rerun when the IMF adds reports to the historical file, then ``python -m
ml.evaluation evaluate`` and ``python -m tests.e2e_regime``.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path

from companies import http
from companies.record import INDEX, Recorder, _name
from economy.connectors import IMF_API, IMF_INDICATORS
from ml import macro_regime as mr

ROOT = Path(__file__).parent.parent
DATA = ROOT / "ml" / "evaluation" / "data" / "regime_history.json"
FIXTURE = ROOT / "tests" / "fixtures" / "regime" / "imf"
HISTORICAL_URL = "https://www.imf.org/external/pubs/ft/weo/data/WEOhistorical.xlsx"
HISTORICAL_PAGE = "https://www.imf.org/en/Publications/WEO/weo-database"
HISTORICAL_FILE = "WEOhistorical.xlsx"
# The historical file's sheets, as (growth, inflation)
SHEETS = ("ngdp_rpch", "pcpi_pch")
# The file names its aggregates, and gives them no ISO code
NAMED = {"Euro area": "eurozone", "World": "global"}
PLACES = 1


def historical(cache: Path) -> bytes:
    path = cache / HISTORICAL_FILE
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(http.get("imf", HISTORICAL_URL).content)
    return path.read_bytes()


def vintages(workbook: bytes) -> dict[str, dict[str, dict[str, list]]]:
    """``{region: {"S2009": {"2008": [growth, inflation], "2009": [...]}}}``:
    each report's figures for its own year and the year before."""
    import openpyxl

    by_iso3 = {r.imf: r.id for r in mr.REGIONS.values()}
    book = openpyxl.load_workbook(io.BytesIO(workbook), read_only=True, data_only=True)
    found: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
    for i, sheet in enumerate(SHEETS):
        rows = book[sheet].iter_rows(values_only=True)
        head = next(rows)
        for row in rows:
            region = by_iso3.get(row[2]) or NAMED.get(row[0])
            if region is None or not isinstance(row[3], int):
                continue
            for column, value in zip(head[4:], row[4:]):
                report = int(column[1:5])                       # "S2009ngdp_rpch"
                if isinstance(value, (int, float)) and row[3] in (report, report - 1):
                    found.setdefault(region, {}).setdefault(column[:5], {}).setdefault(row[3], {})[i] = round(
                        float(value), PLACES)
    return {region: {report: {str(y): [v[0], v[1]] for y, v in sorted(years.items()) if len(v) == 2}
                     for report, years in sorted(reports.items())}
            for region, reports in sorted(found.items())}


def trim(content: bytes) -> bytes:
    """A DataMapper answer kept for the seven regions only."""
    data = json.loads(content)
    wanted = {r.imf for r in mr.REGIONS.values()}
    values = {code: {iso3: row for iso3, row in (rows or {}).items() if iso3 in wanted}
              for code, rows in (data.get("values") or {}).items()}
    return json.dumps({"values": values}, sort_keys=True).encode()


def record(today: date) -> dict[str, dict[int, tuple]]:
    """Read the DataMapper as the refit does, saving the answers as the
    tests' fixture; the figures read."""
    recorder = Recorder(http._requests_transport)
    http.use_transport(recorder, paced=True)
    try:
        series = mr.read(today)
    finally:
        http.use_transport(None)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    for old in FIXTURE.iterdir():
        old.unlink()
    index = {}
    for key, (_, resp) in recorder.responses.items():
        file = _name(key, resp.content_type or "")
        (FIXTURE / file).write_bytes(trim(resp.content))
        index[key] = {"file": file, "status": resp.status, "type": resp.content_type}
    index["_recorded_on"] = today.isoformat()
    (FIXTURE / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return series


def build(cache: Path, today: date) -> dict:
    series = record(today)
    return {
        "written_on": today.isoformat(),
        "sources": {
            "figures": {"publisher": "IMF World Economic Outlook (DataMapper)", "url": mr.IMF_PAGE,
                        "series": [f"{IMF_API}/{code}" for code in IMF_INDICATORS.values()]},
            "reports": {"publisher": "IMF World Economic Outlook, historical forecasts database",
                        "url": HISTORICAL_PAGE, "file": HISTORICAL_URL},
        },
        # Full years only: this year's figure is still a projection
        "actual": {region: {str(y): list(v) for y, v in own.items() if y < today.year}
                   for region, own in sorted(series.items())},
        "reports": vintages(historical(cache)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args(argv)
    data = build(args.cache, datetime.now(timezone.utc).date())
    DATA.write_text(json.dumps(data, indent=None, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"wrote {DATA} ({DATA.stat().st_size:,} bytes) and {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
