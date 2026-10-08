"""The industry multiples the multiple predictor's card is evaluated on (PLAN.md 5.4).

    python -m tests.ml_multiple_history        # rewrites ml/evaluation/data/multiple_history.json

The card (ml/evaluation/multiples.py) tests the predictor walk-forward on
every industry's EV/EBITDA, year by year, in every Damodaran group: exactly
the history the app stores (benchmarks/history.py, ``history.<group>``) and
the current edition (``multiples.<group>``). CI evaluates cards without
network or database, so it reads this file, kept to the two figures the
predictor reads (companies and EV/EBITDA).

Unlike the test fixtures (tests/fixtures/benchmarks, ten industries), this
reads **every industry** from Damodaran's archive, so writing it calls the
archive (about 120 workbooks, paced at a second each). The archive never
changes and the current edition changes each January, so the file is
rewritten once a year. tests/test_multiples.py fails when it disagrees with
the recorded fixtures for the industries those keep.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ml" / "evaluation" / "data" / "multiple_history.json"
COLUMNS = ("firms", "ev_ebitda")


def _figures(row: Mapping) -> dict:
    return {k: row[k] for k in COLUMNS if k in row}


def build(today: date, groups: Optional[list[str]] = None) -> dict:
    """Every group's history and current multiples, read through the same
    calls the scheduled refresh makes (whatever transport is installed)."""
    from benchmarks import damodaran, history
    from benchmarks.catalogue import DATASETS, REGIONS

    tables: dict[str, dict] = {}
    read: dict[str, str] = {}
    for group in groups or list(REGIONS):
        rows: dict[str, dict] = {}
        for year in history.archive_years(today):
            status, found = history.read_archive("multiples", group, year)
            read[history.read_key("multiples", group, year)] = status
            for industry, row in found.items():
                entry = rows.setdefault(industry, {"name": row["name"], "years": {}})
                entry["name"] = row["name"]
                entry["years"][str(year)] = _figures(row)
        tables[history.table_name(group)] = {"published": None, "url": history.ARCHIVE_PAGE,
                                             "rows": dict(sorted(rows.items()))}
        current = damodaran.fetch_table(DATASETS["multiples"], REGIONS[group])
        tables[current.name] = {"published": current.published.isoformat() if current.published else None,
                                "url": current.url,
                                "rows": {k: {"name": r["name"], **_figures(r)} for k, r in sorted(current.rows.items())}}
    return {"about": "Written by python -m tests.ml_multiple_history from Damodaran's archive and current "
                     "EV/EBITDA workbooks (every industry, every group); do not edit by hand.",
            "written_on": today.isoformat(), "read": dict(sorted(read.items())), "tables": tables}


def text(data: Mapping) -> str:
    return json.dumps(data, indent=0, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    from companies import http
    today = datetime.now(timezone.utc).date()
    http.use_transport(http._requests_transport, paced=True)
    try:
        data = build(today)
    finally:
        http.use_transport(None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text(data), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
