"""The industry averages the deal risk score's card is evaluated on (PLAN.md 5.2).

    python -m tests.ml_peer_tables          # rewrites ml/evaluation/data/peer_tables.json

The card (ml/evaluation/deal_risk.py) compares each reference transaction
with its industry's companies in its region, as the app does with the
stored averages. CI evaluates cards without network or database, so it
reads this file: the three data sets the score reads (margins, multiples,
debt) for every group, from the recorded January 2026 workbooks
(tests/fixtures/benchmarks/damodaran), keeping only the columns the score
reads. tests/test_deal_risk.py fails when the file is stale.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ml" / "evaluation" / "data" / "peer_tables.json"


def build() -> dict:
    from benchmarks import damodaran
    from benchmarks.catalogue import DATASETS, REGIONS
    from companies import http
    from companies.record import replay
    from ml.anomaly_detector import DATASETS_READ, METRICS
    from tests.e2e_benchmarks import FIXTURES

    columns = {m.dataset: m.column for m in METRICS}
    http.use_transport(replay(FIXTURES))
    try:
        out = {}
        for dataset in DATASETS_READ:
            for region in REGIONS.values():
                t = damodaran.fetch_table(DATASETS[dataset], region)
                out[t.name] = {"published": t.published.isoformat() if t.published else None, "url": t.url,
                               "rows": {k: {"name": r["name"], "firms": r["firms"],
                                            **({columns[dataset]: r[columns[dataset]]}
                                               if columns[dataset] in r else {})}
                                        for k, r in sorted(t.rows.items())}}
    finally:
        http.use_transport(None)
    return {"about": "Written by python -m tests.ml_peer_tables from tests/fixtures/benchmarks/damodaran "
                     "(Damodaran's January 2026 industry averages); do not edit by hand.",
            "tables": out}


def text() -> str:
    return json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
