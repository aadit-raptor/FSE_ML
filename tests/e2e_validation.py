"""Write the browser tests' recorded validation report (PLAN.md 4.6).

The browser tests' database holds no approved reference transaction and no
opted-in deal with an exit, so Backtest -> Validation replays this: the
repository's ten transactions as the default-risk check scores them once
approved, plus users' deals for the IRR range and loss checks -- six in
Europe, five in the U.S. and two in emerging markets, so Europe shows its
figures, emerging markets are hidden (two) and the U.S. with them (else the
overall figures less Europe's would give emerging markets' alone). Made by the report's own code and response model, so
``tests/test_validation.py`` fails when it is stale.

    python -m tests.e2e_validation
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from api.schemas import ValidationReportResponse
from library import references
from validation import cases, report
from validation.cases import Case

OUT = Path(__file__).resolve().parent.parent / "web" / "e2e" / "fixtures" / "validation.json"
TODAY = date(2026, 10, 8)

# (region, percentile, actual - planned IRR in points, predicted P(loss), lost)
CONTRIBUTED = (
    ("europe", 12.0, -6.0, 0.08, False), ("europe", 35.0, -2.0, 0.10, False), ("europe", 48.0, -0.5, 0.05, False),
    ("europe", 61.0, 1.5, 0.12, False), ("europe", 83.0, 4.0, 0.06, False), ("europe", 3.0, -14.0, 0.20, True),
    ("us", 55.0, 0.5, 0.09, False), ("us", 70.0, 2.0, 0.07, False), ("us", 44.0, -1.0, 0.11, False),
    ("us", 91.0, 7.0, 0.04, False), ("us", 28.0, -3.0, 0.13, False),
    ("emerging", 20.0, -4.0, 0.15, False), ("emerging", 66.0, 2.5, 0.18, False),
)


def _contributed() -> list[Case]:
    out = []
    for region, percentile, error, p_loss, lost in CONTRIBUTED:
        groups = {"region": region, "sector": "industrials", "size": "100m_1bn", "era": "2020_on"}
        common = {"origin": "contributed", "groups": groups, "out_of_time": True}
        out.append(Case(check="irr_range", percentile=percentile, irr_error_pp=error, **common))
        out.append(Case(check="loss", predicted=p_loss, happened=lost, **common))
    return out


def answer() -> dict:
    library = [c for d in references.repository_deals() if (c := cases.library_case(d, today=TODAY))]
    built = report.build(library + _contributed(), generated_at="2026-10-08T02:30:00Z", engine_version="1.0.0",
                         library_included=True, fit_until=cases.fit_until())
    return ValidationReportResponse.model_validate({"report": built}).model_dump(mode="json")


def write() -> None:
    OUT.write_text(json.dumps(answer(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write()
    print(f"wrote {OUT}")
