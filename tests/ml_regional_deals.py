"""The regional deals the surrogate's card is measured on (PLAN.md 5.1).

    python -m tests.ml_regional_deals       # rewrites ml/evaluation/data/regional_deals.json

For every economy the app covers and each industry in
``ml.evaluation.surrogate.INDUSTRIES``: the Monte Carlo inputs "Use sourced
figures" gives (PLAN.md 4.3, 4.4), from the recorded industry workbooks,
their history and the economic series (tests/fixtures/benchmarks,
tests/fixtures/economy), judged on the day the series were recorded, with
the deal's default debt share. tests/test_model_cards.py fails when the
file is stale.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks import risk, starting
from core.deal import DealInputs
from economy.catalogue import AREAS, COUNTRIES
from library.base_rates import sp_region
from ml.evaluation.surrogate import INDUSTRIES, REGIONAL_DEALS, features_from
from tests.e2e_benchmarks import recorded_sources


def build() -> dict:
    tables, series, today = recorded_sources()
    debt_pct = DealInputs().debt_pct
    deals, skipped = [], []
    for country in COUNTRIES:
        currency = AREAS[country].currency
        for industry in INDUSTRIES:
            start = starting.starting_assumptions(country, industry, currency, tables, series, today)
            ranges = risk.risk_assumptions(country, industry, currency, tables, series, today)
            features = features_from({f["field"]: f["value"] for f in ranges["figures"]}, start["inputs"],
                                     debt_pct)
            deal_id = f"{country}|{industry}"
            if features is None:
                skipped.append(deal_id)
                continue
            deals.append({"id": deal_id, "country": country, "industry": industry, "currency": currency,
                          "region": sp_region(country), "features": features})
    return {"recorded_on": today.isoformat(), "debt_pct": debt_pct, "skipped": skipped, "deals": deals}


def text() -> str:
    return json.dumps(build(), indent=1, sort_keys=True) + "\n"


def main() -> None:
    Path(REGIONAL_DEALS).parent.mkdir(parents=True, exist_ok=True)
    REGIONAL_DEALS.write_text(text(), encoding="utf-8", newline="\n")
    print(f"wrote {REGIONAL_DEALS}")


if __name__ == "__main__":
    main()
