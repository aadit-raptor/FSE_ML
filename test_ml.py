"""Tests for the optional ML layer (ml/).

Skipped automatically when the ML dependencies (requirements-ml.txt) are not
installed, so the core test run does not depend on them.
"""

import json
import os

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import ml.anomaly_detector as ad  # noqa: E402


def test_interest_coverage_is_ebitda_over_interest():
    # Leverage is debt/EBITDA, so coverage = 100 / (leverage * rate%).
    # 12.5x at 7.8%: interest is 97.5% of EBITDA -> 1.03x, a covenant breach.
    freescale = ad.check_deal(14.7, 12.5, 4.0, 9.8, 7.8)
    assert any("1.03x" in w for w in freescale.warnings), freescale.warnings
    # 4.2x at 7.26% -> 3.28x: well clear of the 1.5x covenant.
    default = ad.check_deal(10.0, 4.2, 5.0, 26.0, 7.26)
    assert not any("coverage" in w for w in default.warnings), default.warnings


def test_anomaly_flag_separates_historical_outcomes():
    # The flag must carry information: it should catch the historical
    # failures while passing nearly all of the successes.
    with open(os.path.join(ad.BASE, "anomaly_deals.json")) as f:
        deals = json.load(f)
    flagged = {True: [], False: []}
    for d in deals:
        r = ad.check_deal(d["entry_mult"], d["leverage"], d["growth"],
                          d["margin"], d["interest"])
        flagged[d["success"]].append(r.is_anomalous)
    assert all(flagged[False]), "every historical failure should be flagged"
    assert sum(flagged[True]) / len(flagged[True]) <= 0.10, (
        "no more than 10% of historical successes should be flagged")


if __name__ == "__main__":
    test_interest_coverage_is_ebitda_over_interest()
    test_anomaly_flag_separates_historical_outcomes()
    print("test_ml: PASS")
