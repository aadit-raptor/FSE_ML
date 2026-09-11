"""Smoke test for the 3-statement forecasting model.

Run directly (`python test_forecasting.py`) or under pytest.
"""

import random

from pages.forecasting import (
    ForecastAssumptions,
    HistoricalYear,
    _opening_bs_gap,
    run_3_statement_model,
)


def build_ltm(**overrides):
    # The page's default historical inputs (latest year).
    values = dict(
        year="LTM", revenue=265.0, cogs=-163.0, rd=-14.0, sga=-17.0,
        other_income=-0.4, interest_exp=-3.2, interest_inc=5.7, da=10.9,
        sbc=5.3, tax=-13.4, capex=13.3, dividends=13.7, repurchases=73.1,
        cash=237.0, ar=23.2, inventory=4.0, other_current=37.9, ppe_net=41.3,
        other_nca=22.3, ap=55.9, other_cl=32.7, deferred_rev=10.3, ltd=102.5,
        common_stock=40.2, retained_earnings=127.6, oci=-3.5,
    )
    values.update(overrides)
    return HistoricalYear(**values)


def random_assumptions(rng):
    return ForecastAssumptions(
        revenue_growth=rng.uniform(-0.25, 0.40), gross_margin=rng.uniform(0.1, 0.8),
        rd_pct=rng.uniform(0, 0.15), sga_pct=rng.uniform(0, 0.3),
        tax_rate=rng.uniform(0, 0.35), da_pct=rng.uniform(0, 0.1),
        sbc_pct=rng.uniform(0, 0.05), capex_pct=rng.uniform(0, 0.15),
        ar_days=rng.uniform(0, 120), inv_days=rng.uniform(0, 120),
        ap_days=rng.uniform(0, 120), other_cl_pct=rng.uniform(0, 0.2),
        deferred_rev_pct=rng.uniform(0, 0.1), other_nca_pct=rng.uniform(0, 0.2),
        other_income=rng.uniform(-10, 10), dividends=rng.uniform(0, 60),
        repurchases=rng.uniform(0, 150), ltd_change=rng.uniform(-60, 60),
        interest_rate_cash=rng.uniform(0, 0.06),
        interest_rate_debt=rng.uniform(0.02, 0.12), min_cash=rng.uniform(0, 300),
    )


def test_default_opening_balance_sheet_balances():
    assert abs(_opening_bs_gap(build_ltm())) < 0.01


def test_forecast_adds_no_balance_sheet_gap():
    # Every balance sheet line's change must flow through the cash flow
    # statement, so for ANY assumptions each forecast year's gap equals the
    # opening gap. Randomising the opening balance sheet too checks that the
    # model carries an unbalanced input forward rather than absorbing it.
    rng = random.Random(7)
    for _ in range(200):
        ltm = build_ltm(retained_earnings=rng.uniform(0, 300),
                        other_current=rng.uniform(0, 80))
        gap0 = _opening_bs_gap(ltm)
        years = [random_assumptions(rng) for _ in range(rng.randint(1, 10))]
        for y in run_3_statement_model(ltm, years):
            assert abs(y.balance_check - gap0) < 1e-6, (
                f"{y.year}: forecast gap {y.balance_check - gap0:.4f}")


if __name__ == "__main__":
    test_default_opening_balance_sheet_balances()
    test_forecast_adds_no_balance_sheet_gap()
    print("test_forecasting: PASS")
