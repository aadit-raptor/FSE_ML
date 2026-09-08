"""Smoke test for the deterministic LBO engine.

Run directly (`python test_lbo.py`) or under pytest.
"""

from lbo_engine.model import LBOParams, run_lbo


def build_result():
    # run_lbo() takes a single LBOParams object, not loose keyword arguments.
    # The old `ebitda_margin=0.25` input no longer exists: the operating model
    # now derives it from gross_margin - opex_pct + da_pct (0.40 - 0.18 + 0.04).
    params = LBOParams(
        entry_ebitda=100,
        entry_multiple=10,
        exit_multiple=11,
        holding_period=5,
        debt_pct=0.6,
        interest_rate=0.06,
        revenue_growth=0.05,
        gross_margin=0.40,
        opex_pct=0.18,
        capex_pct=0.04,
    )

    result = run_lbo(params)

    return result


def test_run_lbo():
    result = build_result()
    assert result.returns is not None, "returns module produced no result"
    assert -1.0 <= result.irr <= 5.0, f"IRR out of plausible range: {result.irr}"
    assert result.moic > 0, f"MOIC must be positive, got {result.moic}"


if __name__ == "__main__":
    test_run_lbo()
    r = build_result()
    print(f"Entry equity : ${r.returns.entry_equity:,.0f}M")
    print(f"Exit equity  : ${r.returns.net_exit_equity:,.0f}M")
    print(f"IRR          : {r.irr * 100:.1f}%")
    print(f"MOIC         : {r.moic:.2f}x")
    # Diagnostic only: the flag is set solely when the loop breaks early under
    # $0.5M. The default 2-pass mode is expected to run both passes.
    print(f"Interest converged: {r.interest_converged}")
    print("\ntest_lbo: PASS")
