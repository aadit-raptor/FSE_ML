"""Near-exact comparison of nested model outputs.

Locally core/ matches the Streamlit snapshot bit for bit. CI runs on a
different OS whose numpy builds (random draws, linear algebra) can differ in
the last few bits, so tests allow a relative difference of 1e-9 -- far below
any change a logic edit would produce.
"""
import math

REL, ABS = 1e-9, 1e-12


def assert_min_cash_funded(returns, bridge, golden, mincash):
    """Deal outputs once sponsor equity funds the minimum cash (finding 1).

    The snapshot recorded entry equity without the minimum cash. Everything
    about exit is unchanged; entry equity rises by the minimum cash, MOIC and
    IRR follow from it, and the bridge closes with no residual.
    """
    r, br = golden["returns"], golden["equity_bridge"]
    exit_side = ("exit_ebitda", "exit_multiple", "holding_period", "net_debt_at_exit",
                 "exit_ev", "gross_exit_equity", "mgmt_dilution", "net_exit_equity")
    assert_close({k: returns[k] for k in exit_side}, {k: r[k] for k in exit_side}, "returns")
    entry = r["entry_equity"] + mincash
    assert math.isclose(returns["entry_equity"], entry, abs_tol=1e-6)
    assert math.isclose(returns["moic"], r["net_exit_equity"] / entry, abs_tol=1e-4)
    assert math.isclose(returns["irr"], (r["net_exit_equity"] / entry) ** (1 / r["holding_period"]) - 1, abs_tol=1e-5)
    for k in ("ebitda_growth", "multiple_expansion", "deleveraging", "entry_costs", "exit_equity"):
        assert math.isclose(bridge[k], br[k], abs_tol=0.011), k
    assert math.isclose(bridge["entry_equity"], entry, abs_tol=0.011)
    assert abs(bridge["residual"]) <= 0.011


def assert_close(got, expected, path="result"):
    if isinstance(expected, dict):
        assert isinstance(got, dict) and set(got) == set(expected), \
            f"{path}: keys differ {sorted(set(got) ^ set(expected))[:5]}"
        for k in expected:
            assert_close(got[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected), f"{path}: length differs"
        for i, (g, e) in enumerate(zip(got, expected)):
            assert_close(g, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert got == expected, f"{path}: {got!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{path}: {got!r} is not a number"
        assert math.isclose(got, expected, rel_tol=REL, abs_tol=ABS), f"{path}: {got!r} != {expected!r}"
    else:
        assert got == expected, f"{path}: {got!r} != {expected!r}"
