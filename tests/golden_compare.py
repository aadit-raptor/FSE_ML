"""Near-exact comparison of nested model outputs.

Locally core/ matches the Streamlit snapshot bit for bit. CI runs on a
different OS whose numpy builds (random draws, linear algebra) can differ in
the last few bits, so tests allow a relative difference of 1e-9 -- far below
any change a logic edit would produce.
"""
import math

REL, ABS = 1e-9, 1e-12


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
