"""Python coverage floor for CI (PLAN.md 0.3).

    python -m ops.coverage_gate --job core --coverage coverage.json \\
        --floor .coverage-floor [--base-floor base-floor.txt]

``.coverage-floor`` holds one line per CI job (``core 71.2``). The job fails
when its measured total is below its floor, or when the pull request lowers
a floor (or drops a job) compared with the base branch's file: the floor
only rises. When coverage is well above the floor it prints the new figure
to raise it to. Changed lines are checked separately by ``diff-cover``
(tests.yml, 80%). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Coverage this far above the floor is worth recording as the new floor
RAISE_MARGIN = 0.5


def parse_floors(text: str) -> dict[str, float]:
    """``{job: percent}`` from the floor file; '#' starts a comment."""
    floors: dict[str, float] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"floor file line {number}: expected '<job> <percent>', got {raw!r}")
        job, value = parts
        try:
            percent = float(value)
        except ValueError:
            raise ValueError(f"floor file line {number}: {value!r} is not a number") from None
        if not 0 <= percent <= 100:
            raise ValueError(f"floor file line {number}: {percent} is not a percentage")
        if job in floors:
            raise ValueError(f"floor file line {number}: {job} appears twice")
        floors[job] = percent
    return floors


def problems(job: str, measured: float, floors: dict[str, float],
             base_floors: Optional[dict[str, float]]) -> list[str]:
    """Why this job fails the gate; empty when it passes."""
    found = []
    floor = floors.get(job)
    if floor is None:
        found.append(f"{job}: no floor in .coverage-floor; add '{job} {math.floor(measured * 10) / 10}'")
    elif measured < floor:
        found.append(f"{job}: coverage {measured:.2f}% is below the floor {floor}%. "
                     "Add tests for the new code rather than lowering the floor.")
    for base_job, base_value in (base_floors or {}).items():
        new = floors.get(base_job)
        if new is None or new < base_value:
            found.append(f"{base_job}: the floor went from {base_value}% to "
                         f"{'nothing' if new is None else f'{new}%'}; it may only rise, never lower")
    return found


def raise_hint(job: str, measured: float, floors: dict[str, float]) -> Optional[str]:
    """The figure to raise the floor to, when coverage is well above it."""
    floor = floors.get(job)
    if floor is None or measured < floor + RAISE_MARGIN:
        return None
    return f"{job}: coverage {measured:.2f}% is well above the floor {floor}%; raise it to {math.floor(measured * 10) / 10}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True)
    parser.add_argument("--coverage", required=True, help="coverage.json from 'coverage json'")
    parser.add_argument("--floor", default=".coverage-floor")
    parser.add_argument("--base-floor", help="the base branch's floor file; empty or missing = no rise check")
    args = parser.parse_args(argv)

    measured = json.loads(Path(args.coverage).read_text(encoding="utf-8"))["totals"]["percent_covered"]
    floors = parse_floors(Path(args.floor).read_text(encoding="utf-8"))
    base_floors = None
    if args.base_floor and Path(args.base_floor).is_file():
        base_floors = parse_floors(Path(args.base_floor).read_text(encoding="utf-8")) or None

    print(f"Coverage {args.job}: {measured:.2f}% (floor {floors.get(args.job, 'none')}%)")
    hint = raise_hint(args.job, measured, floors)
    if hint:
        print(f"::notice::{hint}")
    found = problems(args.job, measured, floors, base_floors)
    for message in found:
        print(f"::error::{message}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
