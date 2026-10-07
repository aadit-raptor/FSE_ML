"""Fail when a base-rate edition is due to be checked again (PLAN.md 4.5).

    python3 -m ops.check_base_rates [--today 2027-10-07]

The reference library's default and recovery rates are transcribed from
yearly studies (``library/base_rates.py``). Each source records the day its
edition was last confirmed as the newest one free to read; a year later this
check fails, so someone looks for a newer edition and either transcribes it
or confirms there is none (and moves ``checked_on``). Run daily by live.yml.
Standard library only.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Optional

from library import base_rates


def due(today: date) -> list[str]:
    """One line per source that is due a check; empty when all are current."""
    return [f"{sid}: {s['publisher']}, {s['title']} (published {s['published']}), last confirmed "
            f"{s['checked_on']}, due {base_rates.recheck_due(sid).isoformat()}"
            for sid, s in base_rates.SOURCES.items() if base_rates.stale(sid, today)]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--today", type=date.fromisoformat, default=None, help="judge as of this day (ISO)")
    args = parser.parse_args(argv)
    found = due(args.today or date.today())
    if not found:
        print(f"base rates: {len(base_rates.SOURCES)} sources, all confirmed within the year")
        return 0
    print("base rates due a check for a newer edition (library/base_rates.py, then move checked_on):")
    for line in found:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
