"""Writes `research/models/examples.json`: model 1's two companies solved by
hand (PLAN.md R4 part a), as the screen and the add-in open them, from the
fixtures in `tests/fixtures/research/` (public filings only).

    python -m tests.research_examples

`tests/test_research_models_api.py` fails when the file is stale.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from research.models import occupancy
from tests import research_support as support

OUT = Path(__file__).resolve().parent.parent / "research" / "models" / "examples.json"
EXAMPLES = (("yatharth-hospital", "hospitals"), ("lemon-tree-hotels", "hotels"))


def examples() -> list:
    out = []
    for slug, fixture in EXAMPLES:
        company = asdict(support.company(support.load(fixture)))
        company["off"] = list(company["off"])
        out.append({"id": slug, "structure": occupancy.DEFAULT.to_dict(), "company": company})
    return out


def text() -> str:
    return json.dumps(examples(), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.write_text(text(), encoding="utf-8")
    print("wrote", OUT)
