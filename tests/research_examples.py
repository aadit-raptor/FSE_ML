"""Writes `research/models/examples.json`: the companies solved by hand for
each offered model (PLAN.md R4 part a, R7), as the screen and the add-in open
them, from the fixtures in `tests/fixtures/research/` (public filings only).

    python -m tests.research_examples

`tests/test_research_models_api.py` fails when the file is stale.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from research.models import CATALOGUE
from tests import research_support as support

OUT = Path(__file__).resolve().parent.parent / "research" / "models" / "examples.json"
EXAMPLES = (("yatharth-hospital", "hospitals"), ("lemon-tree-hotels", "hotels"),
            ("jk-lakshmi-cement", "model2_jk_lakshmi"), ("star-cement", "model2_star_cement"),
            ("avenue-supermarts", "model3_avenue_supermarts"), ("v-mart-retail", "model3_v_mart"))


def examples() -> list:
    out = []
    for slug, name in EXAMPLES:
        fixture = support.load(name)
        company = asdict(support.company(fixture))
        company["off"] = list(company["off"])
        structure = CATALOGUE.spec(fixture.get("model", "occupancy")).default
        out.append({"id": slug, "structure": structure.to_dict(), "company": company})
    return out


def text() -> str:
    return json.dumps(examples(), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.write_text(text(), encoding="utf-8")
    print("wrote", OUT)
