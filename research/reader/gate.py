"""Whether the reader may read for people (PLAN.md R1; phase R rule 2: "the
reader ships only at a measured accuracy").

The answer is the model card's (``ml/cards/reader.json``, written by
``python -m ml.evaluation evaluate``): its headline set, filings nobody
tuned the prompt or the checks on, must meet R0's bar. **Production reads
only then**; staging and local copies read anyway, with the card's state
shown, so the reader can be tried while its held-out filings are recorded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

CARD = Path(__file__).resolve().parents[2] / "ml" / "cards" / "reader.json"


def measured(path: Optional[Path] = None) -> dict:
    """``{"set", "meets_bar", "cases", "amber_share"}`` of the card's headline set."""
    path = CARD if path is None else path
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
        evaluation = card["evaluation"]
        name = evaluation["headline_set"]
        group = evaluation["sets"][name]
    except (OSError, ValueError, KeyError):
        return {"set": None, "meets_bar": None, "cases": 0, "amber_share": None}
    overall = group.get("overall") or {}
    return {"set": name, "meets_bar": group.get("meets_bar"), "cases": overall.get("cases", 0),
            "amber_share": (overall.get("model") or {}).get("amber_share")}


def open_to_people(environment: str, path: Optional[Path] = None) -> bool:
    return environment != "production" or measured(path)["meets_bar"] is True
