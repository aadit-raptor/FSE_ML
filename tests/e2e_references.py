"""Write the browser tests' recorded reference-library answers (PLAN.md 4.5b).

The browser tests' account is no administrator and their database holds no
approved transaction, so the Reference deals, Review and Settings -> Fees
screens replay these: the repository's ten transactions as the API answers
them once approved, the sourced fees they give, and an administrator's review
queue with every one still open. Made by the API's own functions and
response models, so ``tests/test_references.py`` fails when they are stale.

    python -m tests.e2e_references
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import ReferenceDealsResponse, ReviewQueue, SourcedFeesResponse
from db.references import APPROVALS_NEEDED, Proposal
from library import fees, references, review

OUT = Path(__file__).resolve().parent.parent / "web" / "e2e" / "fixtures" / "references.json"
WHEN = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def _proposal(i: int, deal: dict) -> Proposal:
    return Proposal(id=uuid.UUID(int=i + 1), key=deal["key"], content=deal, origin="repository",
                    status="proposed", created_at=WHEN, decided_at=None, approvals=0, rejections=0,
                    reasons=(), mine=False, my_verdict=None)


def answers() -> dict:
    deals = references.repository_deals()
    approved = [{**d, "approved_at": WHEN} for d in deals]
    queue = {"enabled": True, "proposals": [review._view(_proposal(i, d), []) for i, d in enumerate(deals)],
             "decided": [], "library_size": 0}
    return {
        "references": ReferenceDealsResponse.model_validate(
            {"enabled": True, "deals": [references.summary(d) for d in approved], "awaiting_review": 0}
        ).model_dump(mode="json"),
        "fees": SourcedFeesResponse.model_validate(
            {"enabled": True, "min_deals": fees.MIN_DEALS, "library_size": len(deals),
             "settings": fees.sourced(deals)}).model_dump(mode="json"),
        "review": ReviewQueue.model_validate(queue).model_dump(mode="json"),
        "approvals_needed": APPROVALS_NEEDED,
    }


def render() -> str:
    return json.dumps(answers(), indent=1, ensure_ascii=False, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
