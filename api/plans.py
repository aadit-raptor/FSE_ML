"""Limits per plan, as settings (PLAN.md R9; phase R rule 8: "free now, paid by a switch").

Two kinds of number, each chosen by one setting:

- **A person's plan** (``FSE_PLAN``): how many results reads, pages read and note sentences a person may ask for
  a day. Until billing (PLAN.md 9) gives each account its own, every account is on the plan the service names;
  ``plan_for`` is the one place that will change then. An unknown name is the free plan, so a typo never lifts
  a limit.
- **The AI's tier** (``AI_FREE_TIER``, research/reader/providers.py): how many reads and writings the whole
  service makes a day. On the free tier that is what fits in Google's free allowance (20 requests a day per
  model, two models a read); billed, it is what a day may spend inside the spending cap the user set on the
  Google project, which stays the real backstop.

The free plan and the free tier are exactly the numbers in force before R9 (``Rule.limit`` in api/limits.py,
tested). DEPLOY.md "Paid switches" says what each lifts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    name: str
    reads_per_day: int          # results reads (PLAN.md R1)
    pages_per_day: int          # pages of them
    writes_per_day: int         # a results note's linking sentences (PLAN.md R8)
    document_pages_per_day: int  # pages of documents read for a deal (PLAN.md 6.2): an annual report is long


@dataclass(frozen=True)
class Tier:
    name: str
    reads_per_day: int          # for everyone together
    writes_per_day: int


PLANS = {
    "free": Plan("free", reads_per_day=10, pages_per_day=250, writes_per_day=5,
                 document_pages_per_day=1_000),
    # an analyst covering a sector through results season: about a dozen filings on a heavy day, room to reread
    "research": Plan("research", reads_per_day=40, pages_per_day=1_000, writes_per_day=30,
                     document_pages_per_day=4_000),
}
DEFAULT_PLAN = "free"

TIERS = {
    "free": Tier("free", reads_per_day=30, writes_per_day=10),
    # a filing costs about 0.09 US dollars (research/ai_costs.py): 300 reads is under 30 dollars at most
    "billed": Tier("billed", reads_per_day=300, writes_per_day=300),
}


def plan_for(subject: str) -> Plan:  # noqa: ARG001 -- one plan for every account until billing (PLAN.md 9)
    name = os.environ.get("FSE_PLAN", DEFAULT_PLAN).strip().lower()
    return PLANS.get(name, PLANS[DEFAULT_PLAN])


def tier() -> Tier:
    from research.reader.providers import free_tier

    return TIERS["free" if free_tier() else "billed"]


def limit(per: str, field: str, identity: str) -> int:
    """A plan rule's number: a person's from their plan, the service's from the AI's tier."""
    source = tier() if per == "service" else plan_for(identity)
    return int(getattr(source, field))
