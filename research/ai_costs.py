"""What an AI request costs (PLAN.md R9): its token counts at the provider's **paid** price list.

On the free tier Google charges nothing, so the figure kept for a free request is what it *would* cost once
billing is on (``db.ai_usage`` keeps the tier beside it); on the billed tier it is what Google charges, at list
price, before any discount. Thinking is billed as output.

The list is transcribed, like the published tables elsewhere (``core/risk_sources.py``): ``SOURCE`` and
``CHECKED_ON`` say where and when. A model's price can change on a dated day (Google's 3.x flash models double
on 2027-01-01), so each model holds its prices from each day on. A model not listed has no cost (``None``), and
``tests/test_paid_switch.py`` fails if the reader or the writer may ask one.

Measured on the 38 recorded filings (R0's 30 and R1b's eight, both reads, prices of 2026-10-20): about
**0.09 US dollars a filing** (median 0.093, the dearest 0.19), about 0.0037 a page.
"""
from __future__ import annotations

import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Mapping, Optional

CURRENCY = "USD"
SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
CHECKED_ON = datetime.date(2026, 10, 20)
PER = Decimal(1_000_000)
PLACES = Decimal("0.000001")          # db.models.MoneyAmount keeps six places

_FLASH = ((datetime.date.min, Decimal("0.75"), Decimal("3.75")),
          (datetime.date(2027, 1, 1), Decimal("1.50"), Decimal("7.50")))
# model -> ((from day, US dollars a million input tokens, a million output tokens), ...), oldest first
PRICES: dict[str, tuple] = {
    "gemini-3.8-flash": _FLASH,
    "gemini-3.6-flash": _FLASH,
    # not on Google's page on 2026-10-20; the trackers that list it give 3.8's prices, and so do we
    "gemini-3.7-flash": _FLASH,
    "gemini-3-flash-preview": ((datetime.date.min, Decimal("0.50"), Decimal("3.00")),),
}


def tokens(usage: Optional[Mapping]) -> tuple[int, int]:
    """(input, output) tokens of a provider's usage counts; thinking counts as output."""
    usage = usage or {}

    def count(key: str) -> int:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    return count("promptTokenCount"), count("candidatesTokenCount") + count("thoughtsTokenCount")


def price(model: str, day: datetime.date) -> Optional[tuple[Decimal, Decimal]]:
    found = None
    for since, given, made in PRICES.get(model, ()):
        if since <= day:
            found = (given, made)
    return found


def cost(model: str, usage: Optional[Mapping], day: datetime.date) -> Optional[Decimal]:
    """US dollars, to six places; ``None`` for a model with no price."""
    rates = price(model, day)
    if rates is None:
        return None
    given, made = tokens(usage)
    return ((given * rates[0] + made * rates[1]) / PER).quantize(PLACES, rounding=ROUND_HALF_UP)
