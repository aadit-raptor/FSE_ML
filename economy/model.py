"""What a source's figures become (PLAN.md 4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from economy.catalogue import series_key

# Newest observations kept per series: about a month of a daily rate, two
# years of a monthly one, and a WEO's history plus its projections
KEEP_OBSERVATIONS = {"D": 24, "M": 24, "A": 12}


@dataclass(frozen=True)
class Series:
    """One indicator for one area from one source, newest observations last.

    ``period`` is ``YYYY-MM-DD`` (daily), ``YYYY-MM`` (monthly) or ``YYYY``
    (annual); values are percentages (``3.75`` = 3.75%).
    """

    indicator: str
    area: str
    source: str
    source_series: str          # the series' id at its source
    frequency: str              # "D", "M" or "A"
    url: str                    # a page a person can open to check it
    observations: tuple = ()    # ((period, value), ...), oldest first

    def __post_init__(self):
        kept = sorted({p: float(v) for p, v in self.observations}.items())
        object.__setattr__(self, "observations", tuple(kept[-KEEP_OBSERVATIONS[self.frequency]:]))

    @property
    def key(self) -> str:
        return series_key(self.indicator, self.area, self.source)

    @property
    def latest(self):
        """The newest (period, value), or None."""
        return self.observations[-1] if self.observations else None


@dataclass(frozen=True)
class FxDay:
    """The ECB's euro reference rates for one day: units of each currency
    per euro (``{"USD": 1.1269, ...}``; the euro itself is 1)."""

    day: date
    rates: dict = field(default_factory=dict)
