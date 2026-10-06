"""Which economies, indicators and sources (PLAN.md 4.2).

Every figure is a ``Series`` keyed ``<indicator>.<area>.<source>``: the
indicator (``INDICATORS``), the area (an ISO 3166 alpha-2 country, ``XM``
for the euro area, or a benchmark's code for a reference rate) and the
source it came from (``SOURCES``). Where two sources give the same
indicator, ``economy.views`` says which one stands for the country.

Choices worth knowing:

- **Growth and inflation** come from two places on purpose: the World Bank's
  latest *actual* year and the IMF World Economic Outlook's *projection* for
  the current year. The screen says which it shows.
- **Policy rates** are the BIS's collection of central bank policy rates; a
  euro member's is the euro area's (``EURO_MEMBERS``).
- **Credit spreads**: the corporate spread series on FRED (ICE BofA,
  Moody's) are licensed with "no reproduction without permission", so they
  are not shown. What is shown is the **sovereign spread** of each euro
  member's 10-year yield over Germany's, computed from the OECD's openly
  licensed yields (``economy.views``). Corporate spreads wait for a licensed
  source (phase 12).
- **Reference rates** (core/debt.py ``REFERENCE_RATES``): SOFR and SONIA from
  FRED (New York Fed, Bank of England), €STR and EURIBOR from the ECB. TONA,
  SARON, BBSY and MIBOR have no free source here, so each stands in with the
  rate it tracks and says so (``basis``): the Bank of Japan's, the SNB's and
  the RBI's policy rates, and Australia's 3-month bank bill rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class Area:
    code: str            # ISO 3166-1 alpha-2, or XM for the euro area
    iso3: str            # IMF, World Bank and OECD use these
    currency: str        # ISO 4217


# The economies kept current: the G20's members, the larger euro members,
# and the other large markets deals are done in. The euro area is an area
# of its own (its policy rate and yield); its members take its policy rate.
AREAS: Mapping[str, Area] = {a.code: a for a in (
    Area("US", "USA", "USD"), Area("GB", "GBR", "GBP"), Area("DE", "DEU", "EUR"),
    Area("FR", "FRA", "EUR"), Area("IT", "ITA", "EUR"), Area("ES", "ESP", "EUR"),
    Area("NL", "NLD", "EUR"), Area("CH", "CHE", "CHF"), Area("SE", "SWE", "SEK"),
    Area("NO", "NOR", "NOK"), Area("PL", "POL", "PLN"), Area("JP", "JPN", "JPY"),
    Area("CN", "CHN", "CNY"), Area("IN", "IND", "INR"), Area("KR", "KOR", "KRW"),
    Area("AU", "AUS", "AUD"), Area("CA", "CAN", "CAD"), Area("BR", "BRA", "BRL"),
    Area("MX", "MEX", "MXN"), Area("ZA", "ZAF", "ZAR"), Area("SG", "SGP", "SGD"),
    Area("SA", "SAU", "SAR"), Area("TR", "TUR", "TRY"), Area("ID", "IDN", "IDR"),
    Area("XM", "EA20", "EUR"),
)}
COUNTRIES = tuple(code for code in AREAS if code != "XM")
EURO_MEMBERS = frozenset({"DE", "FR", "IT", "ES", "NL"})
EURO_AREA = "XM"
SPREAD_BENCHMARK = "DE"          # a euro member's sovereign spread is over Germany's yield

INDICATORS = (
    "gdp_growth",         # real GDP, % a year
    "inflation",          # consumer prices, % a year (average)
    "policy_rate",        # %, the central bank's policy rate
    "bond_yield_10y",     # %, long-term government bond yield
    "short_rate_3m",      # %, 3-month interbank rate
    "sovereign_spread",   # % points, over SPREAD_BENCHMARK's 10-year yield (computed)
    "reference_rate",     # %, a floating-rate benchmark (the area is its code)
)

# Where a country's long-term yield comes from when there are two: FRED's
# daily US Treasury yield before the OECD's monthly average
BOND_YIELD_SOURCES = ("fred", "oecd")


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    licence: str
    key_env: Optional[str] = None


SOURCES: Mapping[str, Source] = {s.id: s for s in (
    Source("imf", "IMF World Economic Outlook (DataMapper)",
           "IMF terms of use: free reuse with attribution"),
    Source("worldbank", "World Bank, World Development Indicators", "CC BY 4.0"),
    Source("oecd", "OECD, Monthly Monetary and Financial Statistics", "CC BY 4.0"),
    Source("bis", "BIS, Central bank policy rates", "BIS terms of use: free reuse with attribution"),
    Source("ecb", "ECB Data Portal", "ECB terms of use: free reuse with attribution"),
    Source("fred", "FRED, Federal Reserve Bank of St. Louis",
           "FRED terms of use; SOFR (New York Fed), SONIA (Bank of England), "
           "Treasury yields (US Treasury), each reused with attribution", "FRED_API_KEY"),
)}


@dataclass(frozen=True)
class Candidate:
    key: str             # a Series key
    basis: str           # "benchmark" (the rate itself), or what stands in for it


# Where each benchmark's current level comes from, best first. A candidate is
# used only while its value is current (economy.views.MAX_AGE_DAYS).
REFERENCE_SOURCES: Mapping[str, tuple] = {
    "SOFR": (Candidate("reference_rate.SOFR.fred", "benchmark"), Candidate("policy_rate.US.bis", "policy_rate")),
    "SONIA": (Candidate("reference_rate.SONIA.fred", "benchmark"), Candidate("policy_rate.GB.bis", "policy_rate")),
    "ESTR": (Candidate("reference_rate.ESTR.ecb", "benchmark"), Candidate("policy_rate.XM.bis", "policy_rate")),
    "EURIBOR": (Candidate("reference_rate.EURIBOR.ecb", "benchmark"),
                Candidate("short_rate_3m.XM.oecd", "interbank_3m")),
    "TONA": (Candidate("policy_rate.JP.bis", "policy_rate"),),
    "SARON": (Candidate("policy_rate.CH.bis", "policy_rate"),),
    "BBSY": (Candidate("short_rate_3m.AU.oecd", "interbank_3m"), Candidate("policy_rate.AU.bis", "policy_rate")),
    "MIBOR": (Candidate("policy_rate.IN.bis", "policy_rate"),),
}

# The benchmark a new floating facility starts on, by the deal's currency.
# EURIBOR, not €STR, for the euro: leveraged loans in euros price off it.
CURRENCY_BENCHMARKS: Mapping[str, str] = {
    "USD": "SOFR", "GBP": "SONIA", "EUR": "EURIBOR", "JPY": "TONA",
    "CHF": "SARON", "AUD": "BBSY", "INR": "MIBOR",
}


def series_key(indicator: str, area: str, source: str) -> str:
    return f"{indicator}.{area}.{source}"
