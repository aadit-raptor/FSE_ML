"""What the starting assumptions are read from, and how regions nest (PLAN.md 4.3).

**The source.** Aswath Damodaran (NYU Stern) publishes industry averages
every January for about 43,000 listed companies worldwide, one workbook per
data set and region, built from Bloomberg, Morningstar, Capital IQ and
Compustat. His usage rules ask for the data to be used in corporate finance
and valuation, as here, and thank users who say where it came from; every
figure on screen names him, the workbook and its date. The averages are
**aggregates** (the sum of the group's gross profit over the sum of its
revenue, and so on), so large companies weigh more than small ones.

**Regions.** Damodaran's own groups: the United States, Japan, China and
India each have a file of their own (so for them the figure is the
country's); "Europe" is developed Europe (the EU, the UK, Switzerland and
Scandinavia); "Australia, NZ and Canada" is one group; "Emerging markets" is
Asia other than Japan, Africa, the Middle East, Latin America, and Eastern
Europe outside the EU; "Global" is everyone. A country not listed below is
an emerging market.

**Thin data.** A group with fewer than ``MIN_FIRMS`` companies, or whose
figure can't be used (missing, or impossible such as a negative EBITDA
margin), hands over to the next group in the country's chain: country, then
region, then global. The screen says which group a figure came from and why.

**Size.** None of the free sources splits these figures by company size, so
a deal's size is asked (its EBITDA) but every size gets the same figures,
and the screen says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DAMODARAN_BASE = "https://pages.stern.nyu.edu/~adamodar/pc/datasets"
DAMODARAN_PAGE = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html"
SOURCE = {
    "id": "damodaran",
    "publisher": "Aswath Damodaran, NYU Stern",
    "title": "Damodaran Online: industry averages by region",
    "url": DAMODARAN_PAGE,
    "basis": "Listed companies, aggregated by industry (sums over sums); from Bloomberg, "
             "Morningstar, Capital IQ and Compustat",
    "usage": "Free to use for corporate finance and valuation, with acknowledgement",
}
# A group with fewer companies than this hands over to the next one
MIN_FIRMS = 20


@dataclass(frozen=True)
class Region:
    id: str
    suffix: str      # the workbook's name after the data set: margin<suffix>.xls
    level: str       # "country" (a file of its own) or "region" or "global"


REGIONS: Mapping[str, Region] = {r.id: r for r in (
    Region("us", "", "country"),
    Region("japan", "Japan", "country"),
    Region("china", "China", "country"),
    Region("india", "India", "country"),
    Region("europe", "Europe", "region"),
    Region("aus_nz_canada", "Rest", "region"),
    Region("emerging", "emerg", "region"),
    Region("global", "Global", "global"),
)}

# Developed Europe: the EU, the UK, Switzerland, Scandinavia and the small
# states and dependencies that use their markets
EUROPE = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV",
    "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "GB", "CH", "NO", "IS", "LI", "AD", "MC", "SM", "VA", "GI", "IM", "JE", "GG", "FO",
})
OWN_FILE = {"US": "us", "JP": "japan", "CN": "china", "IN": "india"}
AUS_NZ_CANADA = frozenset({"AU", "NZ", "CA"})


# Retired codes the browser still names (CLDR aliases): an account saved with
# one before the country list dropped them is read as the current country
ALIASES = {
    "AN": "CW", "BU": "MM", "CS": "RS", "DD": "DE", "DY": "BJ", "FX": "FR", "HV": "BF", "NH": "VU",
    "RH": "ZW", "SU": "RU", "TP": "TL", "UK": "GB", "VD": "VN", "YD": "YE", "YU": "RS", "ZR": "CD",
}


def canonical(country: str) -> str:
    return ALIASES.get(country, country)


def chain(country: str) -> tuple[str, ...]:
    """The groups a country's figures are looked for in, closest first."""
    own = OWN_FILE.get(country)
    if own in ("us", "japan"):
        return (own, "global")
    if own:                                   # China and India are emerging markets too
        return (own, "emerging", "global")
    if country in EUROPE:
        return ("europe", "global")
    if country in AUS_NZ_CANADA:
        return ("aus_nz_canada", "global")
    return ("emerging", "global")


def region_of(country: str) -> str:
    """The region a country's economic and tax figures are pooled in when it
    has none of its own: its first group that is not a single country."""
    return next(g for g in chain(country) if REGIONS[g].level != "country")


@dataclass(frozen=True)
class Dataset:
    id: str
    stem: str                       # margin, vebitda ...
    title: str
    columns: Mapping[str, str]      # header (lowercase, no spaces) -> metric


# The columns read, by the header Damodaran prints over them. EV/EBITDA is
# printed twice, for companies with positive EBITDA and for all; the first
# (positive EBITDA) is the one that prices a profitable company.
DATASETS: Mapping[str, Dataset] = {d.id: d for d in (
    Dataset("margins", "margin", "Operating and net margins", {
        "grossmargin": "gross_margin",
        "pre-taxunadjustedoperatingmargin": "operating_margin",
        "ebitda/sales": "ebitda_margin",
    }),
    Dataset("multiples", "vebitda", "Enterprise value multiples", {
        "ev/ebitda": "ev_ebitda",
    }),
    # Capital expenditure over D&A, both from the cash flow statement. Not
    # "Net Cap Ex/Sales": Damodaran's net capex adds acquisitions and R&D
    Dataset("capex", "capex", "Capital expenditures", {
        "capex/deprecn": "capex_to_da",
    }),
    Dataset("working_capital", "wcdata", "Working capital ratios", {
        "accrec/sales": "receivables_sales",
        "inventory/sales": "inventory_sales",
        "accpay/sales": "payables_sales",
        "non-cashwc/sales": "noncash_wc_sales",
    }),
    Dataset("debt", "dbtfund", "Debt ratios and their drivers", {
        "debttoebitda": "debt_ebitda",
        "interestcoverageratio": "interest_coverage",
    }),
)}

COUNTRY_TAX = {
    "id": "country_tax", "stem": "countrytaxrates",
    "title": "Marginal corporate tax rate by country (Tax Foundation's survey)",
}

# Every industry the averages give, except the financial ones: a bank's or an
# insurer's EBITDA and leverage don't mean what a buyout model reads them as
FINANCIAL_INDUSTRIES = frozenset({
    "Bank (Money Center)", "Banks (Regional)", "Brokerage & Investment Banking",
    "Financial Svcs. (Non-bank & Insurance)", "Insurance (General)", "Insurance (Life)",
    "Insurance (Prop/Cas.)", "Investments & Asset Management", "R.E.I.T.", "Reinsurance",
    "Retail (REITs)",
})
# The all-industries row a deal without a sector takes
ALL_INDUSTRIES = "Total Market (without financials)"
ALL_INDUSTRIES_ID = "all"
TOTAL_ROWS = frozenset({"Total Market", "Grand Total"})


def workbook_url(stem: str, region: Region | None = None) -> str:
    return f"{DAMODARAN_BASE}/{stem}{region.suffix if region else ''}.xls"


# Damodaran, "Ratings, Interest Coverage Ratios and Default Spread", January
# 2026, large non-financial firms (the same table as core/risk_sources.py
# COVERAGE_BANDS): the default spread over the risk-free rate for each rating.
# https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html
SPREAD_SOURCE_ID = "damodaran_ratings_2026"
DEFAULT_SPREAD_PCT: Mapping[str, float] = {
    "D": 19.00, "C": 16.00, "CC": 12.61, "CCC": 8.85, "B-": 5.09, "B": 3.21, "B+": 2.75,
    "BB": 1.84, "BB+": 1.38, "BBB": 1.11, "A-": 0.89, "A": 0.78, "A+": 0.70, "AA": 0.55,
    "AAA": 0.40,
}
