"""What every connector hands back, whatever the source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# Identifier kinds a company may carry. ``company_number`` is a national
# register's number (UK Companies House); ``jcn`` Japan's corporate number;
# ``edinet_code`` and ``sec_code`` EDINET's filer code and the Japanese
# securities code; ``cik`` the SEC's.
IDENTIFIERS = ("lei", "isin", "ticker", "cik", "company_number", "edinet_code", "sec_code", "jcn")

# Accounting standards a company's figures follow. A deal takes only the
# first two (core/accounting.py); the others are labels on company data.
IFRS = "ifrs"
US_GAAP = "us_gaap"
UK_GAAP = "uk_gaap"          # FRS 102 (and FRS 101, which is IFRS-based but filed with UK concepts)
JGAAP = "jgaap"
STANDARDS = (IFRS, US_GAAP, UK_GAAP, JGAAP)

# Money in a company's figures is in millions of its filing's own currency
UNIT = "millions"


@dataclass(frozen=True)
class CompanyRef:
    """A company as a source knows it: enough to load it and to tell it apart."""
    source: str
    source_id: str
    name: str
    country: Optional[str] = None            # ISO 3166 alpha-2, when the source says
    identifiers: dict = field(default_factory=dict)
    # The name in the register's own language and script, when it differs
    local_name: Optional[str] = None


@dataclass(frozen=True)
class FilingLink:
    """The filing a year's figures came from."""
    url: str
    filed_on: Optional[date]
    form: str = ""            # "10-K", "20-F", "ESEF annual report", "AA", "有価証券報告書"
    id: str = ""              # the source's id for the filing


@dataclass(frozen=True)
class YearFigures:
    fiscal_year: int                         # named by the year it ends in
    period_end: date
    figures: dict                            # SUMMARY_FIELDS -> float (millions) or None
    filing: FilingLink


@dataclass(frozen=True)
class Warning:
    """Something the reader should know: a code the screen translates, and
    the field it is about, if any. Never a sentence, never a figure."""
    code: str
    field: str = ""


@dataclass(frozen=True)
class CompanyData:
    ref: CompanyRef
    currency: str
    accounting_standard: str
    fiscal_year_end_month: Optional[int]
    years: list                              # YearFigures, oldest first
    warnings: list = field(default_factory=list)
    unit: str = UNIT
