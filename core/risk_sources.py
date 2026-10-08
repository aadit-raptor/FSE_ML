"""Published figures the deal's risk warnings compare it with (PLAN.md 2.8).

Every number a warning shows is either computed from the deal's own model run
or read from one of the tables below, each transcribed from a public source
named beside it. Nothing here is an estimate of ours: to change a figure,
transcribe a newer edition of its source and update ``as_of``.

``tests/test_risk_warnings.py`` checks the tables hang together (default rates
rise with the horizon and fall with the rating; coverage bands tile the line)
and that every warning names a source listed here.
"""
from __future__ import annotations

# Where each figure comes from. ``sample`` says how many observations stand
# behind it, when the source says; supervisory guidance has none.
SOURCES: dict[str, dict] = {
    "sp_default_study_2024": {
        "publisher": "S&P Global Ratings",
        "title": "Default, Transition, and Recovery: 2024 Annual Global Corporate "
                 "Default And Rating Transition Study",
        "published": "2025-03-27",
        "detail": "Table 26, global corporate average cumulative default rates by rating level, 1981-2024",
        # The copy published by S&P Global Ratings Maalot, S&P's Israeli
        # affiliate, which is free to read; spglobal.com keeps it behind a sign-in
        "url": "https://maalot.co.il/Publications/FTS20250331162126.pdf",
        "sample": {"count": 23831, "what": "issuers", "first_year": 1981, "last_year": 2024},
    },
    # The same study's regional tables, which the distress predictor reads
    # (ml/distress_model.py, PLAN.md 5.3); transcribed in library/base_rates.py
    "sp_default_study_2024_regions": {
        "publisher": "S&P Global Ratings",
        "title": "Default, Transition, and Recovery: 2024 Annual Global Corporate "
                 "Default And Rating Transition Study",
        "published": "2025-03-27",
        "detail": "Tables 24 and 25, average cumulative default rates by rating category, global and "
                  "for the U.S., Europe and emerging markets, 1981-2024",
        "url": "https://maalot.co.il/Publications/FTS20250331162126.pdf",
        "sample": {"count": 23831, "what": "issuers", "first_year": 1981, "last_year": 2024},
    },
    "sp_corporate_methodology_2024": {
        "publisher": "S&P Global Ratings",
        "title": "Criteria | Corporates | General: Corporate Methodology",
        "published": "2024-01-07",
        "detail": "Table 17, cash flow/leverage analysis ratios (standard volatility), debt / EBITDA "
                  "to financial risk profile; Table 3, business and financial risk profiles to anchor",
        # The copy published by S&P Global Ratings Maalot, free to read
        "url": "https://www.maalot.co.il/Publications/MT20240214173645.PDF",
        "sample": None,
    },
    "damodaran_ratings_2026": {
        "publisher": "Aswath Damodaran, NYU Stern",
        "title": "Ratings, Interest Coverage Ratios and Default Spread",
        "published": "2026-01",
        "detail": "Interest coverage (EBIT / interest expense) to rating, large non-financial "
                  "firms; built from all rated companies in the United States",
        "url": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html",
        "sample": None,
    },
    "ecb_leveraged_2017": {
        "publisher": "European Central Bank, Banking Supervision",
        "title": "Guidance on leveraged transactions",
        "published": "2017-05",
        "detail": "Section 4: total debt to EBITDA above 6.0 times at deal inception should "
                  "remain exceptional; above it raises concerns for most industries",
        "url": "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/"
               "ssm.leveraged_transactions_guidance_201705.en.pdf",
        "sample": None,
    },
    "us_leveraged_2013": {
        "publisher": "Federal Reserve, FDIC and OCC",
        "title": "Interagency Guidance on Leveraged Lending",
        "published": "2013-03-21",
        "detail": "Leverage above 6X total debt / EBITDA raises concerns for most industries",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1303a1.pdf",
        "sample": None,
    },
    "deal_model": {
        "publisher": "Variater",
        "title": "This deal's own model run",
        "published": None,
        "detail": "Computed from the deal's inputs and Settings",
        "url": None,
        "sample": None,
    },
}

# ECB 2017 section 4 and the US 2013 interagency guidance both draw the line at
# total debt above 6.0 times EBITDA.
LEVERAGE_GUIDANCE_X = 6.0
LEVERAGE_GUIDANCE_SOURCES = ("ecb_leveraged_2017", "us_leveraged_2013")

# Damodaran, January 2026, large non-financial firms: coverage at or above
# ``low`` (and below the next band's) maps to ``rating``. Ordered from the
# weakest band up; the first band has no lower bound.
COVERAGE_BANDS: tuple[tuple[float, str], ...] = (
    (float("-inf"), "D"),
    (0.20, "C"),
    (0.65, "CC"),
    (0.80, "CCC"),
    (1.25, "B-"),
    (1.50, "B"),
    (1.75, "B+"),
    (2.00, "BB"),
    (2.25, "BB+"),
    (2.50, "BBB"),
    (3.00, "A-"),
    (4.25, "A"),
    (5.50, "A+"),
    (6.50, "AA"),
    (8.50, "AAA"),
)

# S&P 2024 study, Table 26: average cumulative default rates (%) after 1..15
# years, by rating at the start. 'CCC/C' is one row in the study.
CUMULATIVE_DEFAULT_PCT: dict[str, tuple[float, ...]] = {
    "AAA": (0.00, 0.03, 0.13, 0.23, 0.34, 0.44, 0.49, 0.57, 0.62, 0.67, 0.70, 0.73, 0.75, 0.81, 0.86),
    "AA+": (0.00, 0.04, 0.04, 0.09, 0.13, 0.18, 0.23, 0.27, 0.32, 0.37, 0.43, 0.48, 0.53, 0.59, 0.65),
    "AA": (0.02, 0.03, 0.08, 0.20, 0.33, 0.44, 0.56, 0.66, 0.74, 0.83, 0.90, 0.95, 1.04, 1.09, 1.14),
    "AA-": (0.02, 0.07, 0.15, 0.21, 0.27, 0.36, 0.42, 0.47, 0.53, 0.59, 0.64, 0.68, 0.70, 0.73, 0.77),
    "A+": (0.04, 0.07, 0.16, 0.27, 0.35, 0.43, 0.52, 0.61, 0.72, 0.83, 0.93, 1.04, 1.16, 1.30, 1.42),
    "A": (0.05, 0.12, 0.19, 0.28, 0.39, 0.53, 0.68, 0.81, 0.96, 1.13, 1.27, 1.37, 1.47, 1.53, 1.66),
    "A-": (0.05, 0.14, 0.22, 0.30, 0.42, 0.55, 0.73, 0.87, 0.97, 1.07, 1.16, 1.27, 1.37, 1.47, 1.55),
    "BBB+": (0.09, 0.23, 0.41, 0.59, 0.79, 1.01, 1.18, 1.37, 1.60, 1.83, 2.04, 2.19, 2.36, 2.55, 2.75),
    "BBB": (0.13, 0.33, 0.51, 0.80, 1.09, 1.38, 1.68, 1.95, 2.24, 2.50, 2.77, 2.99, 3.20, 3.32, 3.51),
    "BBB-": (0.21, 0.63, 1.17, 1.76, 2.40, 2.93, 3.39, 3.83, 4.17, 4.50, 4.84, 5.15, 5.42, 5.81, 6.13),
    "BB+": (0.27, 0.83, 1.50, 2.21, 2.90, 3.58, 4.16, 4.55, 5.00, 5.51, 5.87, 6.29, 6.69, 7.01, 7.45),
    "BB": (0.44, 1.38, 2.67, 3.84, 5.05, 6.06, 6.96, 7.78, 8.59, 9.33, 10.08, 10.66, 11.12, 11.42, 11.74),
    "BB-": (0.87, 2.71, 4.62, 6.57, 8.28, 9.92, 11.31, 12.67, 13.75, 14.67, 15.36, 16.07, 16.71, 17.29, 17.85),
    "B+": (1.83, 4.97, 8.05, 10.72, 12.88, 14.57, 16.11, 17.41, 18.57, 19.62, 20.52, 21.14, 21.83, 22.50, 23.14),
    "B": (2.69, 6.36, 9.68, 12.44, 14.69, 16.70, 18.10, 19.16, 20.13, 21.04, 21.64, 22.25, 22.72, 23.09, 23.52),
    "B-": (5.16, 11.20, 15.94, 19.35, 22.03, 23.91, 25.28, 26.36, 27.19, 27.90, 28.87, 29.50, 30.02, 30.55, 30.99),
    "CCC/C": (26.12, 35.92, 41.32, 44.35, 46.53, 47.57, 48.61, 49.29, 49.89, 50.43, 50.85, 51.32, 51.86, 52.26, 52.30),
}

# Ratings at or above this one are investment grade (S&P's definition); a
# rating below it is speculative grade.
LOWEST_INVESTMENT_GRADE = "BBB-"
RATING_ORDER = tuple(CUMULATIVE_DEFAULT_PCT)   # strongest first

# Damodaran's coverage table names CCC, CC, C and D; S&P's study pools 'CCC'
# to 'C' into one row and has no row for an issuer already in default, so the
# weakest bands all read the 'CCC/C' row (the answer says which row it read).
STUDY_ROW = {"CCC": "CCC/C", "CC": "CCC/C", "C": "CCC/C", "D": "CCC/C"}


def rating_for_coverage(coverage: float) -> tuple[str, float, float | None]:
    """The coverage band's rating and its bounds (``high`` None for the top band)."""
    for i, (low, rating) in enumerate(COVERAGE_BANDS):
        high = COVERAGE_BANDS[i + 1][0] if i + 1 < len(COVERAGE_BANDS) else None
        if high is None or coverage < high:
            return rating, low, high
    raise AssertionError("unreachable: the last band has no upper bound")


def study_row(rating: str) -> str:
    return STUDY_ROW.get(rating, rating)


def is_speculative(rating: str) -> bool:
    row = study_row(rating)
    return RATING_ORDER.index(row) > RATING_ORDER.index(LOWEST_INVESTMENT_GRADE)


def cumulative_default_pct(rating: str, years: int) -> float:
    """S&P's average cumulative default rate (%) for ``rating`` after ``years``
    (1-15; a longer hold reads the study's last column)."""
    row = CUMULATIVE_DEFAULT_PCT[study_row(rating)]
    return row[min(max(int(years), 1), len(row)) - 1]
