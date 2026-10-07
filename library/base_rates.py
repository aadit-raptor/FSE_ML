"""Base rates: how often companies default, and how much lenders get back (PLAN.md 4.5).

Part of the optional reference library: shown on Library -> Base rates and
hidden with it (``library.switch``). Nothing in the deal model, the simulation
or the risk warnings reads these tables, so switching the library off changes
no result.

Every figure is **transcribed** from a published table named in ``TABLES``,
never estimated: to change one, transcribe a newer edition of its source and
update the source's ``published`` and ``checked_on``. The tables are long-run
historical averages and annual series, so they move slowly; ``checked_on`` is
the day someone last confirmed the edition is the newest one free to read,
and ``stale`` turns true a year after it (shown on screen and failed by
``ops/check_base_rates.py`` in the daily production check).

``tests/test_base_rates.py`` checks the transcription against the sources'
own summary rows (Table 4's minimum, maximum and median per rating, Table 5's
maxima, Table 2's and Table 4's totals) and that the tables hang together.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from benchmarks.catalogue import canonical
from core.risk_sources import SOURCES as RISK_SOURCES

# How long a confirmed edition counts as current before someone checks again
RECHECK_MONTHS = 12

SOURCES: dict[str, dict] = {
    # The same document the risk warnings read (core/risk_sources.py)
    "sp_default_study_2024": {
        **{k: v for k, v in RISK_SOURCES["sp_default_study_2024"].items() if k != "detail"},
        "kind": "default",
        "checked_on": "2026-10-07",
        "note": "S&P publishes the study each spring. The 2025 study (published 2026) had no free "
                "copy when this edition was last checked.",
    },
    "gcd_lgd_2020": {
        "publisher": "Global Credit Data",
        "title": "LGD Report 2020: Large Corporate Borrowers",
        "published": "2020-06-01",
        "url": "https://globalcreditdata.org/library/lgd-report-large-corporates-2020/",
        "sample": {"count": 11527, "what": "defaulted borrowers", "first_year": 2000, "last_year": 2016},
        "kind": "recovery",
        "checked_on": "2026-10-07",
        "note": "Loans to large corporate borrowers (sales or assets above EUR 50m) from 58 banks; "
                "resolved defaults only, discounted at 3-month EURIBOR. The newest edition free to "
                "read; later editions go to the consortium's member banks.",
    },
}

# Which table each series comes from
TABLES: dict[str, dict] = {
    "annual_by_rating": {"source": "sp_default_study_2024", "table": "3",
                         "title": "Global corporate annual default rates by rating category"},
    "annual_by_grade": {"source": "sp_default_study_2024", "table": "1",
                        "title": "Global corporate default summary"},
    "speculative_by_region": {"source": "sp_default_study_2024", "table": "5",
                              "title": "Annual speculative-grade corporate default rate by region"},
    "cumulative_global": {"source": "sp_default_study_2024", "table": "24",
                          "title": "Global corporate average cumulative default rates, 1981-2024"},
    "cumulative_by_region": {"source": "sp_default_study_2024", "table": "25",
                             "title": "Average cumulative default rates for corporate entities by region, 1981-2024"},
    "lgd_by_seniority": {"source": "gcd_lgd_2020", "table": "2", "title": "Seniority and collateral"},
    "lgd_by_year": {"source": "gcd_lgd_2020", "table": "3", "title": "LGD by year of default"},
    "lgd_by_region": {"source": "gcd_lgd_2020", "table": "4", "title": "LGD by region"},
}

RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC/C")
GRADES = ("investment_grade", "speculative_grade", "all_rated")

# S&P's regions (Table 5's notes), by the issuer's country. Europe and "other
# developed" are listed by the study; every country not listed below is in
# emerging and frontier markets. Tables 5 and 25 name the same regions, but
# Table 25 has no "other developed" block.
SP_REGIONS = ("us", "europe", "emerging", "other_developed")
SP_REGION_MEMBERS: dict[str, frozenset[str]] = {
    # The U.S. and tax havens: the U.S., Bermuda and the Cayman Islands
    "us": frozenset({"US", "BM", "KY"}),
    # Austria ... the U.K.; the Channel Islands are Jersey and Guernsey
    "europe": frozenset({
        "AT", "BE", "VG", "BG", "JE", "GG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GI", "GR",
        "HU", "IS", "IE", "IM", "IT", "LV", "LI", "LT", "LU", "MT", "MD", "MC", "ME", "NL", "NO", "PL",
        "PT", "RO", "SK", "SI", "ES", "SE", "CH", "GB",
    }),
    "other_developed": frozenset({"AU", "BN", "CA", "IL", "JP", "KR", "NZ", "SG"}),
}

# Global Credit Data's regions (Table 4), by the borrower's country of
# residence; the report does not list their members
GCD_REGIONS = ("africa_middle_east", "asia_oceania", "europe", "latin_america", "north_america", "unknown")

# Table 3: annual default rate (%) by rating category at the start of the year
ANNUAL_BY_RATING_YEARS = {
    1981: (0.0, 0.0, 0.0, 0.0, 0.0, 2.33, 0.0),
    1982: (0.0, 0.0, 0.21, 0.35, 4.24, 3.18, 21.43),
    1983: (0.0, 0.0, 0.0, 0.34, 1.15, 4.7, 6.67),
    1984: (0.0, 0.0, 0.0, 0.68, 1.13, 3.49, 25.0),
    1985: (0.0, 0.0, 0.0, 0.0, 1.48, 6.53, 15.38),
    1986: (0.0, 0.0, 0.18, 0.34, 0.88, 8.77, 23.08),
    1987: (0.0, 0.0, 0.0, 0.0, 0.38, 3.12, 12.28),
    1988: (0.0, 0.0, 0.0, 0.0, 1.05, 3.68, 20.37),
    1989: (0.0, 0.0, 0.18, 0.61, 0.72, 3.41, 33.33),
    1990: (0.0, 0.0, 0.0, 0.58, 3.56, 8.56, 31.25),
    1991: (0.0, 0.0, 0.0, 0.55, 1.67, 13.84, 33.87),
    1992: (0.0, 0.0, 0.0, 0.0, 0.0, 6.99, 30.19),
    1993: (0.0, 0.0, 0.0, 0.0, 0.7, 2.62, 13.33),
    1994: (0.0, 0.0, 0.14, 0.0, 0.28, 3.07, 16.67),
    1995: (0.0, 0.0, 0.0, 0.17, 1.0, 4.57, 28.0),
    1996: (0.0, 0.0, 0.0, 0.0, 0.45, 2.9, 8.0),
    1997: (0.0, 0.0, 0.0, 0.25, 0.19, 3.5, 12.0),
    1998: (0.0, 0.0, 0.0, 0.41, 0.98, 4.65, 42.86),
    1999: (0.0, 0.17, 0.18, 0.19, 0.95, 7.45, 33.82),
    2000: (0.0, 0.0, 0.26, 0.36, 1.15, 7.67, 35.96),
    2001: (0.0, 0.0, 0.26, 0.33, 2.91, 11.34, 45.45),
    2002: (0.0, 0.0, 0.0, 0.99, 2.83, 8.11, 44.19),
    2003: (0.0, 0.0, 0.0, 0.22, 0.57, 4.03, 32.53),
    2004: (0.0, 0.0, 0.08, 0.0, 0.44, 1.45, 15.83),
    2005: (0.0, 0.0, 0.0, 0.07, 0.31, 1.74, 9.02),
    2006: (0.0, 0.0, 0.0, 0.0, 0.3, 0.81, 13.33),
    2007: (0.0, 0.0, 0.0, 0.0, 0.2, 0.25, 15.24),
    2008: (0.0, 0.38, 0.38, 0.49, 0.82, 4.09, 27.27),
    2009: (0.0, 0.0, 0.22, 0.55, 0.76, 10.87, 49.46),
    2010: (0.0, 0.0, 0.0, 0.0, 0.59, 0.86, 22.83),
    2011: (0.0, 0.0, 0.0, 0.07, 0.0, 1.68, 16.54),
    2012: (0.0, 0.0, 0.0, 0.0, 0.3, 1.57, 27.52),
    2013: (0.0, 0.0, 0.0, 0.0, 0.1, 1.52, 24.67),
    2014: (0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 17.42),
    2015: (0.0, 0.0, 0.0, 0.0, 0.16, 2.41, 26.51),
    2016: (0.0, 0.0, 0.0, 0.06, 0.47, 3.74, 33.0),
    2017: (0.0, 0.0, 0.0, 0.0, 0.08, 1.0, 26.56),
    2018: (0.0, 0.0, 0.0, 0.0, 0.0, 0.94, 27.18),
    2019: (0.0, 0.0, 0.0, 0.11, 0.0, 1.5, 29.61),
    2020: (0.0, 0.0, 0.0, 0.0, 0.94, 3.55, 47.88),
    2021: (0.0, 0.0, 0.0, 0.0, 0.0, 0.52, 10.96),
    2022: (0.0, 0.0, 0.0, 0.0, 0.32, 1.1, 13.84),
    2023: (0.0, 0.0, 0.0, 0.11, 0.25, 1.25, 30.89),
    2024: (0.0, 0.0, 0.0, 0.05, 0.17, 1.72, 28.36),
}

# Table 1: defaults (number) and default rates (%) by grade
ANNUAL_BY_GRADE_YEARS = {
    1981: (2, 0, 2, 0.15, 0.0, 0.63),
    1982: (18, 2, 15, 1.22, 0.19, 4.46),
    1983: (12, 1, 10, 0.77, 0.09, 2.96),
    1984: (14, 2, 12, 0.93, 0.17, 3.29),
    1985: (19, 0, 18, 1.12, 0.0, 4.34),
    1986: (34, 2, 30, 1.73, 0.15, 5.73),
    1987: (19, 0, 19, 0.94, 0.0, 2.82),
    1988: (32, 0, 29, 1.38, 0.0, 3.88),
    1989: (44, 3, 35, 1.77, 0.21, 4.7),
    1990: (70, 2, 56, 2.71, 0.14, 8.1),
    1991: (93, 2, 65, 3.22, 0.13, 11.02),
    1992: (39, 0, 32, 1.49, 0.0, 6.1),
    1993: (26, 0, 14, 0.6, 0.0, 2.5),
    1994: (21, 1, 15, 0.62, 0.05, 2.12),
    1995: (35, 1, 29, 1.05, 0.05, 3.54),
    1996: (20, 0, 16, 0.51, 0.0, 1.81),
    1997: (23, 2, 20, 0.63, 0.08, 2.01),
    1998: (57, 4, 49, 1.3, 0.14, 3.75),
    1999: (110, 5, 93, 2.16, 0.17, 5.63),
    2000: (136, 7, 109, 2.46, 0.24, 6.21),
    2001: (230, 7, 172, 3.7, 0.23, 9.7),
    2002: (226, 13, 159, 3.52, 0.41, 9.35),
    2003: (120, 3, 89, 1.88, 0.1, 4.97),
    2004: (56, 1, 38, 0.77, 0.03, 2.02),
    2005: (40, 1, 31, 0.6, 0.03, 1.5),
    2006: (30, 0, 26, 0.47, 0.0, 1.18),
    2007: (24, 0, 21, 0.37, 0.0, 0.91),
    2008: (127, 14, 89, 1.79, 0.42, 3.71),
    2009: (268, 11, 223, 4.15, 0.33, 9.89),
    2010: (83, 0, 64, 1.2, 0.0, 3.02),
    2011: (53, 1, 44, 0.8, 0.03, 1.85),
    2012: (83, 0, 66, 1.13, 0.0, 2.59),
    2013: (81, 0, 62, 1.02, 0.0, 2.23),
    2014: (60, 0, 45, 0.69, 0.0, 1.44),
    2015: (113, 0, 94, 1.36, 0.0, 2.77),
    2016: (163, 1, 143, 2.08, 0.03, 4.23),
    2017: (95, 0, 83, 1.21, 0.0, 2.47),
    2018: (82, 0, 71, 1.02, 0.0, 2.07),
    2019: (118, 2, 92, 1.31, 0.06, 2.55),
    2020: (225, 0, 198, 2.76, 0.0, 5.54),
    2021: (72, 0, 60, 0.85, 0.0, 1.68),
    2022: (85, 0, 71, 0.99, 0.0, 1.94),
    2023: (153, 2, 128, 1.86, 0.06, 3.71),
    2024: (145, 1, 129, 1.91, 0.03, 3.94),
}

# Table 5: speculative-grade default rate (%) by region; None where the study prints N.A.
SPECULATIVE_BY_REGION_YEARS = {
    1981: (0.63, 0.0, None, 0.0),
    1982: (4.49, 0.0, None, 0.0),
    1983: (3.0, 0.0, None, 0.0),
    1984: (3.35, 0.0, 0.0, 0.0),
    1985: (4.43, 0.0, None, 0.0),
    1986: (5.81, 0.0, None, 0.0),
    1987: (2.87, 0.0, None, 0.0),
    1988: (3.92, 0.0, None, 0.0),
    1989: (4.36, 0.0, None, 37.5),
    1990: (7.93, 0.0, None, 28.57),
    1991: (10.69, 50.0, None, 25.0),
    1992: (6.25, 0.0, None, 0.0),
    1993: (2.4, 20.0, 0.0, 0.0),
    1994: (2.21, 0.0, 0.0, 0.0),
    1995: (3.66, 9.09, 0.0, 0.0),
    1996: (1.86, 0.0, 0.0, 2.7),
    1997: (2.18, 0.0, 0.0, 1.92),
    1998: (3.26, 0.0, 8.9, 2.41),
    1999: (5.34, 6.32, 7.35, 4.46),
    2000: (7.38, 2.56, 2.07, 5.22),
    2001: (10.53, 8.46, 6.57, 9.52),
    2002: (7.24, 12.59, 17.26, 4.35),
    2003: (5.59, 3.73, 3.83, 3.42),
    2004: (2.44, 1.62, 0.83, 1.84),
    2005: (2.02, 0.95, 0.24, 1.21),
    2006: (1.37, 1.81, 0.43, 0.69),
    2007: (1.02, 0.96, 0.2, 2.08),
    2008: (4.29, 2.53, 2.19, 4.23),
    2009: (11.78, 8.16, 5.95, 8.96),
    2010: (3.46, 1.02, 1.55, 7.32),
    2011: (2.15, 1.6, 0.38, 3.35),
    2012: (2.65, 2.24, 2.36, 3.33),
    2013: (2.19, 2.88, 1.81, 2.7),
    2014: (1.61, 0.97, 1.3, 2.11),
    2015: (2.85, 2.11, 3.13, 2.75),
    2016: (5.2, 1.95, 3.65, 4.23),
    2017: (3.09, 2.6, 0.92, 2.63),
    2018: (2.42, 1.97, 1.25, 2.64),
    2019: (3.12, 2.27, 1.99, 0.72),
    2020: (6.66, 5.44, 3.24, 4.4),
    2021: (1.54, 1.84, 1.87, 1.54),
    2022: (1.66, 2.22, 2.39, 1.54),
    2023: (4.48, 3.53, 2.26, 2.3),
    2024: (5.13, 4.47, 1.22, 1.2),
}

# Tables 24 (global) and 25 (by region): average cumulative default rate (%) after 1, 2, ... years
CUMULATIVE = {
    'global': {
        'AAA': (0.0, 0.03, 0.13, 0.23, 0.34, 0.44, 0.49, 0.57, 0.62, 0.67, 0.7, 0.73, 0.75, 0.81, 0.86),
        'AA': (0.02, 0.05, 0.11, 0.19, 0.28, 0.37, 0.45, 0.52, 0.59, 0.65, 0.71, 0.76, 0.81, 0.86, 0.9),
        'A': (0.05, 0.11, 0.19, 0.29, 0.39, 0.51, 0.65, 0.78, 0.9, 1.03, 1.14, 1.25, 1.35, 1.45, 1.56),
        'BBB': (0.14, 0.38, 0.67, 1.01, 1.36, 1.71, 2.0, 2.3, 2.58, 2.86, 3.13, 3.35, 3.56, 3.78, 4.01),
        'BB': (0.56, 1.76, 3.12, 4.48, 5.75, 6.93, 7.94, 8.86, 9.68, 10.44, 11.06, 11.65, 12.17, 12.6, 13.05),
        'B': (2.93, 6.93, 10.46, 13.31, 15.6, 17.45, 18.9, 20.06, 21.08, 22.02, 22.82, 23.43, 24.02, 24.57, 25.11),
        'CCC/C': (26.12, 35.92, 41.32, 44.35, 46.53, 47.57, 48.61, 49.29, 49.89, 50.43, 50.85, 51.32, 51.86, 52.26, 52.3),
        'investment_grade': (0.08, 0.21, 0.37, 0.57, 0.77, 0.98, 1.17, 1.34, 1.52, 1.69, 1.85, 1.98, 2.11, 2.24, 2.38),
        'speculative_grade': (3.54, 6.78, 9.55, 11.79, 13.64, 15.15, 16.39, 17.41, 18.32, 19.15, 19.85, 20.44, 20.99, 21.47, 21.93),
        'all_rated': (1.5, 2.91, 4.13, 5.15, 6.0, 6.72, 7.31, 7.81, 8.26, 8.67, 9.02, 9.31, 9.58, 9.83, 10.07),
    },
    'us': {
        'AAA': (0.0, 0.04, 0.16, 0.28, 0.4, 0.53, 0.57, 0.65, 0.73, 0.81, 0.85, 0.89, 0.93, 1.01, 1.1),
        'AA': (0.03, 0.07, 0.16, 0.28, 0.39, 0.53, 0.65, 0.76, 0.86, 0.96, 1.05, 1.12, 1.19, 1.26, 1.34),
        'A': (0.06, 0.17, 0.3, 0.45, 0.61, 0.79, 0.99, 1.18, 1.37, 1.57, 1.75, 1.92, 2.09, 2.23, 2.39),
        'BBB': (0.18, 0.48, 0.83, 1.26, 1.73, 2.19, 2.61, 3.01, 3.42, 3.8, 4.18, 4.46, 4.73, 5.04, 5.34),
        'BB': (0.67, 2.11, 3.82, 5.49, 7.01, 8.46, 9.71, 10.88, 11.92, 12.9, 13.71, 14.49, 15.17, 15.71, 16.26),
        'B': (3.16, 7.52, 11.43, 14.56, 17.08, 19.14, 20.74, 22.03, 23.16, 24.2, 25.07, 25.76, 26.42, 27.04, 27.63),
        'CCC/C': (27.91, 38.94, 44.92, 48.51, 51.07, 52.27, 53.57, 54.32, 55.07, 55.7, 56.31, 56.79, 57.29, 57.7, 57.7),
        'investment_grade': (0.11, 0.28, 0.49, 0.75, 1.02, 1.31, 1.57, 1.83, 2.08, 2.33, 2.57, 2.75, 2.94, 3.12, 3.31),
        'speculative_grade': (3.99, 7.74, 10.98, 13.59, 15.72, 17.49, 18.93, 20.13, 21.2, 22.18, 23.01, 23.71, 24.36, 24.93, 25.46),
        'all_rated': (1.85, 3.61, 5.16, 6.44, 7.52, 8.43, 9.19, 9.84, 10.42, 10.96, 11.43, 11.81, 12.17, 12.5, 12.81),
    },
    'europe': {
        'AAA': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        'AA': (0.0, 0.02, 0.05, 0.09, 0.14, 0.19, 0.22, 0.24, 0.27, 0.27),
        'A': (0.03, 0.06, 0.08, 0.13, 0.19, 0.24, 0.31, 0.34, 0.35, 0.36),
        'BBB': (0.05, 0.14, 0.28, 0.41, 0.56, 0.76, 0.93, 1.09, 1.27, 1.43),
        'BB': (0.36, 1.15, 1.9, 2.61, 3.47, 4.29, 4.94, 5.35, 5.72, 6.13),
        'B': (1.75, 4.72, 7.47, 9.82, 11.88, 13.46, 14.67, 15.53, 16.28, 16.81),
        'CCC/C': (26.26, 36.23, 41.57, 45.22, 47.27, 47.82, 48.12, 48.45, 48.45, 48.93),
        'investment_grade': (0.03, 0.08, 0.14, 0.21, 0.3, 0.39, 0.48, 0.55, 0.61, 0.66),
        'speculative_grade': (2.91, 5.52, 7.67, 9.44, 10.99, 12.19, 13.1, 13.73, 14.26, 14.73),
        'all_rated': (0.88, 1.67, 2.31, 2.84, 3.3, 3.67, 3.96, 4.15, 4.32, 4.45),
    },
    'emerging': {
        'AAA': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        'AA': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        'A': (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
        'BBB': (0.1, 0.41, 0.76, 1.17, 1.55, 1.78, 1.91, 2.02, 2.09, 2.11),
        'BB': (0.54, 1.51, 2.55, 3.55, 4.42, 5.09, 5.67, 6.18, 6.64, 7.0),
        'B': (2.98, 6.09, 8.46, 10.45, 11.96, 13.11, 14.14, 15.01, 15.75, 16.52),
        'CCC/C': (17.58, 22.81, 25.84, 26.52, 27.69, 28.6, 29.37, 30.17, 30.68, 31.04),
        'investment_grade': (0.08, 0.28, 0.52, 0.8, 1.05, 1.21, 1.3, 1.37, 1.42, 1.44),
        'speculative_grade': (2.53, 4.61, 6.29, 7.67, 8.81, 9.69, 10.46, 11.14, 11.71, 12.24),
        'all_rated': (1.44, 2.69, 3.73, 4.63, 5.39, 5.96, 6.44, 6.86, 7.22, 7.54),
    },
}

# Global Credit Data, obligor level: (defaulted borrowers, LGD %). Recovery is
# what is left: 100 - LGD.
LGD_BY_SENIORITY = {
    "secured": (7261, 22.0),
    "secured_primary": (2489, 20.0),
    "secured_secondary": (4772, 23.0),
    "unsecured": (4266, 27.0),
    "unsecured_senior": (3838, 26.0),
    "unsecured_subordinated": (128, 38.0),
    "unsecured_other": (300, 34.0),
    "total": (11527, 24.0),
}
LGD_BY_YEAR = {
    2000: (456, 35.0), 2001: (837, 33.0), 2002: (875, 29.0), 2003: (654, 23.0), 2004: (291, 20.0),
    2005: (344, 19.0), 2006: (346, 19.0), 2007: (412, 29.0), 2008: (1151, 31.0), 2009: (1926, 20.0),
    2010: (1019, 19.0), 2011: (727, 22.0), 2012: (765, 19.0), 2013: (549, 20.0), 2014: (350, 24.0),
    2015: (351, 28.0), 2016: (474, 13.0),
}
LGD_BY_REGION = {
    "africa_middle_east": (286, 20.0),
    "asia_oceania": (1023, 31.0),
    "europe": (4089, 21.0),
    "latin_america": (1269, 31.0),
    "north_america": (4781, 23.0),
    "unknown": (79, 44.0),
}
LGD_TOTAL = LGD_BY_SENIORITY["total"]


def sp_region(country: Optional[str]) -> Optional[str]:
    """S&P's region for an ISO country code (a retired one, such as UK or DD,
    read as its current country); None without a country."""
    if not country:
        return None
    code = canonical(country.strip().upper())
    for region, members in SP_REGION_MEMBERS.items():
        if code in members:
            return region
    return "emerging"


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    return date(year, month, min(day.day, 28))


def recheck_due(source_id: str) -> date:
    return _add_months(date.fromisoformat(SOURCES[source_id]["checked_on"]), RECHECK_MONTHS)


def stale(source_id: str, today: date) -> bool:
    """True once a year has passed since the edition was last confirmed as the newest free one."""
    return today >= recheck_due(source_id)


def _source(source_id: str, today: date) -> dict:
    src = SOURCES[source_id]
    return {"id": source_id, **src, "recheck_due": recheck_due(source_id).isoformat(),
            "stale": stale(source_id, today)}


def _years(rows: dict) -> list[int]:
    return sorted(rows)


def base_rates(today: date, country: Optional[str] = None) -> dict:
    """Every table with its source, in the shape the API answers."""
    by_rating_years = _years(ANNUAL_BY_RATING_YEARS)
    by_grade_years = _years(ANNUAL_BY_GRADE_YEARS)
    region_years = _years(SPECULATIVE_BY_REGION_YEARS)
    return {
        "sources": [_source(s, today) for s in SOURCES],
        "tables": [{"id": k, **v} for k, v in TABLES.items()],
        "country": country.upper() if country else None,
        "sp_region": sp_region(country),
        "default": {
            "ratings": list(RATINGS),
            "annual_by_rating": {
                "years": by_rating_years,
                "rates": {r: [ANNUAL_BY_RATING_YEARS[y][i] for y in by_rating_years] for i, r in enumerate(RATINGS)},
            },
            "annual_by_grade": {
                "years": by_grade_years,
                "defaults": [ANNUAL_BY_GRADE_YEARS[y][0] for y in by_grade_years],
                "rates": {g: [ANNUAL_BY_GRADE_YEARS[y][3 + i] for y in by_grade_years]
                          for i, g in enumerate(("all_rated", "investment_grade", "speculative_grade"))},
            },
            "speculative_by_region": {
                "years": region_years,
                "regions": list(SP_REGIONS),
                "rates": {r: [SPECULATIVE_BY_REGION_YEARS[y][i] for y in region_years] for i, r in enumerate(SP_REGIONS)},
            },
            "cumulative": [
                {"region": region, "table": "cumulative_global" if region == "global" else "cumulative_by_region",
                 "horizons": len(rows["AAA"]), "rates": {k: list(v) for k, v in rows.items()}}
                for region, rows in CUMULATIVE.items()
            ],
        },
        "recovery": {
            "by_seniority": [{"key": k, "defaults": n, "lgd_pct": lgd} for k, (n, lgd) in LGD_BY_SENIORITY.items()],
            "by_year": [{"year": y, "defaults": n, "lgd_pct": lgd} for y, (n, lgd) in LGD_BY_YEAR.items()],
            "by_region": [{"key": k, "defaults": n, "lgd_pct": lgd} for k, (n, lgd) in LGD_BY_REGION.items()],
        },
    }


def coverage() -> list[dict]:
    """What the base rates cover: one row per table (regions, rating bands, years, observations)."""
    def row(table: str, *, regions: int, bands: int, years: tuple[int, int], observations: Optional[int]) -> dict:
        return {"table": table, "source": TABLES[table]["source"], "regions": regions, "bands": bands,
                "first_year": years[0], "last_year": years[1], "observations": observations}
    sp = SOURCES["sp_default_study_2024"]["sample"]
    gcd = SOURCES["gcd_lgd_2020"]["sample"]
    span = (sp["first_year"], sp["last_year"])
    return [
        row("annual_by_rating", regions=1, bands=len(RATINGS), years=span, observations=sp["count"]),
        row("annual_by_grade", regions=1, bands=len(GRADES), years=span, observations=sp["count"]),
        row("speculative_by_region", regions=len(SP_REGIONS), bands=1, years=span, observations=sp["count"]),
        row("cumulative_global", regions=1, bands=len(CUMULATIVE["global"]), years=span, observations=sp["count"]),
        row("cumulative_by_region", regions=len(CUMULATIVE) - 1, bands=len(CUMULATIVE["us"]), years=span,
            observations=sp["count"]),
        row("lgd_by_seniority", regions=1, bands=len(LGD_BY_SENIORITY) - 1,
            years=(gcd["first_year"], gcd["last_year"]), observations=LGD_TOTAL[0]),
        row("lgd_by_year", regions=1, bands=1, years=(min(LGD_BY_YEAR), max(LGD_BY_YEAR)),
            observations=sum(n for n, _ in LGD_BY_YEAR.values())),
        row("lgd_by_region", regions=len(LGD_BY_REGION), bands=1,
            years=(gcd["first_year"], gcd["last_year"]), observations=sum(n for n, _ in LGD_BY_REGION.values())),
    ]
