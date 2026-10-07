"""Base rates (PLAN.md 4.5): transcribed tables, checked against their sources.

A transcription slip can't be caught by recomputing anything, so these tests
use the sources' own summary rows instead: S&P's Table 4 (the minimum,
maximum and median of each rating's annual series), Table 5's descriptive
rows, the identical rows of Tables 24 and 26, and Global Credit Data's
totals, which its breakdowns must add up to. The rest check the tables hang
together and that the staleness rule and the region lookup work.
"""
from datetime import date
from statistics import median

import pytest

from core import risk_sources
from library import base_rates as br

RATINGS = br.RATINGS


def column(rating: str) -> list[float]:
    i = RATINGS.index(rating)
    return [row[i] for row in br.ANNUAL_BY_RATING_YEARS.values()]


# S&P 2024 study, Table 4: descriptive statistics on one-year global default rates
TABLE_4 = {
    "minimum": (0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00),
    "maximum": (0.00, 0.38, 0.38, 0.99, 4.24, 13.84, 49.46),
    "median": (0.00, 0.00, 0.00, 0.06, 0.46, 3.15, 25.75),
    "2008": (0.00, 0.38, 0.38, 0.49, 0.82, 4.09, 27.27),
    "latest_four_quarters": (0.00, 0.00, 0.00, 0.05, 0.17, 1.72, 28.36),
}


def test_every_year_from_1981_to_2024_is_there_once():
    for table in (br.ANNUAL_BY_RATING_YEARS, br.ANNUAL_BY_GRADE_YEARS, br.SPECULATIVE_BY_REGION_YEARS):
        assert list(table) == list(range(1981, 2025))


@pytest.mark.parametrize("rating", RATINGS)
def test_annual_rates_have_the_minimum_maximum_and_median_table_4_prints(rating):
    i = RATINGS.index(rating)
    rates = column(rating)
    assert min(rates) == TABLE_4["minimum"][i]
    assert max(rates) == TABLE_4["maximum"][i]
    # The median of 44 years is the mean of the middle two, printed to two
    # decimals (CCC/C's is 25.755, printed 25.75)
    assert median(rates) == pytest.approx(TABLE_4["median"][i], abs=0.0051)


def test_2008_and_2024_are_the_rows_table_4_repeats():
    assert br.ANNUAL_BY_RATING_YEARS[2008] == TABLE_4["2008"]
    assert br.ANNUAL_BY_RATING_YEARS[2024] == TABLE_4["latest_four_quarters"]


def test_us_speculative_grade_series_has_table_5s_descriptive_rows():
    us = [row[0] for row in br.SPECULATIVE_BY_REGION_YEARS.values()]
    assert sum(us) / len(us) == pytest.approx(4.10, abs=0.005)
    assert median(us) == pytest.approx(3.30, abs=0.005)
    assert (min(us), max(us)) == (0.63, 11.78)


def test_other_regions_maxima_are_table_5s_over_its_1994_2004_window():
    window = [row for y, row in br.SPECULATIVE_BY_REGION_YEARS.items() if 1994 <= y <= 2004]
    for i, printed in ((1, 12.59), (2, 17.26), (3, 9.52)):
        assert max(r[i] for r in window if r[i] is not None) == printed


def test_emerging_markets_has_no_figure_where_the_study_prints_na():
    missing = [y for y, row in br.SPECULATIVE_BY_REGION_YEARS.items() if row[2] is None]
    assert missing == [1981, 1982, 1983, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992]


def test_defaults_by_grade_never_exceed_the_total():
    for year, (total, ig, sg, *_rates) in br.ANNUAL_BY_GRADE_YEARS.items():
        assert ig + sg <= total, year


def test_global_cumulative_rows_equal_table_26s_for_the_same_ratings():
    for rating in ("AAA", "CCC/C"):
        assert br.CUMULATIVE["global"][rating] == risk_sources.CUMULATIVE_DEFAULT_PCT[rating]


@pytest.mark.parametrize("region", ["global", "us", "europe", "emerging"])
def test_cumulative_rates_never_fall_with_the_horizon(region):
    for key, rates in br.CUMULATIVE[region].items():
        assert list(rates) == sorted(rates), (region, key)
        assert len(rates) == len(br.CUMULATIVE[region]["AAA"]), (region, key)


@pytest.mark.parametrize("region", ["global", "us"])
def test_a_weaker_rating_defaults_at_least_as_often_at_every_horizon(region):
    """From 'A' down. 'AAA' issuers have defaulted a little more often than
    'AA' ones in the first years (S&P's own rows say so), so the top two are
    left out rather than the data bent to fit."""
    rows = br.CUMULATIVE[region]
    assert br.CUMULATIVE["global"]["AAA"][2] > br.CUMULATIVE["global"]["AA"][2]
    ordered = RATINGS[RATINGS.index("A"):]
    for strong, weak in zip(ordered, ordered[1:]):
        assert all(a <= b for a, b in zip(rows[strong], rows[weak])), (region, strong, weak)


def test_recovery_breakdowns_add_up_to_global_credit_datas_totals():
    seniority = br.LGD_BY_SENIORITY
    total = seniority["total"][0]
    assert seniority["secured"][0] + seniority["unsecured"][0] == total
    assert seniority["secured_primary"][0] + seniority["secured_secondary"][0] == seniority["secured"][0]
    assert sum(seniority[k][0] for k in ("unsecured_senior", "unsecured_subordinated", "unsecured_other")) \
        == seniority["unsecured"][0]
    assert sum(n for n, _ in br.LGD_BY_REGION.values()) == total
    assert sum(n for n, _ in br.LGD_BY_YEAR.values()) == total
    # Averages weighted by borrowers come back to the printed total within its rounding
    for table in (br.LGD_BY_REGION, br.LGD_BY_YEAR):
        weighted = sum(n * lgd for n, lgd in table.values()) / total
        assert weighted == pytest.approx(seniority["total"][1], abs=1.0)


def test_subordinated_lenders_lose_more_than_senior_and_secured_less_than_unsecured():
    s = br.LGD_BY_SENIORITY
    assert s["unsecured_subordinated"][1] > s["unsecured_senior"][1]
    assert s["secured"][1] < s["unsecured"][1]


def test_an_edition_is_stale_a_year_after_it_was_last_confirmed():
    for source in br.SOURCES:
        assert not br.stale(source, date(2026, 10, 7))
        assert not br.stale(source, date(2027, 10, 6))
        assert br.stale(source, date(2027, 10, 7))


@pytest.mark.parametrize("country,region", [
    ("US", "us"), ("ky", "us"), ("GB", "europe"), ("DE", "europe"), ("JP", "other_developed"),
    ("CA", "other_developed"), ("IN", "emerging"), ("BR", "emerging"), (None, None), ("", None),
    # Retired codes an older account may hold read as their current country
    ("UK", "europe"), ("DD", "europe"),
])
def test_a_country_is_read_into_s_and_ps_region(country, region):
    assert br.sp_region(country) == region


def test_every_table_names_a_source_and_every_source_a_url_sample_and_check_date():
    for table in br.TABLES.values():
        assert table["source"] in br.SOURCES
    for source in br.SOURCES.values():
        assert source["url"].startswith("https://")
        assert source["sample"]["count"] > 0
        date.fromisoformat(source["checked_on"])


def test_the_answer_lists_each_series_in_year_order_with_its_source():
    answer = br.base_rates(date(2026, 10, 7), "gb")
    assert answer["country"] == "GB" and answer["sp_region"] == "europe"
    by_rating = answer["default"]["annual_by_rating"]
    assert by_rating["years"][0] == 1981 and by_rating["rates"]["B"][-1] == 1.72
    assert answer["default"]["speculative_by_region"]["rates"]["europe"][-1] == 4.47
    assert answer["default"]["annual_by_grade"]["rates"]["speculative_grade"][-1] == 3.94
    assert {s["id"] for s in answer["sources"]} == set(br.SOURCES)
    assert all(not s["stale"] for s in answer["sources"])
    assert answer["recovery"]["by_region"][-1] == {"key": "unknown", "defaults": 79, "lgd_pct": 44.0}


def test_the_daily_check_passes_until_an_edition_is_due_and_names_it(capsys):
    from ops import check_base_rates
    assert check_base_rates.main(["--today", "2026-10-07"]) == 0
    assert check_base_rates.main(["--today", "2027-10-07"]) == 1
    out = capsys.readouterr().out
    assert "sp_default_study_2024" in out and "gcd_lgd_2020" in out and "due 2027-10-07" in out
