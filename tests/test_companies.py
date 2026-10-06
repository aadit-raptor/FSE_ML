"""Company filings from many countries (PLAN.md 4.1, companies/).

Every load here replays real responses recorded from the sources
(tests/fixtures/companies, written by ``python -m companies.record``), and
each figure checked is the one printed in that filing: the US (SEC EDGAR),
the UK (a listed company's ESEF report, a private company's Companies House
accounts), the EU (ESEF) and Japan (EDINET, IFRS and Japanese GAAP).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from companies import companies_house, edinet, esef, http, identifiers, ixbrl, sec, sources
from companies.facts import Fact, fiscal_year_of, is_annual, summarize
from companies.items import IFRS_MAP, SUMMARY_FIELDS, ConceptMap, Sum
from companies.model import FilingLink
from companies.record import replay

FIXTURES = Path(__file__).parent / "fixtures" / "companies"
TESCO_LEI = "2138002P5RNKC5W2JZ46"
HEINEKEN_LEI = "724500K5PTPSST86UQ23"


@pytest.fixture(autouse=True)
def fresh_sources(monkeypatch):
    """No keys, empty caches and no network: a test replays what it needs."""
    monkeypatch.delenv(companies_house.KEY_ENV, raising=False)
    monkeypatch.delenv(edinet.KEY_ENV, raising=False)
    sec.reset_cache()
    edinet.reset_cache()
    edinet.use_index(None)
    http.use_transport(lambda req: pytest.fail(f"unrecorded request to {req.host}"))
    yield
    http.use_transport(None)
    edinet.use_index(None)
    sec.reset_cache()
    edinet.reset_cache()


def use(case: str) -> None:
    """Replay one recorded case, from empty caches (its lists are trimmed to it)."""
    sec.reset_cache()
    edinet.reset_cache()
    http.use_transport(replay(FIXTURES / case))


def year(data, fiscal_year):
    return next(y for y in data.years if y.fiscal_year == fiscal_year)


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
def test_lei_and_isin_check_digits_tell_them_from_names_and_tickers():
    assert identifiers.is_lei(TESCO_LEI) and identifiers.is_lei(HEINEKEN_LEI)
    assert not identifiers.is_lei(TESCO_LEI[:-1] + "7")          # one digit off
    assert identifiers.is_isin("GB00BLGZ9862") and identifiers.is_isin("US0378331005")
    assert identifiers.is_isin("JP3633400001")
    assert not identifiers.is_isin("US0378331006")
    assert not identifiers.is_isin("TESCOPLCTESC")
    assert identifiers.is_uk_company_number("00445790") and identifiers.is_uk_company_number("SC036332")
    assert identifiers.is_uk_company_number("445790")              # leading zeros left off
    assert not identifiers.is_uk_company_number("7203")
    assert companies_house.normalise_number("445790") == "00445790"


# ---------------------------------------------------------------------------
# Years, periods and concepts
# ---------------------------------------------------------------------------
def _fact(concept, value, end, start=None, filed=date(2026, 1, 1)):
    return Fact(concept, value, "EUR", end, start, FilingLink(f"https://x/{filed}", filed))


def test_a_year_is_named_by_the_month_it_really_ends():
    assert fiscal_year_of(date(2025, 12, 31)) == (2025, 12)
    assert fiscal_year_of(date(2026, 2, 28)) == (2026, 2)          # Tesco's 53 weeks
    assert fiscal_year_of(date(2025, 1, 3)) == (2024, 12)          # a 52/53-week December year
    assert fiscal_year_of(date(2025, 10, 4)) == (2025, 9)          # Apple's September year


def test_only_flows_over_a_year_count():
    year_flow = _fact("ifrs-full:Revenue", 1, date(2025, 12, 31), date(2025, 1, 1))
    quarter = _fact("ifrs-full:Revenue", 1, date(2025, 12, 31), date(2025, 10, 1))
    weeks53 = _fact("ifrs-full:Revenue", 1, date(2026, 2, 28), date(2025, 2, 23))
    assert is_annual(year_flow) and is_annual(weeks53) and not is_annual(quarter)


def test_the_newest_filing_wins_and_quarters_are_ignored():
    facts = [
        _fact("ifrs-full:Revenue", 100e6, date(2024, 12, 31), date(2024, 1, 1), filed=date(2025, 2, 1)),
        _fact("ifrs-full:Revenue", 104e6, date(2024, 12, 31), date(2024, 1, 1), filed=date(2026, 2, 1)),
        _fact("ifrs-full:Revenue", 30e6, date(2024, 12, 31), date(2024, 10, 1), filed=date(2026, 3, 1)),
        _fact("ifrs-full:Revenue", 110e6, date(2025, 12, 31), date(2025, 1, 1), filed=date(2026, 2, 1)),
    ]
    years, _ = summarize(facts, IFRS_MAP, 3)
    assert [(y.fiscal_year, y.figures["revenue"]) for y in years] == [(2024, 104.0), (2025, 110.0)]
    assert years[0].filing.filed_on == date(2026, 2, 1)           # the link is the restating filing


def test_a_sum_that_needs_every_part_skips_a_year_with_a_part_missing():
    m = ConceptMap("t", {**{k: [] for k in IFRS_MAP.fields},
                         "total_assets": [Sum(("t:A", "t:B", "-t:C"), every=True)],
                         "debt_total": [Sum(("t:D1", "t:D2"))]})
    facts = [_fact("t:A", 10e6, date(2025, 12, 31)), _fact("t:B", 5e6, date(2025, 12, 31)),
             _fact("t:C", 3e6, date(2025, 12, 31)), _fact("t:A", 9e6, date(2024, 12, 31)),
             _fact("t:D1", 2e6, date(2025, 12, 31))]
    years, _ = summarize(facts, m, 3)
    assert [y.fiscal_year for y in years] == [2025]                # 2024 lacks B and C
    assert years[0].figures["total_assets"] == 12.0                 # 10 + 5 - 3
    assert years[0].figures["total_debt"] == 2.0                    # D2 not reported: nothing


# ---------------------------------------------------------------------------
# Inline XBRL
# ---------------------------------------------------------------------------
IXBRL = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
  xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
  xmlns:iso4217="http://www.xbrl.org/2003/iso4217" xmlns:c="http://xbrl.frc.org.uk/fr/2023-01-01/core"
  xmlns:bus="http://xbrl.frc.org.uk/cd/2023-01-01/business">
<body><ix:header><ix:resources>
 <xbrli:context id="y"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier></xbrli:entity>
   <xbrli:period><xbrli:startDate>2024-07-01</xbrli:startDate><xbrli:endDate>2025-06-30</xbrli:endDate></xbrli:period>
 </xbrli:context>
 <xbrli:context id="e"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier></xbrli:entity>
   <xbrli:period><xbrli:instant>2025-06-30</xbrli:instant></xbrli:period></xbrli:context>
 <xbrli:context id="seg"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier>
   <xbrli:segment><xbrldi:explicitMember dimension="bus:AccountingStandardsDimension">bus:FRS102</xbrldi:explicitMember></xbrli:segment>
   </xbrli:entity><xbrli:period><xbrli:instant>2025-06-30</xbrli:instant></xbrli:period></xbrli:context>
 <xbrli:unit id="GBP"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit>
</ix:resources></ix:header>
<p><ix:nonFraction name="c:TurnoverRevenue" contextRef="y" unitRef="GBP" decimals="0" format="ixt:num-dot-decimal">12,345,678</ix:nonFraction></p>
<p><ix:nonFraction name="c:ProfitLoss" contextRef="y" unitRef="GBP" scale="3" sign="-" format="ixt:num-dot-decimal">1,250</ix:nonFraction></p>
<p><ix:nonFraction name="c:Equity" contextRef="e" unitRef="GBP" format="ixt-sec:num-comma-decimal">2.500.000,50</ix:nonFraction></p>
<p><ix:nonFraction name="c:Debtors" contextRef="e" unitRef="GBP" format="ixt:fixed-zero">-</ix:nonFraction></p>
<p><ix:nonFraction name="c:Equity" contextRef="seg" unitRef="GBP" format="ixt:num-dot-decimal">999</ix:nonFraction></p>
<p><ix:nonNumeric name="bus:NameEntityOfficer" contextRef="e">A Person</ix:nonNumeric></p>
</body></html>"""


def test_inline_xbrl_reads_formats_scales_and_signs_and_skips_dimensions():
    facts, members = ixbrl.read(IXBRL, FilingLink("u", None))
    got = {(f.concept, f.end, f.start): f.value for f in facts}
    assert got == {
        ("frc:TurnoverRevenue", date(2025, 6, 30), date(2024, 7, 1)): 12345678.0,
        ("frc:ProfitLoss", date(2025, 6, 30), date(2024, 7, 1)): -1250000.0,
        ("frc:Equity", date(2025, 6, 30), None): 2500000.5,
        ("frc:Debtors", date(2025, 6, 30), None): 0.0,
    }
    assert members == {"FRS102"}
    assert all(f.currency == "GBP" for f in facts)


def test_inline_xbrl_that_is_not_xml_is_refused():
    with pytest.raises(ixbrl.Unreadable):
        ixbrl.read(b"<html><p>not closed</html>", FilingLink("u", None))


# ---------------------------------------------------------------------------
# United States: SEC EDGAR
# ---------------------------------------------------------------------------
def test_us_company_loads_in_us_dollars_under_us_gaap():
    use("sec_mcd")
    data = sources.fetch("sec", "63908")
    assert (data.ref.name, data.ref.country, data.currency, data.accounting_standard) == \
        ("MCDONALDS CORP", "US", "USD", "us_gaap")
    assert data.fiscal_year_end_month == 12 and data.unit == "millions"
    assert [y.fiscal_year for y in data.years] == [2023, 2024, 2025]
    fy25 = year(data, 2025)
    # McDonald's 2025 Form 10-K: revenues, operating income, net income, total assets
    assert fy25.figures["revenue"] == 26885.0
    assert fy25.figures["operating_income"] == 12393.0
    assert fy25.figures["net_income"] == 8563.0
    assert fy25.figures["total_assets"] == 59515.0
    assert fy25.filing.url == ("https://www.sec.gov/Archives/edgar/data/63908/000006390826000035/"
                               "0000063908-26-000035-index.htm")
    assert fy25.filing.form == "10-K" and fy25.filing.filed_on == date(2026, 2, 24)


def test_us_summary_reads_the_same_figures_as_the_edgar_answer():
    """One set of concepts (core/accounting.py) for both readers, so a
    company loaded here and the forecast's EDGAR answer agree."""
    from unittest import mock

    import ml.edgar_extractor as extractor
    raw = json.loads(next((FIXTURES / "sec_mcd").glob("data_CIK0000063908_json_6e00de06.json")).read_bytes())

    class Resp:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def get(url, **_):
        if "company_tickers" in url:
            return Resp({"0": {"cik_str": 63908, "ticker": "MCD", "title": "MCDONALDS CORP"}})
        return Resp({"name": "MCDONALDS CORP"} if "submissions" in url else raw)

    with mock.patch.object(extractor.requests, "get", get), mock.patch.object(extractor.time, "sleep"):
        old = extractor.fetch_financials("MCD", 3)
    use("sec_mcd")
    new = sources.fetch("sec", "63908")
    assert [y.fiscal_year for y in new.years] == old.years
    for name, field in (("revenue", "revenue"), ("operating_income", "operating_income"),
                        ("net_income", "net_income"), ("total_debt", "long_term_debt"),
                        ("cash_and_equivalents", "cash_and_equivalents"), ("total_assets", "total_assets"),
                        ("capital_expenditures", "capital_expenditures"),
                        ("depreciation_amortization", "depreciation_amortization")):
        assert [y.figures[name] for y in new.years] == old.data[field], name


# ---------------------------------------------------------------------------
# United Kingdom (listed) and the EU: ESEF reports
# ---------------------------------------------------------------------------
def test_uk_listed_company_loads_in_pounds_under_ifrs():
    use("esef_tesco")
    data = sources.fetch("esef", TESCO_LEI)
    assert (data.ref.name, data.ref.country, data.currency, data.accounting_standard) == \
        ("TESCO PLC", "GB", "GBP", "ifrs")
    # A 53-week year ending Saturday 28 February 2026
    assert data.fiscal_year_end_month == 2
    assert [(y.fiscal_year, y.period_end) for y in data.years] == [
        (2024, date(2024, 2, 24)), (2025, date(2025, 2, 22)), (2026, date(2026, 2, 28))]
    fy26 = year(data, 2026)
    # Tesco PLC Annual Report 2026: Group income statement and balance sheet (GBP m)
    assert fy26.figures["revenue"] == 73712.0
    assert fy26.figures["operating_income"] == 2985.0
    # No total assets line: non-current 30,991 + current 8,483
    assert fy26.figures["total_assets"] == 39474.0
    assert fy26.figures["total_equity"] == 11457.0
    # IFRS 16 lease liabilities: current 659 + non-current 7,225
    assert fy26.figures["lease_liability"] == 7884.0
    assert fy26.filing.url.startswith(f"https://filings.xbrl.org/{TESCO_LEI}/2026-02-28/ESEF/GB/")
    assert fy26.filing.url.endswith("ixbrlviewer.html")
    # Its capital expenditure is tagged with Tesco's own concept, so it is missing and said so
    assert fy26.figures["capital_expenditures"] is None
    assert [(w.code, w.field) for w in data.warnings] == [("missing_figure", "capital_expenditures")]


def test_eu_company_loads_in_euros_under_ifrs():
    use("esef_heineken")
    data = sources.fetch("esef", HEINEKEN_LEI)
    assert (data.ref.name, data.ref.country, data.currency, data.accounting_standard) == \
        ("Heineken N.V.", "NL", "EUR", "ifrs")
    assert [y.fiscal_year for y in data.years] == [2023, 2024, 2025]
    fy25 = year(data, 2025)
    # Heineken N.V. Annual Report 2025: revenue, operating profit, total assets (EUR m)
    assert fy25.figures["revenue"] == 34257.0
    assert fy25.figures["operating_income"] == 3406.0
    assert fy25.figures["total_assets"] == 53753.0
    assert fy25.filing.url.startswith(f"https://filings.xbrl.org/{HEINEKEN_LEI}/2025-12-31/ESEF/NL/")


def test_xbrl_json_drops_dimensioned_facts_and_moves_midnight_ends_back_a_day():
    doc = {"documentInfo": {"namespaces": {"ifrs-full": "https://xbrl.ifrs.org/taxonomy/2024-03-27/ifrs-full",
                                           "x": "http://example.com/x"}},
           "facts": {
               "a": {"value": "100", "dimensions": {"concept": "ifrs-full:Revenue", "entity": "e", "unit": "iso4217:EUR",
                                                    "period": "2025-01-01T00:00:00/2026-01-01T00:00:00"}},
               "b": {"value": "60", "dimensions": {"concept": "ifrs-full:Revenue", "entity": "e", "unit": "iso4217:EUR",
                                                   "period": "2025-01-01T00:00:00/2026-01-01T00:00:00",
                                                   "x:SegmentAxis": "x:Beer"}},
               "c": {"value": "5", "dimensions": {"concept": "ifrs-full:Assets", "entity": "e", "unit": "iso4217:EUR",
                                                  "period": "2026-01-01T00:00:00"}},
               "d": {"value": "2", "dimensions": {"concept": "ifrs-full:BasicEarningsLossPerShare", "entity": "e",
                                                  "unit": "iso4217:EUR/xbrli:shares",
                                                  "period": "2025-01-01T00:00:00/2026-01-01T00:00:00"}},
               "e": {"value": "7", "dimensions": {"concept": "x:Revenue", "entity": "e", "unit": "iso4217:EUR",
                                                  "period": "2025-01-01T00:00:00/2026-01-01T00:00:00"}},
           }}
    facts = esef.facts_from_xbrl_json(doc, FilingLink("u", None))
    assert [(f.concept, f.value, f.start, f.end) for f in facts] == [
        ("ifrs-full:Revenue", 100.0, date(2025, 1, 1), date(2025, 12, 31)),
        ("ifrs-full:Assets", 5.0, None, date(2025, 12, 31)),
    ]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_search_by_isin_finds_the_company_through_its_lei():
    use("esef_tesco")
    found = sources.search("GB00BLGZ9862")
    assert found.matched == "isin"
    esef_hit = next(r for r in found.results if r.source == "esef")
    assert esef_hit.source_id == TESCO_LEI
    assert esef_hit.identifiers == {"lei": TESCO_LEI, "isin": "GB00BLGZ9862"}
    # Its UK company number, from GLEIF, would go to Companies House, which has no key here
    assert ("companies_house", "not_configured") in found.unavailable


def test_search_by_lei_and_by_name():
    use("search_names")
    by_lei = sources.search(TESCO_LEI)
    assert by_lei.matched == "lei"
    assert [(r.source, r.source_id) for r in by_lei.results] == [("esef", TESCO_LEI)]

    by_name = sources.search("Toyota")
    assert by_name.matched is None
    hits = {(r.source, r.name) for r in by_name.results}
    assert ("sec", "TOYOTA MOTOR CORP/") in hits                           # its 20-F
    assert ("edinet", "TOYOTA MOTOR CORPORATION") in hits                   # its annual report
    toyota = next(r for r in by_name.results if r.source == "edinet" and r.source_id == "E02144")
    assert toyota.local_name == "トヨタ自動車株式会社"
    assert toyota.identifiers == {"edinet_code": "E02144", "sec_code": "7203", "ticker": "7203",
                                  "jcn": "1180301018771"}


def test_search_by_ticker_and_by_securities_code():
    use("sec_mcd")
    assert [(r.source, r.source_id) for r in sources.search("MCD").results][:1] == [("sec", "0000063908")]
    use("search_names")
    assert [r.source_id for r in sources.search("7203").results if r.source == "edinet"] == ["E02144"]


def test_a_failing_source_does_not_fail_the_search():
    def transport(req):
        if req.host == "filings.xbrl.org":
            return http.Response(503, b"")
        return replay(FIXTURES / "search_names")(req)
    http.use_transport(transport)
    found = sources.search("Toyota")
    assert ("esef", "failed") in found.unavailable
    assert any(r.source == "edinet" for r in found.results)


def test_ids_are_checked_against_each_sources_shape():
    assert sources.normalise_id("companies_house", "445790") == "00445790"
    assert sources.normalise_id("edinet", "e02144") == "E02144"
    for source, bad in (("sec", "../etc"), ("esef", "TESCO"), ("edinet", "7203"), ("companies_house", "1/2")):
        with pytest.raises(ValueError):
            sources.normalise_id(source, bad)
    with pytest.raises(sources.UnknownSource):
        sources.normalise_id("bloomberg", "x")


def test_a_source_without_its_key_says_so():
    with pytest.raises(http.NotConfigured):
        sources.fetch("companies_house", "00482197")
    with pytest.raises(http.NotConfigured):
        sources.fetch("edinet", "E02144")


# ---------------------------------------------------------------------------
# HTTP: pacing, failures, and keys never shown
# ---------------------------------------------------------------------------
def test_a_key_in_the_query_is_left_out_of_the_recording_name():
    req = http.Request("https://api.edinet-fsa.go.jp/api/v2/documents.json",
                       {"date": "2025-06-18", "type": "2", "Subscription-Key": "s3cret"},
                       secret_params=("Subscription-Key",))
    assert req.key() == "https://api.edinet-fsa.go.jp/api/v2/documents.json?date=2025-06-18&type=2"


def test_a_failed_request_names_the_host_and_status_never_the_url(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.ConnectionError("https://api.edinet-fsa.go.jp/x?Subscription-Key=s3cret refused")
    monkeypatch.setattr(requests, "get", boom)
    http.use_transport(None, paced=False)
    with pytest.raises(http.SourceError) as caught:
        http.get("edinet", "https://api.edinet-fsa.go.jp/x", params={"Subscription-Key": "s3cret"},
                 secret_params=("Subscription-Key",))
    assert "s3cret" not in str(caught.value) and "http" not in str(caught.value)
    assert caught.value.reason == "unreachable"


@pytest.mark.parametrize("status, kind, reason", [
    (404, http.NotFound, "not_found"), (401, http.SourceError, "refused"),
    (429, http.SourceError, "rate_limited"), (500, http.SourceError, "failed")])
def test_statuses_become_reasons(status, kind, reason):
    http.use_transport(lambda req: http.Response(status, b""))
    with pytest.raises(kind) as caught:
        http.get("sec", "https://data.sec.gov/x")
    assert caught.value.reason == reason


def test_calls_to_one_host_are_spaced(monkeypatch):
    slept = []
    monkeypatch.setattr(http.time, "sleep", slept.append)
    monkeypatch.setattr(http, "_last_call", {})
    http.use_transport(lambda req: http.Response(200, b"{}"), paced=True)
    http.get("edinet", "https://api.edinet-fsa.go.jp/a")
    http.get("edinet", "https://api.edinet-fsa.go.jp/b")
    assert len(slept) == 1 and 0.9 < slept[0] <= http.MIN_INTERVAL_S["api.edinet-fsa.go.jp"]


# ---------------------------------------------------------------------------
# The recorder reproduces its own fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["sec_mcd", "esef_tesco", "esef_heineken", "search_names"])
def test_recording_a_fixture_again_gives_the_same_files(case, tmp_path):
    from companies import record
    out = record.record(record.CASES[case], tmp_path, real=replay(FIXTURES / case))
    committed = json.loads((FIXTURES / case / record.INDEX).read_text(encoding="utf-8"))
    again = json.loads((out / record.INDEX).read_text(encoding="utf-8"))
    assert again == committed
    for entry in committed.values():
        assert (out / entry["file"]).read_bytes() == (FIXTURES / case / entry["file"]).read_bytes()


def test_every_summary_field_has_an_answer_shape():
    from api.schemas import CompanyFigures
    assert set(CompanyFigures.model_fields) == set(SUMMARY_FIELDS)
