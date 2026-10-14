"""The results reader (PLAN.md R1): checks, assembly, providers, recordings.

Everything here replays recorded answers (tests/fixtures/research/reader):
no test calls an AI. The API's side is tests/test_research_reader_api.py.
"""
from __future__ import annotations

import copy
import datetime
import json
import random
import re
import socket
from pathlib import Path

import pytest

from ml.evaluation import reader as evaluation
from research.reader import assemble as asm
from research.reader import checks, gate, lines, pages, prompt, providers, record
from research.reader.pages import Info
from research.reader.read import read_filing
from tests import upload_samples as samples

TODAY = datetime.date(2026, 10, 14)
DEVELOPMENT = evaluation.recordings("development")
BY_NAME = {r["document"]["name"]: r for r in DEVELOPMENT}


def assembled(record, a=None, b="same", **kwargs):
    read_a, read_b = evaluation.reads(record)
    return evaluation.run(record, a if a is not None else read_a, read_b if b == "same" else b, **kwargs)


def cell(result, statement, label_start, column):
    table = next(t for t in result["tables"] if t["statement"] == statement)
    row = next(r for r in table["rows"] if r["label"].lower().startswith(label_start.lower()))
    return row["cells"][column]


# ---------------------------------------------------------------- recordings

def test_the_development_set_is_r0s_thirty_filings_each_read_twice_by_full_models():
    assert len(DEVELOPMENT) == 30
    for found in DEVELOPMENT:
        doc = found["document"]
        assert len(doc["sha256"]) == 64 and doc["pages"] == len(doc["dpi"])
        assert set(found["reads"]) == {"A", "B"}
        assert found["prompt"] in ("r0", "r0-first")
    lite = [r["document"]["name"] for r in DEVELOPMENT
            if any("lite" in r["reads"][w]["model"] for w in "AB")]
    assert lite == ["yatharth-hospital_FY27-Q1"]          # the one the card says


def test_r0s_prompt_is_kept_word_for_word_and_r1_only_adds_the_rows_page():
    r0, r1 = prompt.text("r0"), prompt.text("r1")
    assert r0.startswith("This is the quarterly results filing")
    assert '"page": int' not in r0 and '"page": int' in r1
    assert r1.replace('"page": int, ', "").count("\n") == r0.count("\n") + 1
    assert prompt.text("r1", pictures=True).startswith("These are pictures")
    with pytest.raises(ValueError):
        prompt.text("r9")


def test_the_quarter_comes_from_the_file_name_for_a_march_year():
    assert record.quarter_from_name("eternal_FY27-Q1-scanned") == "2026-06-30"
    assert record.quarter_from_name("tata-steel_FY26-Q4-mixed") == "2026-03-31"
    assert record.quarter_from_name("avenue-supermarts_FY27-Q2") == "2026-09-30"
    assert record.quarter_from_name("report") is None


# ------------------------------------------------------------------ assembly

def test_a_recorded_filing_assembles_to_read_as_figures_with_their_places():
    found = assembled(BY_NAME["avenue-supermarts_FY27-Q2"])
    assert (found["company"], found["currency"], found["unit"]) == ("Avenue Supermarts Limited", "INR", "crores")
    assert found["quarter_end"] == "2026-09-30" and found["problems"] == []
    results = next(t for t in found["tables"] if t["statement"] == "results")
    assert results["scope"] == "consolidated" and results["pages"]
    revenue = next(r for r in results["rows"] if r["line"] == "revenue_from_operations")
    assert revenue["line_source"] == "list" and revenue["page"] in results["pages"]
    assert revenue["cells"][0]["value"] == 19644.01 and revenue["cells"][0]["amber"] is None
    assert revenue["cells"][0]["in_sum"]
    assert found["summary"]["figures"] == sum(t["summary"]["figures"] for t in found["tables"])
    # the one figure the two reads give differently is amber, with the other read's value
    ambers = [(r["label"], c) for t in found["tables"] for r in t["rows"] for c in r["cells"] if c["amber"]]
    assert len(ambers) == 1 and ambers[0][1]["amber"] == checks.READS_DIFFER


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_a_wrong_digit_planted_in_a_recorded_answer_is_flagged(name):
    """PLAN.md R1 "Done when": a wrong digit planted in a recorded answer is
    flagged. Three figures a filing, each shown without amber before."""
    found = BY_NAME[name]
    read_a, read_b = evaluation.reads(found)
    clean = evaluation.run(found, read_a, read_b)
    spots = [(t, r, c) for t, table in enumerate(clean["tables"]) for r, row in enumerate(table["rows"])
             for c, item in enumerate(row["cells"]) if item["value"] and not item["amber"]]
    rng = random.Random(name)
    for t, r, c in rng.sample(spots, 3):
        changed = copy.deepcopy(read_a)
        value = changed["tables"][t]["rows"][r]["values"][c]
        changed["tables"][t]["rows"][r]["values"][c] = evaluation._wrong_digit(value, rng)
        after = evaluation.run(found, changed, read_b)
        assert after["tables"][t]["rows"][r]["cells"][c]["amber"] is not None


def test_one_read_alone_misses_planted_digits_that_two_reads_catch():
    plants = missed_alone = missed_two = 0
    for found in DEVELOPMENT[:10]:
        read_a, read_b = evaluation.reads(found)
        count, missed = evaluation.planted(found, read_a, read_b, read_twice=True)
        _, alone = evaluation.planted(found, read_a, None, read_twice=False)
        plants, missed_two, missed_alone = plants + count, missed_two + missed, missed_alone + alone
    assert plants >= 100 and missed_two == 0 and missed_alone > plants / 4


def test_a_misread_that_still_adds_up_is_caught_by_the_second_read():
    """R0 lesson 1: Hindustan Aeronautics' picture read took a 5 for a 4 in
    revenue and moved the total to match. Here read A is given the same
    trick (revenue and total income both off by the same amount)."""
    found = BY_NAME["avenue-supermarts_FY27-Q2"]
    read_a, read_b = evaluation.reads(found)
    changed = copy.deepcopy(read_a)
    results = next(t for t in changed["tables"] if t["statement"] == "results")
    for row in results["rows"]:
        if row["kind"] in ("revenue_from_operations", "total_income"):
            row["values"][0] += 1000
    alone = evaluation.run(found, changed, None, read_twice=False)
    assert cell(alone, "results", "a) revenue", 0)["amber"] is None    # the sums still hold
    both = evaluation.run(found, changed, read_b)
    assert cell(both, "results", "a) revenue", 0)["amber"] == checks.READS_DIFFER


def test_a_table_the_second_read_did_not_find_is_amber_throughout():
    found = BY_NAME["eternal_FY27-Q1"]
    read_a, read_b = evaluation.reads(found)
    only_results = dict(read_b, tables=[t for t in read_b["tables"] if t["statement"] == "results"])
    result = evaluation.run(found, read_a, only_results)
    segments = next(t for t in result["tables"] if t["statement"] == "segments")
    assert segments["summary"]["amber"] == segments["summary"]["figures"] > 0
    assert {"statement": "segments", "column": None, "code": "table_in_one_read", "detail": "A"} in result["problems"]


def test_a_unit_or_currency_the_reads_disagree_on_is_a_problem():
    found = BY_NAME["avenue-supermarts_FY27-Q2"]
    read_a, read_b = evaluation.reads(found)
    result = evaluation.run(found, read_a, dict(read_b, unit="lakhs", currency="USD"))
    codes = {p["code"] for p in result["problems"]}
    assert {"unit_differs", "currency_differs"} <= codes


# ------------------------------------------------------------------- periods

def test_a_wrong_year_in_every_heading_is_a_period_problem_not_hidden_in_the_figures():
    """Kirloskar's read A wrote 2024 for 2026 in every heading (R0)."""
    found = BY_NAME["kirloskar-pneumatic_FY27-Q1"]
    result = assembled(found)
    codes = [p["code"] for p in result["problems"]]
    assert codes.count("unexpected_period") >= 4 and "no_quarter_column" in codes
    assert result["summary"]["period_problems"] >= 5


def test_without_a_given_quarter_the_reads_must_agree_on_it():
    found = BY_NAME["kirloskar-pneumatic_FY27-Q1"]
    read_a, read_b = evaluation.reads(found)
    result = asm.assemble(read_a, read_b, today=TODAY)
    assert result["quarter_end"] is None and any(p["code"] == "quarter_unknown" for p in result["problems"])
    assert any(p["code"] == "reads_differ_period" for p in result["problems"])
    agreed = asm.assemble(*evaluation.reads(BY_NAME["eternal_FY27-Q1"]), today=TODAY)
    assert (agreed["quarter_end"], agreed["quarter_source"]) == ("2026-06-30", "reads")


def test_a_quarter_after_today_is_never_taken_from_the_reads():
    tables = [{"statement": "results", "columns": [{"period_end": "2027-06-30", "months": 3}]}]
    assert checks.filing_quarter(tables, tables, TODAY) is None


def test_the_expected_periods_follow_the_companys_year_end():
    quarter = datetime.date(2026, 9, 30)
    march = checks.expected_periods(quarter, 3)
    assert march[6] == {quarter, datetime.date(2025, 9, 30)}
    assert datetime.date(2026, 3, 31) in march[12]
    december = checks.expected_periods(quarter, 12)
    assert december[9] == {quarter, datetime.date(2025, 9, 30)}
    assert datetime.date(2025, 12, 31) in december[12]


# ---------------------------------------------------------------- statements

def _table(statement, labels_values, columns=2):
    rows = [{"label": label, "kind": "other", "values": values} for label, values in labels_values]
    table = {"statement": statement, "scope": "consolidated", "pages": [1],
             "columns": [{"period_end": "2026-03-31", "months": 0}] * columns, "rows": rows}
    for row, (line, _source) in zip(rows, lines.assign(statement, rows)):
        row["line"] = line
    return table


def test_a_balance_sheet_that_does_not_balance_is_found_in_its_column():
    sheet = _table("balance_sheet", [("Total assets", [100.0, 90.0]),
                                     ("Total equity and liabilities", [100.0, 95.0])])
    assert checks.balance(sheet) == [(1, 0, 1)]


def test_the_cash_check_allows_a_printed_exchange_difference_and_finds_a_real_gap():
    rows = [("Net cash generated from operating activities", [50.0, 40.0]),
            ("Net cash used in investing activities", [-20.0, -10.0]),
            ("Net cash used in financing activities", [-10.0, -5.0]),
            ("Net increase in cash and cash equivalents", [20.0, 25.0]),
            ("Cash and cash equivalents at the beginning of the year", [30.0, 5.0]),
            ("Effect of exchange rate changes", [1.0, 0.0]),
            ("Cash and cash equivalents at the end of the year", [51.0, 36.0])]
    flow = _table("cash_flow", rows)
    assert [r[0] for r in checks.cash(flow)] == [1]          # 5 + 25 is 30, not 36
    assert [line for line in (r["line"] for r in flow["rows"]) if line] == [
        "cash_from_operations", "cash_from_investing", "cash_from_financing", "net_change_in_cash",
        "opening_cash", "closing_cash"]


def test_a_figure_that_differs_from_what_is_stored_is_amber():
    found = BY_NAME["avenue-supermarts_FY27-Q2"]
    read_a, read_b = evaluation.reads(found)
    stored = [{"statement": "results", "scope": "consolidated", "line": "revenue_from_operations",
               "period_end": "2026-06-30", "months": 3, "value": 18794.53},
              {"statement": "results", "scope": "consolidated", "line": "other_income",
               "period_end": "2026-06-30", "months": 3, "value": 99.0}]
    result = asm.assemble(read_a, read_b, today=TODAY, quarter=datetime.date(2026, 9, 30), stored=stored)
    assert cell(result, "results", "a) revenue", 1)["amber"] is None
    other = cell(result, "results", "b) other income", 1)
    assert (other["amber"], other["other"]) == (checks.STORED_DIFFERS, 99.0)


def test_a_figure_on_a_poor_scan_is_amber_unless_a_sum_confirms_it():
    found = BY_NAME["avenue-supermarts_FY27-Q2"]
    read_a, read_b = evaluation.reads(found)
    page = next(t for t in read_a["tables"] if t["statement"] == "results")["pages"][0]
    dpi = [None] * found["document"]["pages"]
    dpi[page - 1] = 80
    result = asm.assemble(read_a, read_b, today=TODAY, info=Info(len(dpi), tuple(dpi)),
                          quarter=datetime.date(2026, 9, 30))
    results = next(t for t in result["tables"] if t["statement"] == "results")
    cells = [c for r in results["rows"] for c in r["cells"] if c["value"]]
    assert all(c["amber"] is not None for c in cells if not c["in_sum"])
    assert any(c["amber"] is None and c["in_sum"] for c in cells)
    assert any(c["amber"] == checks.POOR_SCAN for c in cells)


# --------------------------------------------------------------------- lines

def test_labels_are_matched_from_the_fixed_list_first_and_the_ai_only_for_the_rest():
    rows = [{"label": "1. Revenue from operations", "kind": "other", "values": [1.0]},
            {"label": "Turnover", "kind": "revenue_from_operations", "values": [2.0]},
            {"label": "Income from treasury", "kind": "other_income", "values": [3.0]},
            {"label": "d) Finance costs", "kind": "finance_costs", "values": [4.0]},
            {"label": "Profit for the period", "kind": "net_profit", "values": [5.0]}]
    assert lines.assign("results", rows) == [
        ("revenue_from_operations", "list"),   # the list, from the company's own words
        (None, None),                          # the line is taken: once a table
        ("other_income", "ai"),                # the list has no wording for it; the AI's kind is allowed
        ("finance_costs", "list"),
        ("net_profit", "ai")]
    assert lines.assign("results", [{"label": "Odd", "kind": "finance_costs", "values": [1.0]}]) == [(None, None)]
    assert lines.clean("(ii) Less: Exceptional items") == "exceptional items"


# -------------------------------------------------------------------- answers

def test_an_answer_wrapped_in_a_list_is_unwrapped_and_anything_else_refused():
    found = BY_NAME["eternal_FY27-Q1"]["reads"]["A"]["read"]
    assert asm.unwrap([found])["tables"]
    for bad in (None, "text", [found, found], {"tables": "x"}):
        with pytest.raises(asm.Unreadable):
            asm.unwrap(bad)


def test_an_answer_is_cut_to_sane_sizes_and_cleaned():
    answer = {"company": "A\x00B" + "x" * 500, "unit": "CRORES", "currency": "inr",
              "tables": [{"statement": "nonsense", "scope": "both", "pages": [1, "2", True],
                          "columns": [{"period_end": "2026-06-30", "months": 3}] * 40,
                          "rows": [{"label": "r", "values": [1, "(2)", "x", float("nan")]}] * 900}]}
    read = asm.unwrap(answer)
    table = read["tables"][0]
    assert read["company"].startswith("AB") and len(read["company"]) == asm.MAX_LABEL
    assert (read["unit"], read["currency"]) == ("crores", "INR")
    assert (table["statement"], table["scope"], table["pages"]) == ("results", "consolidated", [1])
    assert len(table["columns"]) == asm.MAX_COLUMNS and len(table["rows"]) == asm.MAX_ROWS
    assert table["rows"][0]["values"][:4] == [1.0, -2.0, None, None]


# ---------------------------------------------------------------------- pages

def test_the_page_tool_counts_draws_and_judges_a_pdf_in_its_own_process(tmp_path):
    path = tmp_path / "filing.pdf"
    path.write_bytes(samples.pdf())
    found = pages.info(path)
    assert (found.pages, found.dpi) == (1, (None,)) and not found.poor(1)
    drawn = pages.render(path, [1])
    assert len(drawn) == 1 and drawn[0].startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(pages.PagesError):
        pages.render(path, [2])
    (tmp_path / "junk.pdf").write_bytes(b"not a pdf")
    with pytest.raises(pages.PagesError):
        pages.info(tmp_path / "junk.pdf")


def test_the_png_writer_writes_what_it_is_given():
    import zlib

    data = pages.png(bytes([0, 255, 10, 20, 30, 40]), 2, 1, 3)
    assert data[12:16] == b"IHDR" and zlib.decompress(data[data.index(b"IDAT") + 4:-16]) == bytes(
        [0, 0, 255, 10, 20, 30, 40])


# ------------------------------------------------------------------ providers

class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.content = json.dumps(body).encode()

    def json(self):
        return self._body


class FakeSession:
    """Answers per model; records every URL and header it was given."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def post(self, url, headers=None, json=None, timeout=None, **_):  # noqa: A002
        self.calls.append((url, headers))
        model = url.rsplit("/", 1)[1].split(":")[0]
        return self.answers[model]

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, headers))
        return self.answers["list"]

    def delete(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))


def ok(read):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(read)}]}}],
                              "usageMetadata": {"totalTokenCount": 9}})


DAY = FakeResponse(429, {"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"quotaId": "RequestsPerDay"}]}})
BUSY = FakeResponse(503, {"error": {"status": "UNAVAILABLE", "message": "high demand"}})


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(providers.Gemini, "_spent", {})
    for name in ("GEMINI_MODELS_A", "GEMINI_MODELS_B", "READER_HIGH_RESOLUTION", "AI_FREE_TIER"):
        monkeypatch.delenv(name, raising=False)

    def make(answers):
        session = FakeSession(answers)
        return providers.Gemini("SECRET-KEY-123", session=session), session
    return make


def test_a_model_whose_day_is_spent_or_busy_is_passed_over_for_the_next(gemini):
    read = {"tables": []}
    provider, session = gemini({"gemini-3.8-flash": DAY, "gemini-3.7-flash": BUSY, "gemini-3.6-flash": ok(read)})
    answer = provider.read("A", providers.Document("x", pdf=b"%PDF"))
    assert (answer.model, answer.read) == ("gemini-3.6-flash", read)
    assert all("SECRET" not in url for url, _ in session.calls)
    assert all(headers["x-goog-api-key"] == "SECRET-KEY-123" for _, headers in session.calls)
    # the spent model is skipped next time without a call
    again = provider.read("A", providers.Document("x", pdf=b"%PDF"))
    assert again.model == "gemini-3.7-flash" or again.model == "gemini-3.6-flash"
    assert sum("3.8" in url for url, _ in session.calls) == 1


def test_read_b_never_uses_read_as_model_and_says_try_tomorrow_when_all_are_spent(gemini):
    provider, session = gemini({m: DAY for m in providers.MODELS_B})
    with pytest.raises(providers.ProviderError) as refused:
        provider.read("B", providers.Document("x", pictures=[b"png"]), exclude=frozenset({"gemini-3.8-flash"}))
    assert refused.value.code == providers.SPENT
    assert not any("3.8-flash" in url for url, _ in session.calls)


def test_busy_models_say_try_again_soon_and_a_refused_key_stops_at_once(gemini):
    provider, _ = gemini({m: BUSY for m in providers.MODELS_A})
    with pytest.raises(providers.ProviderError) as busy:
        provider.read("A", providers.Document("x", pdf=b"%PDF"))
    assert busy.value.code == providers.BUSY
    refused = FakeResponse(400, {"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}})
    provider, session = gemini({m: refused for m in providers.MODELS_A})
    with pytest.raises(providers.ProviderError) as stopped:
        provider.read("A", providers.Document("x", pdf=b"%PDF"))
    assert stopped.value.code == providers.KEY_REFUSED and len(session.calls) == 1
    assert "SECRET" not in str(stopped.value)


def test_the_status_lists_the_models_without_spending_or_showing_the_key(gemini):
    listed = FakeResponse(200, {"models": [{"name": f"models/{m}"} for m in providers.MODELS_B]})
    provider, session = gemini({"list": listed})
    found = provider.status()
    assert found["status"] == "ok" and found["key"] == "accepted" and found["free_tier"] is True
    assert "SECRET" not in json.dumps(found) and session.calls[0][0].endswith("/models")
    provider, _ = gemini({"list": FakeResponse(400, {"error": {"status": "INVALID_ARGUMENT"}})})
    assert provider.status()["key"] == "refused"


def test_replay_answers_by_the_documents_hash_and_nothing_else():
    replay = providers.Replay()
    found = BY_NAME["eternal_FY27-Q1"]
    answer = replay.read("B", providers.Document(found["document"]["sha256"]))
    assert answer.model == found["reads"]["B"]["model"] and answer.prompt_version == "r0"
    with pytest.raises(providers.ProviderError) as unknown:
        replay.read("A", providers.Document("0" * 64))
    assert unknown.value.code == providers.NOT_RECORDED
    fallback = providers.Replay(default="eternal_FY27-Q1")
    assert fallback.read("A", providers.Document("0" * 64)).model == found["reads"]["A"]["model"]


def test_the_switch_never_replays_in_production_and_is_off_without_a_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "replay")
    monkeypatch.setenv("FSE_ENV", "production")
    assert providers.get_provider() is None
    monkeypatch.setenv("FSE_ENV", "staging")
    assert isinstance(providers.get_provider(), providers.Replay)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert providers.get_provider() is None
    monkeypatch.setattr(providers, "_status", [0.0, None, None])
    assert providers.status() == {"status": "off", "provider": "gemini", "key": "missing"}
    monkeypatch.setenv("AI_FREE_TIER", "0")
    assert providers.free_tier() is False


def test_reading_a_filing_from_recordings_makes_no_network_call(tmp_path, monkeypatch):
    """CI makes no AI call: the whole road on replay, with sockets refused."""
    def refuse(*args, **kwargs):
        raise AssertionError("a network connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    path = tmp_path / "filing.pdf"
    path.write_bytes(samples.pdf())
    found = read_filing(path, providers.Replay(default="eternal_FY27-Q1"), today=TODAY,
                        quarter=datetime.date(2026, 6, 30))
    assert found["summary"]["figures"] > 100 and found["reads"]["B"]["model"]
    assert found["document"]["pages"] == 1


# ------------------------------------------------------------------- the gate

def test_production_reads_only_once_the_cards_headline_set_meets_the_bar(tmp_path):
    card = json.loads(gate.CARD.read_text())
    assert gate.measured()["set"] == "held_out"
    assert gate.open_to_people("staging") is True
    path = tmp_path / "reader.json"
    card["evaluation"]["sets"]["held_out"]["meets_bar"] = None
    path.write_text(json.dumps(card))
    assert gate.open_to_people("production", path) is False
    card["evaluation"]["sets"]["held_out"]["meets_bar"] = True
    path.write_text(json.dumps(card))
    assert gate.open_to_people("production", path) is True
    assert gate.measured(tmp_path / "missing.json")["meets_bar"] is None


def test_the_card_measures_the_development_set_and_waits_for_held_out_filings():
    card = json.loads(gate.CARD.read_text())
    sets = card["evaluation"]["sets"]
    development = sets["development"]
    assert development["overall"]["cases"] == 30 and development["meets_bar"] is True
    assert development["overall"]["model"]["planted_missed_share"] == 0.0
    assert development["overall"]["model"]["amber_share"] < 0.02
    assert development["overall"]["baseline"]["planted_missed_share"] > 0.5
    assert sets["held_out"]["meets_bar"] is None or isinstance(sets["held_out"]["meets_bar"], bool)


def test_the_truth_file_holds_bse_figures_for_ten_filings_and_none_is_missed():
    with_truth = [r for r in DEVELOPMENT if r.get("truth")]
    assert len(with_truth) == 10
    totals = [0, 0, 0]
    for found in with_truth:
        result = assembled(found)
        counted = evaluation.truth(found, result)
        totals = [x + y for x, y in zip(totals, counted)]
    assert totals[0] >= 230 and totals[2] == 0 and totals[1] >= 0.95 * totals[0]


def test_a_wrong_figure_shown_without_amber_counts_against_the_truth_file():
    """Both reads wrong the same way, outside every sum (earnings per share):
    nothing can flag it, and the truth file says so."""
    tried = 0
    for found in (r for r in DEVELOPMENT if r.get("truth")):
        if not any(f["label"].startswith("Basic") or "EPS" in f["label"] for f in found["truth"]["figures"]):
            continue
        read_a, read_b = evaluation.reads(found)
        before = evaluation.truth(found, evaluation.run(found, read_a, read_b))
        for read in (read_a, read_b):
            table = next(t for t in read["tables"] if t["statement"] == "results")
            for row in table["rows"]:
                if re.search("basic|diluted", row["label"].lower()) and row["values"] and row["values"][0] is not None:
                    row["values"][0] += 1.0
        after = evaluation.truth(found, evaluation.run(found, read_a, read_b))
        column_amber = before[2] == after[2]       # an amber cell in the column already flags it
        assert after[2] >= before[2] + 1 or column_amber
        tried += not column_amber
    assert tried >= 3


def test_the_recordings_are_what_record_from_r0_writes(tmp_path):
    """The fixtures can be rebuilt only where R0's PDFs are (the Mac): here
    each recording is checked against itself as the recorder writes it."""
    for found in DEVELOPMENT[:3]:
        target = tmp_path / "x.json"
        record.write(target, found)
        name = found["document"]["name"]
        assert target.read_text() == (providers.RECORDINGS / "development" / f"{name}.json").read_text()


def test_the_page_tool_child_is_given_none_of_the_servers_secrets(monkeypatch, tmp_path):
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(kwargs["env"])
        raise OSError("stop")

    monkeypatch.setenv("GEMINI_API_KEY", "SECRET")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setattr(pages.subprocess, "run", fake_run)
    with pytest.raises(pages.PagesError):
        pages.info(tmp_path / "x.pdf")
    assert set(seen) == {"PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING"}


def test_recordings_are_public_filings_with_no_key_in_them():
    for path in Path(providers.RECORDINGS).rglob("*.json"):
        text = path.read_text()
        assert "AIza" not in text and "x-goog-api-key" not in text


# ------------------------------------------------- the child, the recorder

def test_the_page_tools_child_answers_info_and_draws_pages(tmp_path, capsys):
    path = tmp_path / "filing.pdf"
    path.write_bytes(samples.pdf())
    assert pages.main(["info", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"pages": 1, "dpi": [None]}
    assert pages.main(["render", str(path), str(tmp_path), "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {"files": ["page-1.png"]}
    assert (tmp_path / "page-1.png").read_bytes().startswith(b"\x89PNG")
    assert pages.main(["render", str(path), str(tmp_path), "3"]) == 4
    assert pages.main(["nonsense"]) == 2
    (tmp_path / "junk.pdf").write_bytes(b"junk")
    assert pages.main(["info", str(tmp_path / "junk.pdf")]) == 3


def test_a_scanned_page_reports_its_resolution(tmp_path):
    """A page that is one picture (a scan) at 72 pixels an inch is a poor scan."""
    import zlib

    pixels = zlib.compress(b"\xff" * (612 * 792))
    image = (b"/Type /XObject /Subtype /Image /Width 612 /Height 792 /ColorSpace /DeviceGray "
             b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d" % len(pixels))
    draw = b"q 612 0 0 792 0 0 cm /Im1 Do Q"
    body = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /XObject << /Im1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(draw) + draw + b"\nendstream",
            b"<< " + image + b" >>\nstream\n" + pixels + b"\nendstream"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, item in enumerate(body, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + item + b"\nendobj\n"
    table = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(body) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(body) + 1, table)
    path = tmp_path / "scan.pdf"
    path.write_bytes(bytes(out))
    found = pages.info(path)
    assert found.dpi == (72,) and found.poor(1) and not found.poor(2)


BSE_TABLE = """<table>
<tr><td><b>Type</b></td><td>Un-Audited</td></tr>
<tr><td><b>Date Begin</b></td><td>01-Jul-26</td></tr>
<tr><td><b>Date End</b></td><td>30-Sep-26</td></tr>
<tr><td><b>Description</b></td><td><b>Amount (Rs. million)</b></td></tr>
<tr><td>Net Sales</td><td>1,96,440.10</td></tr>
<tr><td>Other Income</td><td>135.30</td></tr>
<tr><td>Expenditure</td><td>-1,86,327.00</td></tr>
<tr><td>Net Profit Margin</td><td>3.78</td></tr>
</table>"""


def test_bses_table_becomes_a_truth_file_only_in_millions(tmp_path):
    path = tmp_path / "x.xls"
    path.write_text(BSE_TABLE)
    truth = record.bse_figures(path)
    assert truth["unit"] == "millions"
    assert [(f["label"], f["period_end"], f["months"], f["value"]) for f in truth["figures"]] == [
        ("Net Sales", "2026-09-30", 3, 196440.1), ("Other Income", "2026-09-30", 3, 135.3),
        ("Expenditure", "2026-09-30", 3, -186327.0)]
    path.write_text(BSE_TABLE.replace("Rs. million", "Rs. lakhs"))
    assert record.bse_figures(path) is None


def test_bses_grouping_of_lines_is_traced_to_read_as_own_lines():
    found = copy.deepcopy(BY_NAME["avenue-supermarts_FY27-Q2"])
    for figure in found["truth"]["figures"]:
        figure.pop("grouped", None)
        figure.pop("grouped_by", None)
    record.group_truth(found)
    tax = next(f for f in found["truth"]["figures"] if f["label"] == "Tax")
    assert [g[:2] for g in tax["grouped"]] == [["3 Profit before tax", 1], ["5 Net profit after tax", -1]]
    assert not any("grouped" in f for f in found["truth"]["figures"] if f["label"] == "Net Sales")


def _saved_r0(folder: Path, stem: str, model: str, which: str, read: dict) -> None:
    (folder / f"{stem}.{which}.{model}.json").write_text(json.dumps(
        {"ok": True, "model": model, "read": read, "usage": {"totalTokenCount": 5}, "seconds": 3}))


def test_r0s_saved_reads_become_recordings_with_the_pdfs_hash(tmp_path):
    samples_dir = tmp_path / "samples"
    out = samples_dir / "r0" / "out"
    out.mkdir(parents=True)
    (samples_dir / "xbrl").mkdir()
    pdf = samples_dir / "acme_FY27-Q2-scanned.pdf"
    pdf.write_bytes(samples.pdf())
    source = BY_NAME["avenue-supermarts_FY27-Q2"]["reads"]
    _saved_r0(out, pdf.stem, "gemini-3.8-flash", "SA", source["A"]["read"])
    _saved_r0(out, pdf.stem, "gemini-3-flash-preview", "SB", source["B"]["read"])
    _saved_r0(out, pdf.stem, "gemini-3.5-flash-lite", "SB", {"tables": []})     # a lite read is passed over
    (samples_dir / "xbrl" / "acme_FY27-Q2.xls").write_text(BSE_TABLE)
    first = samples_dir / "other_FY27-Q1.pdf"
    first.write_bytes(samples.pdf(padding=10))
    table = dict(source["A"]["read"]["tables"][0], company="Other", currency="INR", unit="lakhs", page=3)
    (out / f"{first.stem}.A.gemini-3.7-flash.json").write_text(json.dumps({"ok": True, "table": table, "usage": {}}))
    (out / f"{first.stem}.B.gemini-3.7-flash.json").write_text(json.dumps({"ok": True, "table": table, "usage": {}}))
    (samples_dir / "lonely_FY27-Q1.pdf").write_bytes(samples.pdf(padding=20))
    written = record.from_r0(samples_dir, tmp_path / "recordings")
    assert written == ["acme_FY27-Q2", "other_FY27-Q1"]
    acme = json.loads((tmp_path / "recordings" / "acme_FY27-Q2.json").read_text())
    assert acme["document"]["sha256"] == providers.sha256(pdf.read_bytes())
    assert acme["document"]["quarter_end"] == "2026-09-30" and acme["prompt"] == "r0"
    assert acme["reads"]["B"]["model"] == "gemini-3-flash-preview" and len(acme["truth"]["figures"]) == 3
    other = json.loads((tmp_path / "recordings" / "other_FY27-Q1.json").read_text())
    assert other["prompt"] == "r0-first" and other["reads"]["A"]["read"]["unit"] == "lakhs"
    assert asm.unwrap(other["reads"]["A"]["read"])["tables"][0]["pages"] == [table["pages"][0]]


class Scripted:
    """A provider that answers from a list and counts what it was asked."""

    name = "gemini"
    free_tier = True
    needs_pictures = True

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def read(self, which, document, *, exclude=frozenset()):
        self.asked.append((which, len(document.pictures), tuple(sorted(exclude))))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return providers.Answer(answer, f"model-{which}", prompt.PROMPT_VERSION, {"totalTokenCount": 3}, 1.0)


def test_live_recording_saves_each_answer_and_carries_on_after_a_spent_day(tmp_path, monkeypatch):
    read = copy.deepcopy(BY_NAME["eternal_FY27-Q1"]["reads"]["A"]["read"])
    for table in read["tables"]:
        table["pages"] = [1]
    pdf = tmp_path / "acme_FY27-Q2.pdf"
    pdf.write_bytes(samples.pdf())
    target = tmp_path / "held_out" / "acme_FY27-Q2.json"
    spent = Scripted([read, providers.ProviderError(providers.SPENT)])
    with pytest.raises(providers.ProviderError):
        record.read_one(spent, pdf, target, "held_out", None)
    saved = json.loads(target.read_text())
    assert set(saved["reads"]) == {"A"} and saved["document"]["quarter_end"] == "2026-09-30"
    tomorrow = Scripted([read])
    assert record.read_one(tomorrow, pdf, target, "held_out", None) == "A by model-A, B by model-B"
    assert tomorrow.asked == [("B", 1, ("model-A",))]        # A is never asked twice; B sees a picture
    assert record.read_one(Scripted([]), pdf, target, "held_out", None).startswith("A by")
    saved["prompt"] = "r0"
    target.write_text(json.dumps(saved))
    assert "left as it is" in record.read_one(Scripted([]), pdf, target, "held_out", None)


def test_the_recorder_reads_its_key_from_a_file_or_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nGEMINI_API_KEY= abc \n")
    assert record._key(env) == "abc"
    monkeypatch.setenv("GEMINI_API_KEY", "xyz")
    assert record._key(None) == "xyz"
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit):
        record._key(None)


def test_the_recorder_command_stops_for_the_day_when_every_model_is_spent(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "acme_FY27-Q2.pdf"
    pdf.write_bytes(samples.pdf())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(record, "RECORDINGS", tmp_path / "recordings")
    monkeypatch.setattr(record, "Gemini", lambda key: Scripted([providers.ProviderError(providers.SPENT)]))
    assert record.main(["read", "--set", "held_out", str(pdf)]) == 0
    assert "run the same command again tomorrow" in capsys.readouterr().out


def test_a_big_pdf_goes_through_googles_file_store_and_is_deleted_after(gemini, monkeypatch):
    monkeypatch.setattr(providers, "INLINE_LIMIT", 3)
    deleted = []
    read = {"tables": []}
    started = FakeResponse(200, {}, {"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/x"})
    stored = FakeResponse(200, {"file": {"name": "files/abc", "uri": "https://g/files/abc", "state": "ACTIVE"}})
    provider, session = gemini({"gemini-3.8-flash": ok(read)})
    answers = iter([started, stored])
    real_post = session.post

    def post(url, **kwargs):
        if "upload" in url:
            return next(answers)
        return real_post(url, **kwargs)

    session.post = post
    session.delete = lambda url, headers=None, timeout=None: deleted.append(url)
    assert provider.read("A", providers.Document("x", pdf=b"%PDF-big")).model == "gemini-3.8-flash"
    assert deleted == [f"{providers.BASE}/files/abc"]


def test_an_upload_address_that_is_not_googles_is_never_followed(gemini, monkeypatch):
    monkeypatch.setattr(providers, "INLINE_LIMIT", 3)
    provider, session = gemini({})
    session.post = lambda url, **kwargs: FakeResponse(200, {}, {"x-goog-upload-url": "https://evil.example/x"})
    with pytest.raises(providers.ProviderError) as refused:
        provider.read("A", providers.Document("x", pdf=b"%PDF-big"))
    assert refused.value.code == providers.NETWORK


def test_a_fenced_or_blocked_answer_is_handled(gemini):
    assert providers.parse_json('```json\n{"tables": []}\n```') == {"tables": []}
    with pytest.raises(providers.ProviderError):
        providers.parse_json("not json")
    provider, _ = gemini({m: FakeResponse(200, {"candidates": []}) for m in providers.MODELS_A})
    with pytest.raises(providers.ProviderError) as blocked:
        provider.read("A", providers.Document("x", pdf=b"%PDF"))
    assert blocked.value.code == providers.BLOCKED


def test_the_drill_fails_only_on_a_reader_whose_key_does_not_work():
    from ops.scheduled import live_problems

    base = {"live_distinct": 1, "live_moves": 1, "memory_peak_mb": 100, "memory_limit_mb": 512}
    assert live_problems({**base, "reader": "ok"}) == []
    assert live_problems({**base, "reader": "off"}) == []
    problems = live_problems({**base, "reader": "error", "reader_key": "refused"})
    assert len(problems) == 1 and "key refused" in problems[0]


def test_a_page_read_a_guessed_past_the_end_is_never_drawn(tmp_path):
    read = copy.deepcopy(BY_NAME["eternal_FY27-Q1"]["reads"]["A"]["read"])
    read["tables"][0]["pages"], read["tables"][1]["pages"] = [1], [15]
    path = tmp_path / "filing.pdf"
    path.write_bytes(samples.pdf())
    provider = Scripted([read, copy.deepcopy(read)])
    found = read_filing(path, provider, today=TODAY, quarter=datetime.date(2026, 6, 30))
    assert provider.asked[1] == ("B", 1, ("model-A",)) and found["reads"]["B"]["model"] == "model-B"


def test_read_bs_table_of_the_same_scope_is_the_one_compared():
    found = BY_NAME["eternal_FY27-Q1"]
    read_a, read_b = evaluation.reads(found)
    standalone = copy.deepcopy(read_b["tables"][0])
    standalone["scope"] = "standalone"
    for row in standalone["rows"]:
        row["values"] = [None if v is None else v + 7 for v in row["values"]]
    swapped = dict(read_b, tables=[standalone, *read_b["tables"]])
    result = evaluation.run(found, read_a, swapped)
    assert result["tables"][0]["summary"]["amber"] == 0


def test_a_page_with_a_huge_declared_size_is_drawn_within_the_pixel_cap(tmp_path):
    body = samples.pdf().replace(b"/MediaBox [0 0 612 792]", b"/MediaBox [0 0 14400 14400]")
    path = tmp_path / "huge.pdf"
    path.write_bytes(body)
    import struct

    drawn = pages.render(path, [1])[0]
    width, height = struct.unpack(">II", drawn[16:24])
    assert width * height <= pages.MAX_PIXELS * 1.01 and width > 1000


def test_calling_the_childs_work_in_process_sets_no_limit_on_the_caller(tmp_path, capsys):
    import resource

    before = resource.getrlimit(resource.RLIMIT_CPU)
    path = tmp_path / "filing.pdf"
    path.write_bytes(samples.pdf())
    pages.main(["info", str(path)])
    capsys.readouterr()
    assert resource.getrlimit(resource.RLIMIT_CPU) == before
