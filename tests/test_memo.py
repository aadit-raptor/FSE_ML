"""Investment memos (PLAN.md 6.3), without a database.

The task's two proofs:

- **every figure in a memo is its run's**: the memo of a version is built from
  ``ml.memo.run_version``, and every figure in it (tables, sentences, notes)
  is compared with what the screens' own endpoints answer for the same
  inputs, asked here through the API, with the path read by this test's own
  map, not the memo's; the Word and PDF files carry each figure as shown;
- **no company name or identifier reaches the free provider**: a deal full of
  names and numbers is written through Gemini's own code, its HTTP answered
  here, and nothing of the deal, no digit and no figure is in what was sent;
  the app fills every figure in afterwards.

Then the refusals: a planted figure, label, digit, number word or rating word
stops the memo; the AI's answers that break a rule go back to the template.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api.main import app
from ml import memo, memo_writer
from research import report_files
from research.reader import providers
from research.reader.providers import Answer, ProviderError
from research.report import DraftRefused, Figure, Label, Paragraph, Text, Typed, segments

client = TestClient(app)

# A deal nothing like the defaults, and a second one with facilities, tax rules, leases and fiscal years
PLAIN = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5, "debt_pct": 55.0,
         "currency": "EUR", "unit": "thousands"}
LISTED = {"ebitda": 80.0, "entry_mult": 8.0, "exit_mult": 8.5, "hold": 5, "currency": "GBP", "unit": "millions",
          "first_fiscal_year": 2027, "fiscal_year_end_month": 3, "country": "GB", "industry": "machinery",
          "tax_loss_carryforward": True, "accounting_standard": "ifrs", "lease_cost": 4.0, "lease_liability": 30.0,
          "tranches": [{"name": "Kestrel TLB", "kind": "institutional_term_loan", "amount": 300.0, "floating": True,
                        "reference_level": 4.5, "margin": 4.0},
                       {"name": "Kestrel PIK", "kind": "pik_notes", "amount": 60.0, "fixed_rate": 11.0}]}
SETTINGS = {"tx_fee_pct": 2.5, "mc_n": 20000, "mc_hurdle": 18.0}
# Three years of results and an exit, in the deal's currency
ACTUALS = {"currency": "GBP", "unit": "millions",
           "years": [{"revenue": 260.0, "ebitda": 84.0, "net_income": 20.0, "fcf": 30.0, "total_debt": 330.0},
                     {"revenue": 280.0, "ebitda": 90.0, "net_income": 25.0, "fcf": 34.0, "total_debt": 300.0},
                     {"revenue": 300.0, "ebitda": 97.0, "net_income": 30.0, "fcf": 40.0, "total_debt": 270.0}],
           "exit": {"exit_ev": 880.0, "net_debt_at_exit": 260.0, "sponsor_equity_entry": 330.0}}

CASES = {"defaults": ({}, {}, None), "plain": (PLAIN, SETTINGS, None), "listed": (LISTED, {}, ACTUALS)}


def post(path: str, body: dict) -> dict:
    answer = client.post(path, json=body)
    assert answer.status_code == 200, answer.text
    return answer.json()


def independent(inputs: dict, settings: dict, actuals) -> dict:
    """What the screens are answered for this deal, asked through the API with the Monte Carlo screen's inputs
    built here (MonteCarloProvider.tsx: the Settings' mc_* values, the deal's EBITDA, multiple and holding period;
    seed 42)."""
    deal = post("/api/deal/run", {"inputs": inputs, "settings": settings})
    full = deal_inputs(inputs)
    mc = {"n": settings.get("mc_n", 50000), "hurdle": settings.get("mc_hurdle", 20.0),
          "ebitda": full["ebitda"], "entry_mult": full["entry_mult"], "hold": full["hold"]}
    sim = post("/api/montecarlo/run", {"mc": mc, "deal": inputs, "settings": settings, "seed": 42})
    scenarios = post("/api/montecarlo/scenarios", {"mc": mc, "deal": inputs, "settings": settings, "seed": 42})
    plan = post("/api/backtesting/plan-vs-actual", {"plan": inputs, "settings": settings, "actuals": actuals}) \
        if actuals else None
    return {"inputs": full, "deal": deal, "sim": sim, "scenarios": scenarios, "plan": plan}


def deal_inputs(inputs: dict) -> dict:
    from api.schemas import DealInputsIn

    return DealInputsIn(**inputs).model_dump()


def expected(answers: dict, path: str):
    """A memo figure's value, read from the endpoints' answers by this test's own map of the memo's paths."""
    parts = path.split(".")
    head, rest = parts[0], parts[1:]

    def walk(node, keys):
        for key in keys:
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    deal, sim, plan = answers["deal"], answers["sim"], answers["plan"]
    if head == "inputs":
        return answers["inputs"][rest[0]]
    if head == "returns":
        return deal["returns"][rest[0]]
    if path == "debt.at_close":
        return deal["debt_schedule"]["total_beginning_debt"][0]
    if path == "debt.at_exit":
        return deal["debt_schedule"]["total_ending_debt"][-1]
    if head == "bridge":
        return deal["equity_bridge"][rest[0]]
    if head == "credit":
        return deal["credit"][rest[0]]
    if head == "warnings":
        return walk(deal["risk_warnings"], rest)
    if path == "simulation.n":
        return sim["n"]
    if path == "simulation.seed":
        return 42
    if head == "simulation":
        return sim["summary"][rest[0]]
    if head == "explanations":
        return walk(sim["explanations"], rest)
    if head == "scenarios":
        return answers["scenarios"]["scenarios"][rest[0]][rest[1]]
    if head == "plan_actual":
        return plan[rest[0]] if len(rest) == 1 else walk(plan, rest)
    raise AssertionError("a memo figure from nowhere: %s" % path)


def drafted(inputs, settings, actuals, *, name="Projekt Kestrel GmbH", label="Board pack", language="en",
            grouping="thousands", provider=None):
    run = memo.run_version(inputs, settings, actuals)
    s = memo.Sources(run, memo.Meta(name, 3, label, "2026-10-20"), language=language, grouping=grouping)
    written = memo_writer.write(s, provider)
    s = memo.with_sentences(s, {k: v["text"] for k, v in written["sections"].items()})
    return s, memo.build(s), written


def all_figures(draft) -> list[Figure]:
    return [seg for seg, _, _ in segments(draft) if isinstance(seg, Figure)]


# ---------------------------------------------------------------------------------------------- every figure
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_figure_in_a_memo_is_the_model_runs(case):
    inputs, settings, actuals = CASES[case]
    s, draft, _ = drafted(inputs, settings, actuals)
    answers = independent(inputs, settings, actuals)
    figures = all_figures(draft)
    assert len(figures) >= 50, len(figures)
    for f in figures:
        value = expected(answers, f.row)
        assert f.source == "run" and f.value == pytest.approx(value, rel=1e-12, abs=1e-12), f.row
        assert f.text == memo.show(value, f.unit, s.grouping), f.row
    paths = {f.row for f in figures}
    # the full metric set, the drivers and the scenarios are all there
    for path in ("returns.irr", "returns.moic", "simulation.p_loss", "simulation.p5_irr", "credit.coverage",
                 "credit.default_pct", "explanations.base_irr", "scenarios.recession.median_irr",
                 "scenarios.bull.p_above_hurdle"):
        assert path in paths, path
    if actuals:
        assert {"plan_actual.actual.irr", "plan_actual.plan.irr", "plan_actual.attribution.exit_multiple"} <= paths


def test_the_memo_shows_the_sentences_figures_and_names_from_the_run():
    s, draft, written = drafted(LISTED, {}, ACTUALS)
    sentences = [b for b in draft.blocks if isinstance(b, Paragraph) and b.kind == "sentence"]
    assert {name for name in written["sections"]} == set(memo_writer.SECTIONS)
    assert len(sentences) == 5 and all(any(isinstance(x, Figure) for x in b.parts) for b in sentences)
    labels = {seg.source: seg.text for seg, _, _ in segments(draft) if isinstance(seg, Label)}
    assert labels["deal_name"] == "Projekt Kestrel GmbH" and labels["money"] == "GBP millions"
    # the deal's years have fiscal names: a March year end from 2027
    assert all(re.fullmatch(r"FY\d{4}/\d{2}", text) for src, text in labels.items() if src.startswith("warning_year"))


def test_the_word_and_pdf_files_carry_every_figure_as_shown():
    s, draft, _ = drafted(LISTED, {}, ACTUALS)
    shown = {f.text for f in all_figures(draft) if f.text != "–"}
    word = report_files.word(draft, "Kestrel, investment memo", lang="de-DE")
    with zipfile.ZipFile(io.BytesIO(word)) as z:
        document, styles = z.read("word/document.xml").decode(), z.read("word/styles.xml").decode()
    pdf = report_files.pdf(draft, "Kestrel, investment memo")
    text = " ".join(" ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages).split())
    for figure in shown:
        assert figure in document and figure in text, figure
    assert 'w:lang w:val="de-DE"' in styles


def test_a_run_is_kept_small_and_as_plain_json():
    run = memo.run_version(LISTED, {}, ACTUALS)
    assert len(json.dumps(run)) < 8000
    assert json.loads(json.dumps(run)) == run


def test_a_saved_actual_result_that_does_not_fit_the_plan_is_said_not_compared():
    other = {**ACTUALS, "currency": "EUR"}
    s, draft, _ = drafted(LISTED, {}, other)
    assert s.run["plan_actual"] == {"refused": True}
    words = " ".join(seg.text for seg, _, _ in segments(draft) if isinstance(seg, Text))
    assert "could not be set against" in words


# ---------------------------------------------------------------------------------------------- the check
def _swap(draft, old, new):
    from research.report import Draft, Heading, Table

    def cell(parts):
        return tuple(new if x is old else x for x in parts)

    blocks = []
    for b in draft.blocks:
        if isinstance(b, (Heading, Paragraph)):
            blocks.append(type(b)(cell(b.parts), *(b.level,) if isinstance(b, Heading) else (b.kind,)))
        elif isinstance(b, Table):
            blocks.append(Table(tuple(cell(c) for c in b.header), tuple(tuple(cell(c) for c in r) for r in b.rows)))
        else:
            blocks.append(b)
    return Draft(tuple(blocks), draft.grouping)


@pytest.fixture(scope="module")
def built():
    s, draft, _ = drafted(PLAIN, SETTINGS, None)
    return s, draft


def _first(draft, kind, where=lambda x: True):
    return next(seg for seg, _, _ in segments(draft) if isinstance(seg, kind) and where(seg))


@pytest.mark.parametrize("plant, code", [
    (lambda f: Figure(f.source, f.row, f.period, f.unit, f.value * 1.01, f.text), "figure_differs"),
    (lambda f: Figure(f.source, f.row, f.period, f.unit, f.value, f.text + "0"), "figure_differs"),
    (lambda f: Figure(f.source, "returns.irr_typed", f.period, f.unit, f.value, f.text), "figure_differs"),
    (lambda f: Figure(f.source, f.row, f.period, "money", f.value, f.text), "figure_differs"),
    (lambda f: Figure("model", f.row, f.period, f.unit, f.value, f.text), "figure_differs"),
])
def test_a_planted_figure_is_refused(built, plant, code):
    s, draft = built
    irr = _first(draft, Figure, lambda f: f.row == "returns.irr")
    with pytest.raises(DraftRefused) as refused:
        memo.check(_swap(draft, irr, plant(irr)), s)
    assert refused.value.code == code


@pytest.mark.parametrize("text, code", [
    ("The deal returns 25% a year.", "number_not_from_model"),
    ("Returns doubled over the period.", "number_not_from_model"),
    ("We would buy this deal.", "call_in_text"),
])
def test_words_holding_a_number_or_a_call_are_refused(built, text, code):
    s, draft = built
    words = _first(draft, Text, lambda t: t.text.strip())
    with pytest.raises(DraftRefused) as refused:
        memo.check(_swap(draft, words, Text(text)), s)
    assert refused.value.code == code


@pytest.mark.parametrize("planted", [Label("deal_name", "Another deal"), Label("company_name", "Projekt Kestrel GmbH"),
                                     Typed("rating", "Buy")])
def test_a_planted_name_or_field_is_refused(built, planted):
    s, draft = built
    name = _first(draft, Label, lambda x: x.source == "deal_name")
    with pytest.raises(DraftRefused) as refused:
        memo.check(_swap(draft, name, planted), s)
    assert refused.value.code == "label_differs"


def test_every_catalogue_sentence_of_the_memo_keeps_the_tools_rules():
    """A translation that slips in a digit or a call would stop every memo: the catalogue's own words are held to
    the rules a sentence is (the unit words and the file title are labels, which may name numbers)."""
    from research.report import words_problem

    doc = memo.catalogue("en")["doc"]
    for key, text in doc.items():
        if key.startswith("unit_") or key == "file_title":
            continue
        assert words_problem(memo_writer.SLOT.sub(" ", text), key.startswith("template_")) is None, key


# ---------------------------------------------------------------------------------------------- the AI
SECRET_NAMES = ("Kestrel", "HRB", "Lazard", "machinery", "GBP", "Projekt")


class GoogleHeard:
    """Google's API as Gemini's own code calls it: every request recorded whole, a note writer's answer given."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def post(self, url, headers=None, json=None, timeout=None, **_):  # noqa: A002
        from tests.test_paid_switch import Reply

        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json})
        return Reply(200, {"candidates": [{"content": {"parts": [{"text": _json(self.answer)}]}}],
                           "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 120}})


def _json(x) -> str:
    return json.dumps(x)


def test_no_company_name_or_identifier_is_sent_on_the_free_provider(monkeypatch):
    monkeypatch.setenv("AI_FREE_TIER", "1")
    gemini = providers.Gemini("SECRET-KEY-123")
    assert gemini.free_tier
    heard = GoogleHeard({"summary": "The deal turns {entry_equity} into {exit_equity}, an IRR of {irr}.",
                         "metrics": "The IRR is above the hurdle of {hurdle}, and {loss} of paths lose money."})
    monkeypatch.setattr(gemini, "_http", heard)
    monkeypatch.setattr(providers.Gemini, "_spent", {})
    name, label = "Kestrel Holdings GmbH (HRB 123456)", "Board pack for Lazard, 2026-10-01"
    s, draft, written = drafted(LISTED, SETTINGS, ACTUALS, name=name, label=label, provider=gemini)
    assert len(heard.requests) == 1
    request = heard.requests[0]
    sent = "".join(part.get("text", "") for part in request["json"]["contents"][0]["parts"])
    everything = _json(request["json"]) + request["url"] + _json({k: v for k, v in request["headers"].items()
                                                                 if k.lower() != "x-goog-api-key"})
    for secret in SECRET_NAMES + (name, label, "2027", "123456", "GB", "ifrs", "Board pack"):
        assert secret not in everything, secret
    assert not any(ch.isdigit() for ch in sent), "a digit reached the AI"
    shown = {f.text for f in all_figures(draft)}
    assert not any(t in sent for t in shown if t not in ("–",))
    # the app filled the figures into the AI's sentences afterwards
    assert written["sections"]["summary"]["source"] == "ai"
    summary = next(b for b in draft.blocks if isinstance(b, Paragraph) and b.kind == "sentence")
    filled = [x for x in summary.parts if isinstance(x, Figure)]
    assert [f.row for f in filled] == ["returns.entry_equity", "returns.net_exit_equity", "returns.irr"]


def test_the_prompt_holds_nothing_of_the_deal_but_which_way_its_figures_point():
    s, _, _ = drafted(LISTED, SETTINGS, ACTUALS, name="Kestrel Holdings GmbH")
    text = memo_writer.prompt(s)
    assert "Kestrel" not in text and not re.search(r"\d", text)
    sections = json.loads(text.split("Sections:\n", 1)[1])
    assert {x["section"] for x in sections} == set(memo_writer.SECTIONS)


class Writes:
    def __init__(self, answer=None, fail=False):
        self.answer, self.fail, self.sent = answer, fail, []

    def write(self, text):
        self.sent.append(text)
        if self.fail:
            raise ProviderError(providers.BUSY)
        return Answer(self.answer, "stand-in", "", {"promptTokenCount": 10, "candidatesTokenCount": 5})


@pytest.mark.parametrize("answer", [
    "The IRR is {irr}, well above twenty.",          # a number word
    "The IRR is {irr} on 5 years.",                   # a digit
    "The IRR of {irr} makes it attractive.",          # a call
    "The IRR is {irr} and the price {price}.",        # a slot not the section's
    "No slot at all.",
    "{irr} " + "x" * 700,                              # too long
])
def test_an_ai_sentence_that_breaks_a_rule_is_written_by_its_template(answer):
    s, draft, written = drafted(PLAIN, SETTINGS, None, provider=Writes({"summary": answer}))
    assert written["sections"]["summary"]["source"] == "ai_refused"
    assert written["sections"]["summary"]["text"] == memo_writer.template("summary", s)


def test_without_an_ai_or_when_it_fails_the_templates_write_and_say_why():
    _, _, off = drafted(PLAIN, SETTINGS, None)
    assert {v["source"] for v in off["sections"].values()} == {"ai_off"}
    _, _, failed = drafted(PLAIN, SETTINGS, None, provider=Writes(fail=True))
    assert {v["source"] for v in failed["sections"].values()} == {"ai_failed"}


def test_a_stored_ai_sentence_stands_while_its_directions_still_hold():
    s, _, written = drafted(PLAIN, SETTINGS, None, provider=Writes({"summary": "Equity grows to {exit_equity}."}))
    assert memo_writer.stored(written, s)["summary"] == ("Equity grows to {exit_equity}.", "ai")
    moved = json.loads(json.dumps(written))
    moved["sections"]["summary"]["directions"] = ["the equity at exit is lower than the equity invested at entry"]
    assert memo_writer.stored(moved, s)["summary"][1] == "template"


# ---------------------------------------------------------------------------------------------- the language
@pytest.fixture
def german(tmp_path, monkeypatch):
    english = json.loads((memo.MESSAGES / "en.json").read_text(encoding="utf-8"))
    (tmp_path / "en.json").write_text(json.dumps(english), encoding="utf-8")
    de = {"memo": {"doc": {"title": "Investitionsmemo", "h_summary": "Zusammenfassung",
                           "template_summary": "Der Deal bringt {irr} im Jahr."}},
          "engine": {"driverGrowth": "Wachstum"}}
    (tmp_path / "de.json").write_text(json.dumps(de), encoding="utf-8")
    monkeypatch.setattr(memo, "MESSAGES", tmp_path)
    memo.catalogue.cache_clear()
    yield
    memo.catalogue.cache_clear()


def test_a_memo_is_written_in_the_readers_language_with_english_for_what_it_lacks(german):
    assert memo.language_for("de-DE") == "de" and memo.language_for("fr-FR") == "en"
    ai = Writes({"summary": "The deal returns {irr}."})
    s, draft, written = drafted(PLAIN, SETTINGS, None, language="de", provider=ai)
    assert ai.sent == [] and {v["source"] for v in written["sections"].values()} == {"not_english"}
    words = " ".join(seg.text for seg, _, _ in segments(draft) if isinstance(seg, (Text, Label)))
    assert "Investitionsmemo" in words and "Zusammenfassung" in words and "Wachstum" in words
    assert "Der Deal bringt" in words and "Where the gain comes from" in words


def test_figures_follow_the_readers_digit_grouping():
    s, draft, _ = drafted({"ebitda": 15000.0, "unit": "crores", "currency": "INR"}, {}, None, grouping="lakh")
    money = [f for f in all_figures(draft) if f.unit == "money" and f.value and abs(f.value) >= 100000]
    assert money and all(re.fullmatch(r"-?\d{1,2}(,\d{2})*,\d{3}\.\d", f.text) for f in money)


def test_show_writes_each_unit():
    assert memo.show(0.2116, "share") == "21.2%" and memo.show(2.611, "times") == "2.6x"
    assert memo.show(1.923, "times2") == "1.92x" and memo.show(12.88, "pct") == "12.9%"
    assert memo.show(None, "pct") == "–" and memo.show(math.nan, "money") == "–"
    with pytest.raises(KeyError):
        memo.unit_of("returns.made_up")


def test_the_api_image_carries_the_memos_words():
    """The memo's words live in web/messages, which the API reads: the image must copy them and a change to them
    must rebuild it (CLAUDE.md "Gotchas": a new top-level directory the API reads goes in three places)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert "COPY web/messages ./web/messages" in (root / "Dockerfile").read_text()
    assert "!web/messages" in (root / ".dockerignore").read_text()
    assert "- web/messages/**" in (root / "render.yaml").read_text()
