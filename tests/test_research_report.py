"""Results notes (PLAN.md R8): every figure in a draft is its model's, a planted wrong number is refused, and no
draft holds a rating the analyst did not type.

The figures are checked here against an evaluation of the model made in the test, not against the draft's own
check, so a check that waved a wrong figure through would still be caught. The AI is a stand-in that answers what
each test gives it: what it is sent (no company, no figure) and what is done with its answer (a number, a call or
an unknown slot sends the section back to its template) are what is tested.
"""
from __future__ import annotations

import dataclasses
import io
import math
import re
import zipfile

import pytest
from pypdf import PdfReader

from research import report, report_files, writer
from research.evaluate import evaluate
from research.reader.providers import Answer, ProviderError
from research.report import (
    CALL_IN_TEXT, FIGURE_DIFFERS, LABEL_DIFFERS, NUMBER_NOT_FROM_MODEL, TYPED_DIFFERS, DraftRefused, Figure, Label,
    Paragraph, Table, Text, Typed,
)
from research.rollforward import roll
from tests.test_research_rollforward import AS_OF, half_year, yatharth


def sources(grouping="thousands", **changes):
    model, company = yatharth()
    s = report.Sources(model=model, company=company, result=evaluate(model, company, partial=True), grouping=grouping)
    s = dataclasses.replace(s, **changes)
    return dataclasses.replace(s, sentences=writer.stored_sentences(None, s))


def rolled_sources(**changes):
    """Yatharth rolled forward on its half year: the comparison it stored, and the version from before as the one
    the estimates are set against."""
    model, company = yatharth()
    before = report.Earlier(3, model, company, evaluate(model, company, partial=True))
    rolled = roll(model, company, AS_OF, half_year(model, company))
    s = report.Sources(model=model, company=rolled.company, result=evaluate(model, rolled.company, partial=True),
                       comparison=rolled.comparison, earlier=before)
    s = dataclasses.replace(s, **changes)
    return dataclasses.replace(s, sentences=writer.stored_sentences(None, s))


def figures(draft):
    return [seg for seg, _, _ in report.segments(draft) if isinstance(seg, Figure)]


def whole_text(draft) -> str:
    return " ".join(seg.text for seg, _, _ in report.segments(draft))


class Writes:
    """A stand-in AI: answers `answer` and keeps what it was sent."""

    def __init__(self, answer=None, fails=False):
        self.answer, self.fails, self.sent = answer, fails, []

    def write(self, text):
        self.sent.append(text)
        if self.fails:
            raise ProviderError("allowance_spent")
        return Answer(self.answer, "stand-in", "")


# ---------------------------------------------------------------------------------------------- every figure
@pytest.mark.parametrize("make", [sources, rolled_sources], ids=["saved", "rolled"])
def test_every_figure_in_a_draft_is_its_models_cell_shown_as_the_note_shows_it(make):
    s = make()
    draft = report.build(s)
    found = figures(draft)
    assert len(found) > 40
    model_result = evaluate(s.model, s.company, partial=True)
    units = {r.key: r.unit for r in s.model.data_rows()}
    for f in found:
        if f.source == "model":
            value, unit = model_result.value(f.row, f.period), units[f.row]
        elif f.source == "earlier":
            value = evaluate(s.earlier.model, s.earlier.company, partial=True).value(f.row, f.period)
            unit = units[f.row]
        else:
            index, _, what = f.row.partition(".")
            value, unit = s.comparison["lines"][int(index)][what], report.COMPARISON_FIELDS[what]
        assert f.value == pytest.approx(value, rel=1e-12), (f.source, f.row, f.period)
        assert f.unit == unit and f.text == report.show(value, unit, s.grouping)
    # and the note says them: last year's revenue as the filing has it, the EPS to the paisa
    assert report.show(12071.72, "money") == "12,071.7"
    assert "12,071.7" in [f.text for f in found if (f.row, f.period) == ("rev", "ly")]


def test_figures_follow_the_accounts_digit_grouping():
    assert report.number(12345678.9, 1, "lakh") == "1,23,45,678.9"
    assert report.number(-12345678.9, 1, "thousands") == "-12,345,678.9"
    assert report.number(-0.04, 1) == "0.0" and report.show(0.2412, "share") == "24.1%"
    assert report.show(None, "money") == report.MISSING and report.show(46311, "date") == "2026-10-16"
    assert report.grouping_for("locale", "en-IN") == "lakh" and report.grouping_for("locale", "en-GB") == "thousands"
    assert report.grouping_for("lakh", "en-US") == "lakh" and report.grouping_for("thousands", "hi-IN") == "thousands"
    lakh = report.build(sources("lakh"))
    revenue = [f for f in figures(lakh) if (f.row, f.period) == ("rev", "y1")][0]
    assert revenue.text == report.number(revenue.value, 1, "lakh") and revenue.text.count(",") >= 1


def test_the_note_has_its_sections_and_charts_from_the_same_cells():
    draft = report.build(rolled_sources())
    headings = [report.plain(b.parts) for b in draft.blocks if isinstance(b, report.Heading)]
    assert headings[1:] == ["The analyst's view", "Summary", "Results against the estimate", "Changes to estimates",
                            "Valuation", "The argument", "Disclosures"]
    charts = [b for b in draft.blocks if isinstance(b, report.Chart)]
    assert len(charts) == 2
    summary = next(b for b in draft.blocks if isinstance(b, Table))
    revenue_row = next(r for r in summary.rows if r[0][0].source == "row:rev")
    assert [c[0] for c in revenue_row[1:]] == list(charts[0].series[0][1])


# ---------------------------------------------------------------------------------------------- planted numbers
def _replace_first(draft, test, make):
    """The draft with the first segment for which `test` holds replaced by `make(segment)`."""
    done = []

    def swap(parts):
        out = []
        for seg in parts:
            if not done and test(seg):
                done.append(seg)
                seg = make(seg)
            out.append(seg)
        return tuple(out)

    blocks = []
    for b in draft.blocks:
        if isinstance(b, (report.Heading, Paragraph)):
            b = dataclasses.replace(b, parts=swap(b.parts))
        elif isinstance(b, Table):
            b = dataclasses.replace(b, header=tuple(swap(c) for c in b.header),
                                    rows=tuple(tuple(swap(c) for c in row) for row in b.rows))
        elif isinstance(b, report.Chart):
            b = dataclasses.replace(b, series=tuple((n, swap(figs)) for n, figs in b.series))
        blocks.append(b)
    assert done, "nothing to plant in"
    return dataclasses.replace(draft, blocks=tuple(blocks))


@pytest.mark.parametrize("plant", [
    lambda f: dataclasses.replace(f, value=f.value * 1.01),                    # another value, shown as its own
    lambda f: dataclasses.replace(f, value=f.value * 1.01, text=report.show(f.value * 1.01, f.unit)),
    lambda f: dataclasses.replace(f, text=f.text.replace("1", "7", 1) if "1" in f.text else "7" + f.text),
    lambda f: dataclasses.replace(f, period="y3" if f.period != "y3" else "y2"),
    lambda f: dataclasses.replace(f, source="earlier"),                         # no earlier version to read
], ids=["value", "value_and_text", "text", "period", "source"])
def test_a_planted_wrong_figure_is_refused(plant):
    s = sources()
    draft = report.build(s)
    planted = _replace_first(draft, lambda seg: isinstance(seg, Figure) and seg.row == "rev" and seg.period == "y1",
                             plant)
    with pytest.raises(DraftRefused) as refused:
        report.check(planted, s)
    assert refused.value.code == FIGURE_DIFFERS


@pytest.mark.parametrize("words, code", [
    ("Revenue grew 18% in the year.", NUMBER_NOT_FROM_MODEL),
    ("Revenue nearly doubled.", NUMBER_NOT_FROM_MODEL),
    ("Margins rose by two points.", NUMBER_NOT_FROM_MODEL),
    ("Revenue reached Rs 1,650 crore.", NUMBER_NOT_FROM_MODEL),
    ("We would buy the shares.", CALL_IN_TEXT),
])
def test_a_number_or_a_call_planted_in_the_tools_words_is_refused(words, code):
    s = sources()
    draft = report.build(s)
    planted = _replace_first(draft, lambda seg: isinstance(seg, Text) and seg.text == "Summary",
                             lambda seg: Text(words))
    with pytest.raises(DraftRefused) as refused:
        report.check(planted, s)
    assert refused.value.code == code


def test_a_label_or_a_typed_field_that_is_not_its_sources_is_refused():
    s = sources(fields=report.Fields(rating="Add", argument="Volumes recover."))
    draft = report.build(s)
    wrong_label = _replace_first(draft, lambda seg: isinstance(seg, Label) and seg.source == "period:cy",
                                 lambda seg: Label(seg.source, "FY99"))
    wrong_rating = _replace_first(draft, lambda seg: isinstance(seg, Typed) and seg.field == "rating",
                                  lambda seg: Typed("rating", "Buy"))
    rating_as_text = _replace_first(draft, lambda seg: isinstance(seg, Typed) and seg.field == "rating",
                                    lambda seg: Text("Add"))
    for planted, code in ((wrong_label, LABEL_DIFFERS), (wrong_rating, TYPED_DIFFERS)):
        with pytest.raises(DraftRefused) as refused:
            report.check(planted, s)
        assert refused.value.code == code
    with pytest.raises(DraftRefused) as refused:    # typed, "Add" is the analyst's; in the tool's words, a call
        report.check(rating_as_text, s)
    assert refused.value.code == CALL_IN_TEXT


# ---------------------------------------------------------------------------------------------- the analyst's call
RATINGS = re.compile(r"\b(buy|sell|hold|accumulate|reduce|add|outperform|underperform|overweight|underweight|"
                     r"neutral)\b", re.IGNORECASE)


@pytest.mark.parametrize("make", [sources, rolled_sources], ids=["saved", "rolled"])
def test_no_draft_holds_a_rating_or_a_target_the_analyst_did_not_type(make):
    s = make()
    draft = report.build(s)
    assert not RATINGS.search(whole_text(draft))
    assert not any(isinstance(seg, Typed) for seg, _, _ in report.segments(draft))
    typed = make(fields=report.Fields(rating="Accumulate", target=412.5, argument="Beds open on time.",
                                      note_date="2026-10-20"))
    note = report.build(typed)
    ratings = [seg for seg, _, _ in report.segments(note) if RATINGS.search(seg.text)]
    assert ratings == [Typed("rating", "Accumulate")]
    assert Typed("target", "412.50") in [seg for seg, _, _ in report.segments(note)]
    # the AI's sentences never carry one, however it is asked: a call sends the section to its template
    ai = Writes({"outlook": "Revenue climbs from {rev_ly} to {rev_cy}, so we rate the shares a buy.",
                 "valuation": "The value of {dcf_value} is attractive against {price}."})
    written = writer.write(make(), ai)["sections"]
    assert written["outlook"]["source"] == writer.AI_REFUSED and written["valuation"]["source"] == writer.AI_REFUSED


# ---------------------------------------------------------------------------------------------- the AI's sentences
def test_the_ai_is_sent_no_company_no_words_of_its_sector_and_no_figure():
    s = rolled_sources()
    ai = Writes({})
    writer.write(s, ai)
    sent = ai.sent[0]
    assert s.company.name not in sent and "Yatharth" not in sent and "hospital" not in sent.lower()
    assert "beds" not in sent.lower()
    shown = {f.text for f in figures(report.build(s)) if f.text != report.MISSING}
    for text in shown:
        assert text not in sent, text
    assert not re.search(r"\d", re.sub(r"\{[a-z0-9_]+\}|\"[a-z0-9_]+\"", "", sent))


def test_the_ais_sentences_are_kept_with_their_slots_and_filled_from_the_model():
    s = sources()
    ai = Writes({"outlook": "Revenue rises from {rev_ly} in {ly} to {rev_cy} in {cy}.",
                 "valuation": "The discounted cash flow gives {dcf_value} a share; the price typed is {price}."})
    written = writer.write(s, ai)
    assert written["sections"]["outlook"] == {"text": "Revenue rises from {rev_ly} in {ly} to {rev_cy} in {cy}.",
                                              "source": writer.AI, "directions": writer.facts(s)["outlook"][1]}
    assert written["model"] == "stand-in" and written["prompt_version"] == writer.PROMPT_VERSION
    s = dataclasses.replace(s, sentences=writer.stored_sentences(written, s))
    draft = report.build(s)
    sentence = next(b for b in draft.blocks if isinstance(b, Paragraph) and b.kind == "sentence")
    rev = [seg for seg in sentence.parts if isinstance(seg, Figure)]
    assert [(f.row, f.period) for f in rev] == [("rev", "ly"), ("rev", "cy")]
    assert report.plain(sentence.parts).startswith("Revenue rises from 12,071.7 in FY26 (actual) to ")


@pytest.mark.parametrize("answer", [
    "Revenue grows to {rev_cy}, up 37%.",                    # a digit
    "Revenue grows by nearly half to {rev_cy}.",             # a number word
    "Revenue grows to {rev_cy} against {dcf_value}.",        # another section's slot
    "Revenue grows to {revenue_next}.",                      # a slot nobody made
    "Revenue grows strongly.",                               # no slot at all
    "Revenue grows to {rev_cy} {",                           # a broken slot
    "x" * 700 + " {rev_cy}",                                 # too long
    42,                                                      # not text
])
def test_an_ai_sentence_that_breaks_a_rule_is_written_by_its_template(answer):
    s = sources()
    written = writer.write(s, Writes({"outlook": answer}))["sections"]["outlook"]
    assert written == {"text": writer.template("outlook", s), "source": writer.AI_REFUSED}


def test_without_the_ai_or_when_it_fails_the_templates_write_and_say_why():
    s = rolled_sources()
    off = writer.write(s, None)
    failed = writer.write(s, Writes(fails=True))
    assert set(off["sections"]) == set(writer.SECTIONS)
    assert {v["source"] for v in off["sections"].values()} == {writer.AI_OFF}
    assert {v["source"] for v in failed["sections"].values()} == {writer.AI_FAILED}
    draft = report.build(dataclasses.replace(s, sentences=writer.stored_sentences(off, s)))
    assert sum(1 for b in draft.blocks if isinstance(b, Paragraph) and b.kind == "sentence") == 4


def test_a_stored_sentence_whose_slot_the_model_no_longer_has_falls_back_to_its_template():
    s = sources()
    stored = {"sections": {"outlook": {"text": "Revenue was {rev_ly}; next it is {rev_y9}.", "source": writer.AI}}}
    assert writer.stored_sentences(stored, s)["outlook"] == writer.template("outlook", s)


# ---------------------------------------------------------------------------------------------- the files
def test_the_word_file_holds_the_drafts_text_and_native_charts_of_its_figures():
    s = rolled_sources(fields=report.Fields(rating="Add", argument="Line one.\nLine two."),
                       details=report.Details("Example Securities", "Registered as a research analyst."))
    draft = report.build(s)
    data = report_files.word(draft, "Note")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        document = z.read("word/document.xml").decode()
        charts = sorted(n for n in z.namelist() if n.startswith("word/charts/"))
        chart = z.read(charts[0]).decode()
        assert "variater.com" in z.read("docProps/core.xml").decode()
    assert len(charts) == 2
    for f in figures(draft):
        assert f.text in document or f.text == report.MISSING or any(
            b for b in draft.blocks if isinstance(b, report.Chart) and f in sum((tuple(x[1]) for x in b.series), ()))
    for word in ("Example Securities", "Registered as a research analyst.", "Line one.", "Line two.", ">Add<"):
        assert word in document
    revenue = next(b for b in draft.blocks if isinstance(b, report.Chart)).series[0][1]
    values = [float(v) for v in re.findall(r"<c:v>([-0-9.e]+)</c:v>", chart)]
    assert all(any(math.isclose(f.value, v) for v in values) for f in revenue if f.value is not None)


def test_the_pdf_holds_the_drafts_text_and_figures():
    s = rolled_sources(details=report.Details("Example Securities", ""))
    draft = report.build(s)
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(report_files.pdf(draft, "Note"))).pages)
    flat = " ".join(text.split())
    assert "Example Securities" in flat and s.company.name in flat
    for f in figures(draft):
        if f.text != report.MISSING:
            assert f.text in flat, f.text


def test_a_draft_is_checked_before_a_file_is_written_from_it():
    """The writers take only what `build` answers, and `build` checks: a model whose figures break the rules never
    reaches a file."""
    s = sources()
    s = dataclasses.replace(s, sentences={"outlook": "Revenue grows twelve times to {rev_cy}."})
    with pytest.raises(DraftRefused) as refused:
        report.build(s)
    assert refused.value.code == NUMBER_NOT_FROM_MODEL


def test_the_report_screens_words_name_a_rating_or_a_target_only_as_the_analysts_own_fields():
    """The screen labels the analyst's fields ("Rating, in your words"); nothing else on it words a call."""
    import json
    from pathlib import Path

    from tests.test_research_valuation import calls_in

    words = json.loads((Path(__file__).parents[1] / "web" / "messages" / "en.json").read_text(encoding="utf-8"))
    fields = {"ratingLabel", "targetLabel", "targetInvalid"}
    found = {key: calls_in(text) for key, text in words["report"].items() if calls_in(text)}
    assert set(found) <= fields, found
    assert "picks none" in words["report"]["tool"]


def test_an_ai_sentence_whose_directions_changed_is_written_by_its_template():
    """A sentence written when revenue rose must not stand beside figures that now fall."""
    s = sources()
    ai = Writes({"outlook": "Revenue rises from {rev_ly} in {ly} to {rev_cy} in {cy}."})
    written = writer.write(s, ai)
    assert writer.stored(written, s)["outlook"][1] == writer.AI
    company = s.company
    values = {**company.values, "rev": {**company.values["rev"], "ly": company.values["rev"]["ly"] * 10}}
    lower = dataclasses.replace(company, values=values)
    t = sources(company=lower, result=evaluate(s.model, lower, partial=True))
    assert writer.stored(written, t)["outlook"] == (writer.template("outlook", t), writer.TEMPLATE)


# ---------------------------------------------------------------------------------------------- the providers' side
def test_gemini_writes_with_the_first_model_that_has_allowance_and_skips_a_spent_one(monkeypatch):
    from research.reader import providers

    monkeypatch.setattr(providers.Gemini, "_spent", {})
    monkeypatch.setenv("GEMINI_MODELS_WRITE", "model-a,model-b")
    gemini = providers.Gemini("SECRET-KEY-123")
    asked = []

    def generate(model, parts):
        asked.append((model, parts))
        if model == "model-a":
            raise ProviderError(providers.SPENT)
        return '{"outlook": "Revenue rises to {rev_cy}."}', {"tokens": 9}

    monkeypatch.setattr(gemini, "_generate", generate)
    answer = gemini.write("the prompt")
    assert answer.read == {"outlook": "Revenue rises to {rev_cy}."} and answer.model == "model-b"
    assert [m for m, _ in asked] == ["model-a", "model-b"] and asked[1][1] == [{"text": "the prompt"}]
    assert gemini.write("again").model == "model-b" and [m for m, _ in asked][-1] == "model-b"   # a's day is spent
    monkeypatch.setattr(gemini, "_generate", lambda m, p: (_ for _ in ()).throw(ProviderError(providers.KEY_REFUSED)))
    with pytest.raises(ProviderError):
        gemini.write("refused")


def test_replay_answers_a_recorded_writing_by_its_prompt_and_refuses_one_not_recorded(monkeypatch, tmp_path):
    import json as _json

    from research.reader import providers

    monkeypatch.setattr(providers, "WRITINGS", tmp_path)
    (tmp_path / ("%s.json" % providers.sha256(b"the prompt"))).write_text(
        _json.dumps({"answer": {"outlook": "x {rev_cy}"}, "model": "recorded"}), encoding="utf-8")
    assert providers.Replay().write("the prompt").read == {"outlook": "x {rev_cy}"}
    with pytest.raises(ProviderError):
        providers.Replay().write("another prompt")
