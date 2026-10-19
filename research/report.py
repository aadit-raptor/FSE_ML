"""A results note drafted from a saved sector model (PLAN.md R8).

**Every number comes from the model.** A draft is a list of blocks (a
heading, a paragraph, a table, a chart) made of segments, and each segment
says where it came from:

- ``Figure``: a cell of the saved model (``model``), of the version it is set
  against (``earlier``), or a figure of the comparison a roll forward stored
  (``comparison``), with its value and the text it is shown as;
- ``Label``: a name the model, the company or the account holds (a period's
  label, a row's words, the firm's name), which may hold digits ("FY27");
- ``Typed``: what the analyst typed for this note (the argument, the rating,
  the target, the date); the tool never writes in these;
- ``Text``: the tool's own words, and the AI's linking sentences: **no digit,
  no number word and no rating** may stand in them.

``check`` works every one of them out again from its source and refuses the
draft (``DraftRefused``) when a figure's value or text differs, a label is
not its source's, or a word of the tool's holds a number or a call. The Word
and PDF writers only ever write a checked draft.

**A tool, not an adviser** (phase R, rule 1): the rating, the target and the
argument are the analyst's fields, empty until typed; the tool's words never
name a rating (``RATING_WORDS``) or a call (``CALLS``).
"""
from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from research.definition import Company, Model, label, words_for
from research.evaluate import Result
from research.pieces import CY, LY, ROY, YTD
from research.valuation import MULTIPLES, VALUE_MULTIPLE, VALUE_YEAR

# ---------------------------------------------------------------------------------------------- what a call is
# Ratings, in any case: none may stand in the tool's words (the analyst's own fields are theirs)
RATING_WORDS = re.compile(r"\b(buy|sell|hold|accumulate|reduce|outperform|underperform|overweight|underweight|"
                          r"neutral|market\s*perform|sector\s*perform)\b", re.IGNORECASE)
# What a recommendation is worded with: refused in the AI's sentences (the field labels are the tool's own)
CALLS = re.compile(r"\b(recommend\w*|rating|rated|target\s*price|price\s*target|upside|downside|undervalued|"
                   r"overvalued|cheap|expensive|attractive|bullish|bearish|fair\s*value)\b", re.IGNORECASE)
NUMBER_WORDS = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
                          r"forty|fifty|hundred|thousand|lakhs?|crores?|millions?|billions?|half|halved|double|"
                          r"doubled|triple|tripled|twice|percent|per\s*cent|dozen)\b", re.IGNORECASE)

# Refusals, as codes the screen words (web/messages/en.json report.refused_*)
FIGURE_DIFFERS = "figure_differs"           # a figure is not its cell's value, or not shown as it
NUMBER_NOT_FROM_MODEL = "number_not_from_model"     # a digit or number word in the tool's words
LABEL_DIFFERS = "label_differs"
TYPED_DIFFERS = "typed_differs"
CALL_IN_TEXT = "call_in_text"               # a rating or a recommendation in the tool's words
NO_MODEL_VALUES = "no_model_values"


class DraftRefused(ValueError):
    def __init__(self, code: str, where: str):
        super().__init__("%s: %s" % (code, where))
        self.code, self.where = code, where


# ---------------------------------------------------------------------------------------------- segments
@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Label:
    source: str                 # company_name, basis, unit, firm, period:<p>, row:<key>, line:<i>, choice:<name>
    text: str


@dataclass(frozen=True)
class Figure:
    source: str                 # model, earlier, comparison
    row: str                    # a row's key; for the comparison, ``<line index>.<field>``
    period: str                 # a period's name ("" for the comparison)
    unit: str
    value: Optional[float]
    text: str


@dataclass(frozen=True)
class Typed:
    field: str                  # argument, rating, target, note_date, disclosures
    text: str


Segment = Union[Text, Label, Figure, Typed]


@dataclass(frozen=True)
class Heading:
    parts: tuple
    level: int = 1


@dataclass(frozen=True)
class Paragraph:
    parts: tuple
    kind: str = "text"          # text, sentence (linking), analyst, note


@dataclass(frozen=True)
class Table:
    header: tuple               # cells; a cell is a tuple of segments
    rows: tuple


@dataclass(frozen=True)
class Chart:
    title: tuple
    categories: tuple           # Labels
    series: tuple               # (name segments, tuple of Figures)
    unit: str = "money"


Block = Union[Heading, Paragraph, Table, Chart]


@dataclass(frozen=True)
class Draft:
    blocks: tuple
    grouping: str = "thousands"


# ---------------------------------------------------------------------------------------------- what a draft reads
@dataclass(frozen=True)
class Fields:
    """What the analyst types for the note. Empty until typed: the tool fills none of them."""
    argument: str = ""
    rating: str = ""
    target: Optional[float] = None
    note_date: str = ""             # ISO date
    compare_version: Optional[int] = None


@dataclass(frozen=True)
class Details:
    """The account's firm name and disclosures, typed once for every note."""
    firm: str = ""
    disclosures: str = ""


@dataclass(frozen=True)
class Earlier:
    """The version the estimates are set against."""
    number: int
    model: Model
    company: Company
    result: Result


@dataclass(frozen=True)
class Sources:
    model: Model
    company: Company
    result: Result
    fields: Fields = Fields()
    details: Details = Details()
    comparison: Optional[dict] = None       # the latest roll forward's, as stored with its version
    earlier: Optional[Earlier] = None
    grouping: str = "thousands"             # thousands or lakh
    sentences: dict = field(default_factory=dict)   # section -> text with {slot}s (research/writer.py)


# ---------------------------------------------------------------------------------------------- numbers as text
DECIMALS = {"money": 1, "count": 0, "count3": 3, "days": 0, "share": 1, "rupees": 2, "rupees2": 2, "times": 1,
            "factor": 3, "check": 2}
MISSING = "–"          # an en dash stands where the model holds nothing


def _grouped(digits: str, grouping: str) -> str:
    if grouping == "lakh" and len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        pairs = []
        while len(head) > 2:
            pairs.insert(0, head[-2:])
            head = head[:-2]
        return ",".join(([head] if head else []) + pairs + [tail])
    return "{:,}".format(int(digits)) if digits else "0"


def number(value: float, decimals: int, grouping: str = "thousands") -> str:
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0                       # never "-0.0"
    text = "%.*f" % (decimals, abs(rounded))
    whole, _, part = text.partition(".")
    out = _grouped(whole, grouping) + ("." + part if part else "")
    return ("-" if rounded < 0 else "") + out


def excel_date(serial: float) -> str:
    return (datetime.date(1899, 12, 30) + datetime.timedelta(days=int(serial))).isoformat()


def show(value: Optional[float], unit: str, grouping: str = "thousands") -> str:
    """How a figure is written in the note."""
    if value is None or not math.isfinite(value):
        return MISSING
    if unit == "date":
        return excel_date(value)
    if unit == "share":
        return number(value * 100, DECIMALS["share"], grouping) + "%"
    if unit == "times":
        return number(value, DECIMALS["times"], grouping) + "x"
    return number(value, DECIMALS.get(unit, 2), grouping)


def grouping_for(digit_grouping: str, locale: str) -> str:
    """The account's digit grouping (PLAN.md 2.3a) as the note writes it: lakh, or thousands."""
    if digit_grouping == "lakh" or (digit_grouping == "locale" and locale.upper().endswith("-IN")):
        return "lakh"
    return "thousands"


# ---------------------------------------------------------------------------------------------- reading a source
def _rows(model: Model) -> dict:
    return {r.key: r for r in model.data_rows()}


def model_figure(s: Sources, row: str, period: str, source: str = "model") -> Figure:
    model, result = (s.model, s.result) if source == "model" else (s.earlier.model, s.earlier.result)
    unit = _rows(model)[row].unit
    value = result.value(row, period)
    return Figure(source, row, period, unit, value, show(value, unit, s.grouping))


COMPARISON_FIELDS = {"estimate": "money", "actual": "money", "difference": "money", "share": "share"}


def comparison_figure(s: Sources, index: int, what: str) -> Figure:
    value = s.comparison["lines"][index][what]
    unit = COMPARISON_FIELDS[what]
    return Figure("comparison", "%d.%s" % (index, what), "", unit, value, show(value, unit, s.grouping))


def period_label(company: Company, period: str) -> str:
    return str(company.period_labels.get(period) or period.upper())


def row_words(model: Model, company: Company, row: str) -> str:
    return label(_rows(model)[row].label, words_for(model, company)).strip()


CHOICE_WORDS = {VALUE_MULTIPLE: {"ev_ebitda": "EV/EBITDA", "pe": "P/E"}}


def _label_text(s: Sources, source: str) -> str:
    kind, _, name = source.partition(":")
    if kind == "company_name":
        return s.company.name
    if kind == "basis":
        return s.company.basis
    if kind == "unit":
        return s.company.unit_name
    if kind == "firm":
        return s.details.firm
    if kind == "period":
        return period_label(s.company, name)
    if kind == "earlier_period":
        return period_label(s.earlier.company, name)
    if kind == "row":
        return row_words(s.model, s.company, name)
    if kind == "line":
        return str(s.comparison["lines"][int(name)]["label"])
    if kind == "comparison_period":
        return str(s.comparison.get("period") or "")
    if kind == "comparison_estimate":
        return str(s.comparison.get("estimate_label") or "")
    if kind == "choice":
        return CHOICE_WORDS[name][s.company.choices[name]]
    if kind == "version":
        return str(s.earlier.number)
    raise KeyError(source)


def make_label(s: Sources, source: str) -> Label:
    return Label(source, _label_text(s, source))


def _typed_text(s: Sources, name: str) -> str:
    f = s.fields
    if name == "target":
        return "" if f.target is None else number(f.target, 2, s.grouping)
    if name == "disclosures":
        return s.details.disclosures
    return str(getattr(f, name))


def typed(s: Sources, name: str) -> Typed:
    return Typed(name, _typed_text(s, name))


# ---------------------------------------------------------------------------------------------- the check
def _figure_value(s: Sources, f: Figure) -> tuple[Optional[float], str]:
    if f.source == "comparison":
        index, _, what = f.row.partition(".")
        line = s.comparison["lines"][int(index)]
        return line[what], COMPARISON_FIELDS[what]
    model, result = (s.model, s.result) if f.source == "model" else (s.earlier.model, s.earlier.result)
    return result.value(f.row, f.period), _rows(model)[f.row].unit


def words_problem(text: str, strict: bool = True) -> Optional[str]:
    """Why the tool's (or, `strict`, the AI's) words may not stand in a note, or None. The tool's own field labels
    ("Rating:", "Target") are words of a call; only the AI's sentences are held to `CALLS` too."""
    if any(ch.isdigit() for ch in text) or NUMBER_WORDS.search(text):
        return NUMBER_NOT_FROM_MODEL
    if RATING_WORDS.search(text) or (strict and CALLS.search(text)):
        return CALL_IN_TEXT
    return None


def _check_segment(s: Sources, seg: Segment, where: str, strict: bool = False) -> None:
    if isinstance(seg, Figure):
        if seg.source not in ("model", "earlier", "comparison") or (seg.source == "earlier" and s.earlier is None) \
                or (seg.source == "comparison" and s.comparison is None):
            raise DraftRefused(FIGURE_DIFFERS, where)
        try:
            value, unit = _figure_value(s, seg)
        except (KeyError, IndexError, ValueError):
            raise DraftRefused(FIGURE_DIFFERS, where) from None
        same = (value is None and seg.value is None) or (
            value is not None and seg.value is not None and math.isclose(value, seg.value, rel_tol=1e-12, abs_tol=1e-9))
        if not same or seg.unit != unit or seg.text != show(value, unit, s.grouping):
            raise DraftRefused(FIGURE_DIFFERS, where)
    elif isinstance(seg, Label):
        try:
            expected = _label_text(s, seg.source)
        except (KeyError, IndexError, ValueError, AttributeError):
            raise DraftRefused(LABEL_DIFFERS, where) from None
        if seg.text != expected:
            raise DraftRefused(LABEL_DIFFERS, where)
    elif isinstance(seg, Typed):
        if seg.field not in ("argument", "rating", "target", "note_date", "disclosures") \
                or seg.text != _typed_text(s, seg.field):
            raise DraftRefused(TYPED_DIFFERS, where)
    elif isinstance(seg, Text):
        problem = words_problem(seg.text, strict)
        if problem:
            raise DraftRefused(problem, where)
    else:
        raise DraftRefused(LABEL_DIFFERS, where)


def segments(draft: Draft):
    """Every segment of a draft, with where it stands and whether it is held to the AI's rules (a linking
    sentence)."""
    for i, block in enumerate(draft.blocks):
        if isinstance(block, (Heading, Paragraph)):
            strict = isinstance(block, Paragraph) and block.kind == "sentence"
            for seg in block.parts:
                yield seg, "block %d" % i, strict
        elif isinstance(block, Table):
            for r, row in enumerate((block.header,) + tuple(block.rows)):
                for c, cell in enumerate(row):
                    for seg in cell:
                        yield seg, "block %d, row %d, column %d" % (i, r, c), False
        elif isinstance(block, Chart):
            for seg in block.title + block.categories:
                yield seg, "block %d" % i, False
            for name, figures in block.series:
                for seg in tuple(name) + tuple(figures):
                    yield seg, "block %d" % i, False


def check(draft: Draft, s: Sources) -> Draft:
    """The draft, if every segment is its source's; else ``DraftRefused`` naming the first that is not."""
    if draft.grouping != s.grouping:
        raise DraftRefused(FIGURE_DIFFERS, "grouping")
    for seg, where, strict in segments(draft):
        _check_segment(s, seg, where, strict)
    return draft


# ---------------------------------------------------------------------------------------------- building a draft
SUMMARY_ROWS = ("rev", "ebitda", "ebitda_m", "pat_own", "eps")
VALUE_ROWS = (("v_rate", LY), ("v_g", LY), ("v_mult", LY), ("v_ps", LY), ("v_mps", LY), ("v_price", LY),
              ("v_date", LY), ("v_gap_dcf", LY), ("v_gap_mult", LY))


def years(model: Model) -> list[str]:
    """The note's columns: last year, the current year and every forecast year."""
    return [LY, CY] + [p.name for p in model.periods if p.kind == "forecast"]


def _cell(*parts) -> tuple:
    return tuple(parts)


def _t(text: str) -> tuple:
    return (Text(text),)


def sentence(s: Sources, section: str, slots: dict) -> Optional[Paragraph]:
    """A linking sentence: its text with each ``{slot}`` replaced by the figure or label the slot stands for."""
    from research.writer import parse

    text = s.sentences.get(section)
    if not text:
        return None
    return Paragraph(tuple(parse(text, slots)), kind="sentence")


def _summary(s: Sources, have: dict) -> list:
    cols = years(s.model)
    rows = [r for r in SUMMARY_ROWS if r in have]
    header = (_t("") ,) + tuple((make_label(s, "period:" + p),) for p in cols)
    body = tuple((((make_label(s, "row:" + r),),) + tuple((model_figure(s, r, p),) for p in cols)) for r in rows)
    return [Heading(_t("Summary"), 2), Table(header, body)]


def _against_estimate(s: Sources) -> list:
    out: list = [Heading(_t("Results against the estimate"), 2)]
    if not s.comparison or not s.comparison.get("lines"):
        return out + [Paragraph(_t("No results are set against an estimate yet: the comparison appears once the "
                                   "model is rolled forward on new results."), kind="note")]
    lines = s.comparison["lines"]
    header = (_t(""), (Text("Estimate, "), make_label(s, "comparison_estimate")),
              (Text("Reported, "), make_label(s, "comparison_period")), _t("Difference"), _t("Difference, share"))
    body = tuple(((make_label(s, "line:%d" % i),),) + tuple((comparison_figure(s, i, w),)
                                                           for w in ("estimate", "actual", "difference", "share"))
                 for i in range(len(lines)) if lines[i].get("kind") == "flow")
    return out + [Table(header, body)]


def changed_rows(s: Sources) -> list[tuple[str, str, str]]:
    """(row, this period, the earlier version's period with the same label) for every summary figure that moved."""
    if s.earlier is None:
        return []
    earlier_rows = _rows(s.earlier.model)
    by_label = {period_label(s.earlier.company, p): p for p in years(s.earlier.model)}
    out = []
    for p in [CY] + [q.name for q in s.model.periods if q.kind == "forecast"]:
        old_p = by_label.get(period_label(s.company, p))
        if old_p is None:
            continue
        for r in SUMMARY_ROWS:
            if r not in earlier_rows or r not in _rows(s.model):
                continue
            now, then = s.result.value(r, p), s.earlier.result.value(r, old_p)
            if now is not None and then is not None and not math.isclose(now, then, rel_tol=1e-9, abs_tol=1e-9):
                out.append((r, p, old_p))
    return out


def _estimates(s: Sources) -> list:
    out: list = [Heading(_t("Changes to estimates"), 2)]
    if s.earlier is None:
        return out + [Paragraph(_t("No earlier version is chosen to set the estimates against."), kind="note")]
    moved = changed_rows(s)
    if not moved:
        return out + [Paragraph((Text("No summary figure differs from version "), make_label(s, "version"),
                                 Text(".")), kind="note")]
    header = (_t(""), _t("Year"), (Text("Version "), make_label(s, "version")), _t("Now"))
    body = tuple(((make_label(s, "row:" + r),), (make_label(s, "period:" + p),),
                  (model_figure(s, r, old, "earlier"),), (model_figure(s, r, p),)) for r, p, old in moved)
    return out + [Table(header, body)]


def _charts(s: Sources, have: dict) -> list:
    cols = years(s.model)
    cats = tuple(make_label(s, "period:" + p) for p in cols)
    series = tuple(((make_label(s, "row:" + r),), tuple(model_figure(s, r, p) for p in cols))
                   for r in ("rev", "ebitda") if r in have)
    title = (Text("Revenue and EBITDA, "), make_label(s, "unit"))
    blocks: list = [Chart(title, cats, series)] if series else []
    if all(r in have for r in ("v_ps", "v_mps", "v_price")):
        names = tuple(make_label(s, "row:" + r) for r in ("v_ps", "v_mps", "v_price"))
        blocks.append(Chart(_t("Value per share and the price typed"), names,
                            ((_t("Per share"), tuple(model_figure(s, r, LY) for r in ("v_ps", "v_mps", "v_price"))),),
                            unit="rupees2"))
    return blocks


def _valuation(s: Sources, have: dict) -> list:
    rows = [(r, p) for r, p in VALUE_ROWS if r in have]
    if not rows:
        return []
    out: list = [Heading(_t("Valuation"), 2)]
    if VALUE_MULTIPLE in s.company.choices and s.company.choices[VALUE_MULTIPLE] in MULTIPLES:
        year = s.company.choices.get(VALUE_YEAR)
        line = [Text("The target multiple is "), make_label(s, "choice:" + VALUE_MULTIPLE)]
        if year:
            line += [Text(" on "), make_label(s, "period:" + year)]
        out.append(Paragraph(tuple(line) + (Text(", the analyst's choice in the model."),), kind="note"))
    body = tuple(((make_label(s, "row:" + r),), (model_figure(s, r, p),)) for r, p in rows)
    return out + [Table((_t(""), (make_label(s, "period:" + LY),)), body)]


def build(s: Sources) -> Draft:
    """The note for a saved model, checked."""
    have = {r.key for r in s.model.data_rows()}
    if "rev" not in have:
        raise DraftRefused(NO_MODEL_VALUES, "rev")
    from research.writer import slots_for

    slots = slots_for(s)
    f = s.fields
    blocks: list = [
        Paragraph((make_label(s, "firm"),) if s.details.firm else _t("Firm name not typed: add it in the report "
                                                                     "details."), kind="note"),
        Heading((make_label(s, "company_name"),), 1),
        Paragraph(((make_label(s, "basis"), Text(". ")) if s.company.basis else ()) +
                  (Text("Money in "), make_label(s, "unit"), Text(".")), kind="note"),
        Paragraph((Text("Note dated "), typed(s, "note_date")) if f.note_date else _t("Note date not typed."),
                  kind="note"),
        Heading(_t("The analyst's view"), 2),
        Paragraph((Text("Rating: "), typed(s, "rating")) if f.rating.strip()
                  else _t("Rating: not typed. The analyst picks it; the tool picks none."), kind="analyst"),
        Paragraph((Text("Target, the analyst's: "), typed(s, "target")) if f.target is not None
                  else _t("Target, the analyst's: not typed."), kind="analyst"),
    ]
    blocks += _summary(s, have)
    for section in ("outlook",):
        par = sentence(s, section, slots)
        if par:
            blocks.append(par)
    blocks += _against_estimate(s)
    par = sentence(s, "results", slots)
    if par and s.comparison:
        blocks.append(par)
    blocks += _estimates(s)
    par = sentence(s, "estimates", slots)
    if par and changed_rows(s):
        blocks.append(par)
    blocks += _charts(s, have)
    blocks += _valuation(s, have)
    par = sentence(s, "valuation", slots)
    if par:
        blocks.append(par)
    blocks.append(Heading(_t("The argument"), 2))
    if f.argument.strip():
        blocks.append(Paragraph((typed(s, "argument"),), kind="analyst"))
    else:
        blocks.append(Paragraph(_t("The analyst's argument goes here: not typed yet."), kind="note"))
    blocks.append(Heading(_t("Disclosures"), 2))
    if s.details.disclosures.strip():
        blocks.append(Paragraph((typed(s, "disclosures"),), kind="analyst"))
    else:
        blocks.append(Paragraph(_t("Disclosures not typed: add the firm's disclosures in the report details "
                                   "before this note goes to anyone."), kind="note"))
    blocks.append(Paragraph(_t("The figures come from the analyst's model; the view, the target and the argument "
                               "are the analyst's own."), kind="note"))
    return check(Draft(tuple(blocks), s.grouping), s)


# ---------------------------------------------------------------------------------------------- as data
def plain(parts) -> str:
    return "".join(seg.text for seg in parts)


def to_dict(draft: Draft) -> dict:
    """The draft for the screen's preview: blocks of plain text, figures marked."""
    def seg(x):
        kind = type(x).__name__.lower()
        return {"kind": kind, "text": x.text, **({"source": x.source} if isinstance(x, (Figure, Label)) else {})}

    def cell(parts):
        return [seg(x) for x in parts]

    out = []
    for b in draft.blocks:
        if isinstance(b, Heading):
            out.append({"type": "heading", "level": b.level, "parts": cell(b.parts)})
        elif isinstance(b, Paragraph):
            out.append({"type": "paragraph", "kind": b.kind, "parts": cell(b.parts)})
        elif isinstance(b, Table):
            out.append({"type": "table", "header": [cell(c) for c in b.header],
                        "rows": [[cell(c) for c in row] for row in b.rows]})
        else:
            out.append({"type": "chart", "title": cell(b.title), "unit": b.unit,
                        "categories": [x.text for x in b.categories],
                        "series": [{"name": plain(n), "values": [f.value for f in figs], "texts": [f.text for f in figs]}
                                   for n, figs in b.series]})
    return {"blocks": out}


__all__ = ["CALLS", "Chart", "Details", "Draft", "DraftRefused", "Earlier", "Fields", "Figure", "Heading", "Label",
           "NUMBER_WORDS", "Paragraph", "RATING_WORDS", "Sources", "Table", "Text", "Typed", "build", "changed_rows",
           "check", "grouping_for", "make_label", "model_figure", "plain", "segments", "show", "to_dict", "words_problem",
           "years", "ROY", "YTD"]
