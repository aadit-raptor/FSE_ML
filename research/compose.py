"""A sector model put together from its segments (PLAN.md R4): the last
changes an analyst makes for a company -- add a line of revenue, drop one,
swap the block a line is worked out by -- are changes to a **structure**, and
the model is built again from it, so no change can leave a formula naming a
row that is gone.

A **structure** names the model it started from, its segments in order (each
a key ``s1`` ... ``s8`` and a block), the segment whose count of places costs
per place per day follow (or none), and how many forecast years. A **spec**
is a model as the catalogue offers it: its words per sector, its default
structure and its notes.

Rules that keep every structure buildable: a share follows a line before it;
an analyst's own formula names only days, constants and the rows of lines
before it; at most one line is, for the past, revenue less the others; the
segment costs follow counts places. `normalise` mends what a drop or a swap
breaks (a share whose line is gone follows line 1; costs no longer per place
when the counting line is gone), so the screen never offers a change that
fails; `read` refuses anything else from outside.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from research import blocks, formula, lower
from research.definition import Heading, Model, ModelError, check
from research.formula import FormulaError
from research.pieces import UNIT, days_row, periods

SEGMENT_KEYS = tuple("s%d" % i for i in range(1, 9))
MAX_FORECAST_YEARS = 10
_SEGMENT_WORD = re.compile(r"(s[1-8])(?:_|$)")


@dataclass(frozen=True)
class Segment:
    key: str
    block: str
    of: str | None = None               # share: the line it is a share of
    rest_of_revenue: bool = False       # share: for the past, total revenue less the other lines
    formula: str | None = None          # share: the analyst's own forecast in place of the share

    def to_dict(self) -> dict:
        return {"key": self.key, "block": self.block, "of": self.of, "rest_of_revenue": self.rest_of_revenue,
                "formula": self.formula}


@dataclass(frozen=True)
class Structure:
    model: str
    segments: tuple
    places: str | None = None
    forecast_years: int = 2

    def segment(self, key: str) -> Segment:
        return next(s for s in self.segments if s.key == key)

    def to_dict(self) -> dict:
        return {"model": self.model, "segments": [s.to_dict() for s in self.segments], "places": self.places,
                "forecast_years": self.forecast_years}


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    about: str
    default: Structure
    sectors: dict                       # sector -> its words (segment words apply while the segment keeps its block)
    notes: tuple
    offered: bool = False               # listed for analysts only once two companies are solved by hand (rule 4)
    vetted: bool = False                # and read by a practising analyst


# ---------------------------------------------------------------------------
# Checking a structure
# ---------------------------------------------------------------------------
def _refs_allowed(seg: Segment, before: list) -> None:
    try:
        tree = formula.parse(seg.formula)
    except FormulaError as e:
        raise ModelError("the formula of line %s: %s" % (seg.key, e)) from None
    for ref in formula.refs(tree):
        owner = _SEGMENT_WORD.match(ref.name)
        if ref.name in ("days", UNIT) or (owner and owner.group(1) in before and ref.name.startswith(owner.group(1) + "_")):
            continue
        raise ModelError("the formula of line %s may name only days, %s and the rows of lines before it, not %r"
                         % (seg.key, UNIT, ref.name))


def validate(structure: Structure, specs: dict) -> None:
    """Refuse a structure that would not build, saying why."""
    if structure.model not in specs:
        raise ModelError("there is no model called %r" % structure.model)
    if not 1 <= len(structure.segments) <= len(SEGMENT_KEYS):
        raise ModelError("a model has between 1 and %d lines of revenue" % len(SEGMENT_KEYS))
    if not isinstance(structure.forecast_years, int) or not 1 <= structure.forecast_years <= MAX_FORECAST_YEARS:
        raise ModelError("forecast years run from 1 to %d" % MAX_FORECAST_YEARS)
    seen: list = []
    for seg in structure.segments:
        if seg.key not in SEGMENT_KEYS or seg.key in seen:
            raise ModelError("line %r: lines are keyed s1 to s8, each once" % seg.key)
        if seg.block not in blocks.BLOCKS:
            raise ModelError("line %s: there is no block called %r" % (seg.key, seg.block))
        if seg.block == "share":
            if seg.formula is not None:
                if not isinstance(seg.formula, str):
                    raise ModelError("the formula of line %s must be text" % seg.key)
                _refs_allowed(seg, seen)
            elif seg.of not in seen:
                raise ModelError("line %s is a share of a line before it, not of %r" % (seg.key, seg.of))
        elif seg.of is not None or seg.formula is not None or seg.rest_of_revenue:
            raise ModelError("line %s: only a share line follows another line or has its own formula" % seg.key)
        seen.append(seg.key)
    if sum(bool(s.rest_of_revenue) for s in structure.segments) > 1:
        raise ModelError("at most one line is, for the past, revenue less the others")
    if structure.places is not None and (structure.places not in seen
                                         or structure.segment(structure.places).block not in blocks.COUNTING):
        raise ModelError("costs per place follow a line that counts places, not %r" % structure.places)


def normalise(structure: Structure) -> Structure:
    """Mend what dropping or swapping a line broke, the way the screen would."""
    keys = [s.key for s in structure.segments]
    mended = []
    for i, seg in enumerate(structure.segments):
        if seg.block != "share":
            seg = replace(seg, of=None, formula=None, rest_of_revenue=False)
        elif seg.formula is None and seg.of not in keys[:i]:
            seg = replace(seg, of=keys[0] if i else None)
            if i == 0:                  # line 1 cannot be a share of anything: it grows on last year instead
                seg = replace(seg, block="growth", rest_of_revenue=False)
        mended.append(seg)
    places = structure.places
    counting = [s.key for s in mended if s.block in blocks.COUNTING]
    if places not in counting:
        places = counting[0] if counting and places is not None else None
    return replace(structure, segments=tuple(mended), places=places)


# ---------------------------------------------------------------------------
# The analyst's changes
# ---------------------------------------------------------------------------
def add_segment(structure: Structure, block: str, of: str | None = None) -> Structure:
    free = [k for k in SEGMENT_KEYS if k not in {s.key for s in structure.segments}]
    if not free:
        raise ModelError("a model has at most %d lines of revenue" % len(SEGMENT_KEYS))
    seg = Segment(free[0], block, of=(of or structure.segments[0].key) if block == "share" else None)
    return normalise(replace(structure, segments=structure.segments + (seg,)))


def drop_segment(structure: Structure, key: str) -> Structure:
    if len(structure.segments) == 1:
        raise ModelError("a model keeps at least one line of revenue")
    return normalise(replace(structure, segments=tuple(s for s in structure.segments if s.key != key)))


def swap_block(structure: Structure, key: str, block: str) -> Structure:
    segments = tuple(replace(s, block=block, formula=None) if s.key == key else s for s in structure.segments)
    return normalise(replace(structure, segments=segments))


# ---------------------------------------------------------------------------
# Building the model
# ---------------------------------------------------------------------------
def _words(spec: Spec, structure: Structure) -> dict:
    """Each sector's words for this structure: a segment's own words only while it keeps the default's block."""
    default = {s.key: s.block for s in spec.default.segments}
    present = {s.key: s.block for s in structure.segments}
    out = {}
    for sector, own in spec.sectors.items():
        words = {}
        for n, seg in enumerate(structure.segments, start=1):
            words.update(blocks.words(seg.block, seg.key, n))
        for word, text in own.items():
            owner = _SEGMENT_WORD.match(word)
            if owner is None or (owner.group(1) in present and present[owner.group(1)] == default.get(owner.group(1))):
                words[word] = text
        out[sector] = words
    return out


def changes(spec: Spec, structure: Structure) -> list:
    """What the analyst changed from the model as it is offered, in words for the Notes sheet."""
    if structure.segments == spec.default.segments and structure.places == spec.default.places:
        return []
    out = ["Changed for this company from the %s model, whose lines of revenue were %s: these lines are not "
           "among the companies the model was solved by hand for." % (
               spec.name, ", ".join("%s (%s)" % (s.key, s.block) for s in spec.default.segments))]
    out.append("Lines now: %s." % ", ".join("%s (%s)" % (s.key, s.block) for s in structure.segments))
    out.append("Costs per place per day follow %s." % (structure.places or "no line: costs are a share of revenue only"))
    return out


def build(spec: Spec, structure: Structure, specs: dict | None = None) -> Model:
    """The model a structure describes, checked."""
    validate(structure, specs or {spec.id: spec})
    revenue_rows, keys = [], [s.key for s in structure.segments]
    rows: list = [Heading("PERIOD"), days_row()]
    for n, seg in enumerate(structure.segments, start=1):
        others = tuple(k for k in keys if k != seg.key) if seg.rest_of_revenue else ()
        rows += blocks.rows(seg.block, seg.key, n, of=seg.of, rest_of_revenue=others, formula=seg.formula)
        revenue_rows.append(seg.key + "_rev")
    places = structure.places + "_avg" if structure.places else None
    rows += lower.rows(revenue_rows, places)
    model = Model(
        id=spec.id, name=spec.name, periods=periods(structure.forecast_years),
        constants={UNIT: "{currency} in one unit of the statements (prices per day are in {currency})"},
        rows=tuple(rows), sectors=_words(spec, structure), notes=tuple(spec.notes) + tuple(changes(spec, structure)),
        vetted=spec.vetted)
    check(model)
    return model


def read(data, specs: dict) -> Structure:
    """A structure from outside (the screen, the add-in): anything misshapen is a ModelError."""
    try:
        segments = tuple(Segment(str(s["key"]), str(s["block"]), s.get("of"), bool(s.get("rest_of_revenue", False)),
                                 s.get("formula")) for s in data["segments"])
        structure = Structure(str(data["model"]), segments, data.get("places"), data.get("forecast_years", 2))
    except (KeyError, TypeError, AttributeError) as e:
        raise ModelError("the structure is not in the format: %s" % e) from None
    validate(structure, specs)
    return structure


@dataclass(frozen=True)
class Catalogue:
    specs: dict = field(default_factory=dict)

    def spec(self, model_id: str) -> Spec:
        if model_id not in self.specs:
            raise ModelError("there is no model called %r" % model_id)
        return self.specs[model_id]

    def build(self, structure: Structure) -> Model:
        return build(self.spec(structure.model), structure, self.specs)

    def sectors(self) -> dict:
        """sector -> the model that serves it."""
        return {sector: spec.id for spec in self.specs.values() for sector in spec.sectors}


def fit(model: Model, company):
    """The company with only what this model takes, after a change to its lines: figures for rows or cells
    the model now works out or no longer has, choices it no longer asks, and segments it no longer has are
    left out, and a choice a new line asks for takes its first option. Answers the company and what was left
    out, as (row, period) pairs."""
    from research.definition import Pick, Typed, rule_for

    rows = {r.key: r for r in model.data_rows()}
    periods = {p.name: p for p in model.periods}
    choices = model.choices()
    picked = {k: v for k, v in company.choices.items() if k in choices}       # a wrong answer is still refused
    for choice in choices:
        if choice not in picked:
            first = next(rule.options for r in rows.values() for rule in r.rules.values()
                         if isinstance(rule, Pick) and rule.choice == choice)
            picked[choice] = next(iter(first))
    dropped, values, marks = [], {}, {}
    for key, by_period in company.values.items():
        for period, value in by_period.items():
            row, at = rows.get(key), periods.get(period)
            rule = rule_for(row, at) if row is not None and at is not None else None
            if isinstance(rule, Pick):
                rule = rule.options.get(picked[rule.choice])
            if isinstance(rule, Typed):
                values.setdefault(key, {})[period] = value
            elif value is not None:
                dropped.append((key, period))
    for key, by_period in company.marks.items():
        kept = {p: m for p, m in by_period.items() if p in values.get(key, {})}
        if kept:
            marks[key] = kept
    off = tuple(s for s in company.off if s in model.segments())
    return replace(company, choices=picked, values=values, marks=marks, off=off), dropped
