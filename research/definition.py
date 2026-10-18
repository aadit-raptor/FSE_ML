"""A sector model as data (PLAN.md R4): rows down, periods across, a rule in
each cell.

**A rule** is one of: `FIGURE` (a number typed from the documents, which may
be missing), `ASSUME` (the analyst's assumption, which must be given), a
formula (`research/formula.py`), a `Pick` between rules that the company's
file settles, or None for an empty cell. A row gives its rules by period name,
by kind of period, or for a group of kinds; the narrowest one wins.

**A company** is only data too: its words, its unit, its choices and the
numbers typed for it. `cells` puts the two together into what stands in each
cell, which is all the evaluator and the workbook writer read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from research import formula
from research.formula import FormulaError, Node

KINDS = ("past_year", "to_date", "rest", "current", "forecast")
# A rule's key may be a period's name, its kind, or one of these groups (narrowest first)
GROUPS = (("past", ("past_year", "to_date")),
          ("year_end", ("past_year", "to_date", "current", "forecast")),      # every period with a balance sheet
          ("all", KINDS))
UNITS = ("money", "count", "count3", "days", "share", "rupees", "rupees2", "times", "check")
MARKS = ("amber", "red")
_NAME = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_SLOT = re.compile(r"\{([a-z][a-z0-9_]*)\}", re.ASCII)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_TEXT = 200              # characters in a name, a word or a period's label
MAX_REASON = 1000           # characters in the reason for a mark
MAX_NUMBER = 1e15           # no typed figure is larger: Excel keeps fifteen digits
WIDEST_ADDRESS = "$XFD$1048576"
RESERVED = set(formula.FUNCTIONS) | {"prev"} | set(KINDS) | {g for g, _ in GROUPS}


class ModelError(ValueError):
    """The model definition itself is wrong."""


class InputError(ValueError):
    """What was typed for a company does not fit the model."""


class Typed(Enum):
    FIGURE = "figure"           # from the documents; blank is allowed and reads as nil
    ASSUME = "assumption"       # the analyst's; must be given; yellow and named in the workbook


FIGURE, ASSUME = Typed.FIGURE, Typed.ASSUME


@dataclass(frozen=True)
class Pick:
    """A rule the company's file settles: `choice` names the question, `options` the answers."""
    choice: str
    options: dict


Rule = Union[None, str, Typed, Pick]


@dataclass(frozen=True)
class Period:
    name: str
    kind: str
    prev: str | None = None     # the period a balance rolls from


@dataclass(frozen=True)
class Row:
    key: str
    label: str                  # may hold {word} slots the company's words fill
    unit: str
    rules: dict
    bold: bool = False
    segment: str | None = None  # a company may switch a segment off: its typed cells then read nil


@dataclass(frozen=True)
class Heading:
    label: str


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    periods: tuple
    constants: dict             # name -> label: single cells, the same in every period
    rows: tuple
    sectors: dict = field(default_factory=dict)     # sector -> its words; empty when the model has no slots
    notes: tuple = ()
    vetted: bool = False        # True only once a practising analyst has read it (PLAN.md phase R, rule 4)

    def data_rows(self) -> list[Row]:
        return [r for r in self.rows if isinstance(r, Row)]

    def period(self, name: str) -> Period:
        return next(p for p in self.periods if p.name == name)

    def choices(self) -> dict:
        """Every choice a company must make, with its options."""
        out: dict = {}
        for row in self.data_rows():
            for rule in row.rules.values():
                if isinstance(rule, Pick):
                    out.setdefault(rule.choice, set()).update(rule.options)
        return out

    def segments(self) -> set:
        return {r.segment for r in self.data_rows() if r.segment}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "vetted": self.vetted, "notes": list(self.notes),
            "periods": [{"name": p.name, "kind": p.kind, "prev": p.prev} for p in self.periods],
            "constants": dict(self.constants), "sectors": {s: dict(w) for s, w in self.sectors.items()},
            "rows": [{"heading": r.label} if isinstance(r, Heading) else
                     {"key": r.key, "label": r.label, "unit": r.unit, "bold": r.bold, "segment": r.segment,
                      "rules": {k: _rule_out(v) for k, v in r.rules.items()}} for r in self.rows],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        rows = tuple(Heading(r["heading"]) if "heading" in r else
                     Row(r["key"], r["label"], r["unit"], {k: _rule_in(v) for k, v in r["rules"].items()},
                         bool(r.get("bold")), r.get("segment")) for r in data["rows"])
        model = cls(id=data["id"], name=data["name"],
                    periods=tuple(Period(p["name"], p["kind"], p.get("prev")) for p in data["periods"]),
                    constants=dict(data.get("constants", {})), rows=rows, sectors=dict(data.get("sectors", {})),
                    notes=tuple(data.get("notes", ())), vetted=bool(data.get("vetted")))
        check(model)
        return model

    @classmethod
    def read(cls, data) -> "Model":
        """`from_dict` for data from outside: anything misshapen is a ModelError, never a crash."""
        try:
            return cls.from_dict(data)
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            if isinstance(e, ModelError):
                raise
            raise ModelError("the model is not in the format: %s %s" % (type(e).__name__, e)) from None


def _rule_out(rule: Rule):
    if isinstance(rule, Typed):
        return {"typed": rule.value}
    if isinstance(rule, Pick):
        return {"choice": rule.choice, "options": {k: _rule_out(v) for k, v in rule.options.items()}}
    return rule


def _rule_in(rule) -> Rule:
    if isinstance(rule, dict) and "typed" in rule:
        return Typed(rule["typed"])
    if isinstance(rule, dict) and "choice" in rule:
        return Pick(rule["choice"], {k: _rule_in(v) for k, v in rule["options"].items()})
    if rule is None or isinstance(rule, str):
        return rule
    raise ModelError("a rule that is not typed, a choice, a formula or empty: %r" % (rule,))


@dataclass
class Company:
    """What is typed for one company: words, unit, choices and numbers. No formulas."""
    name: str
    unit_name: str                                   # "Rs lakh": what the statements' money is in
    sector: str = ""
    basis: str = ""                                  # "Consolidated, Ind AS"
    words: dict = field(default_factory=dict)        # overrides the sector's own
    period_labels: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    choices: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)       # row -> period -> number
    off: tuple = ()                                  # segments switched off
    marks: dict = field(default_factory=dict)        # row -> period -> {"level": amber|red, "why": ...}


@dataclass(frozen=True)
class Cell:
    """What stands in one cell: a formula's tree, or a typed number (None = left blank)."""
    tree: Node | None = None
    typed: Typed | None = None
    value: float | None = None


# ---------------------------------------------------------------------------
# Checking a definition
# ---------------------------------------------------------------------------
def rule_for(row: Row, period: Period) -> Rule:
    """The narrowest rule a row gives for a period: its name, its kind, then the groups."""
    for key in (period.name, period.kind):
        if key in row.rules:
            return row.rules[key]
    for group, kinds in GROUPS:
        if group in row.rules and period.kind in kinds:
            return row.rules[group]
    return None


def _options(rule: Rule) -> list:
    return list(rule.options.values()) if isinstance(rule, Pick) else [rule]


def check(model: Model) -> None:
    """Refuse a definition that could not be worked out, saying what is wrong with it."""
    names = [p.name for p in model.periods]
    for p in model.periods:
        if not _NAME.fullmatch(p.name) or p.name in RESERVED:
            raise ModelError("period name %r is not a plain lower-case name" % p.name)
        if names.count(p.name) > 1:
            raise ModelError("period %r is there twice" % p.name)
        if p.kind not in KINDS:
            raise ModelError("period %r has the unknown kind %r" % (p.name, p.kind))
        if p.prev is not None and (p.prev not in names or p.prev == p.name):
            raise ModelError("period %r rolls from %r, which is not another period" % (p.name, p.prev))
    rows = model.data_rows()
    keys = [r.key for r in rows]
    for name in list(model.constants) + keys:
        if not _NAME.fullmatch(name) or name in RESERVED or name in names:
            raise ModelError("%r cannot name a row or a constant (a plain lower-case name that is not a period or a reserved word)" % name)
    for key in keys:
        if keys.count(key) > 1 or key in model.constants:
            raise ModelError("the name %r is used twice" % key)
    allowed = set(names) | set(KINDS) | {g for g, _ in GROUPS}
    options: dict = {}
    for row in rows:
        if row.unit not in UNITS:
            raise ModelError("row %r has the unknown unit %r" % (row.key, row.unit))
        for where, rule in row.rules.items():
            if where not in allowed:
                raise ModelError("row %r has a rule for %r, which is no period, kind or group" % (row.key, where))
            if isinstance(rule, Pick):
                if any(isinstance(o, Pick) for o in rule.options.values()) or not rule.options:
                    raise ModelError("the choice %r in row %r must offer plain rules" % (rule.choice, row.key))
                if options.setdefault(rule.choice, set(rule.options)) != set(rule.options):
                    raise ModelError("the choice %r offers different options in different rows (row %r)" % (rule.choice, row.key))
        for period in model.periods:
            for rule in _options(rule_for(row, period)):
                if isinstance(rule, str):
                    _check_formula(model, row, period, rule, set(keys), set(names))
                elif rule is not None and not isinstance(rule, Typed):
                    raise ModelError("row %r has a rule that is not typed, a choice, a formula or empty" % row.key)
    for sector, words in model.sectors.items():
        for item in list(model.constants.values()) + [r.label for r in model.rows]:
            for slot in _SLOT.findall(item):
                if slot not in words:
                    raise ModelError("sector %r has no word for %r in %r" % (sector, slot, item))


def _check_formula(model: Model, row: Row, period: Period, text: str, keys: set, names: set) -> None:
    try:
        tree = formula.parse(text)
        formula.to_excel(tree, lambda ref: WIDEST_ADDRESS)       # it must also fit in an Excel cell
    except FormulaError as e:
        raise ModelError("row %r: %s" % (row.key, e)) from None
    for ref in formula.refs(tree):
        if ref.name in model.constants:
            if ref.period is not None:
                raise ModelError("row %r: the constant %r has no periods" % (row.key, ref.name))
        elif ref.name not in keys:
            raise ModelError("row %r names %r, which is no row or constant" % (row.key, ref.name))
        elif ref.period == "prev":
            if period.prev is None:
                raise ModelError("row %r asks for %s[prev] in %r, which has no period before it" % (row.key, ref.name, period.name))
        elif ref.period is not None and ref.period not in names:
            raise ModelError("row %r names the unknown period %r" % (row.key, ref.period))


# ---------------------------------------------------------------------------
# A model and a company together
# ---------------------------------------------------------------------------
def words_for(model: Model, company: Company) -> dict:
    """The words that fill the labels' slots: the sector's, then the company's own."""
    if not isinstance(company.sector, str) or (model.sectors and company.sector not in model.sectors):
        raise InputError("sector %r is not one of this model's: %s" % (company.sector, ", ".join(sorted(model.sectors))))
    return {**model.sectors.get(company.sector, {}), **company.words}


def label(text: str, words: dict) -> str:
    """A label with its {word} slots filled. Only whole words go in: no formatting, no attributes."""
    def fill(match) -> str:
        if match.group(1) not in words:
            raise InputError("no word for %r in the label %r" % (match.group(1), text))
        return words[match.group(1)]
    return _SLOT.sub(fill, text)


def _number(row: str, period: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) > MAX_NUMBER:
        raise InputError("row %r in %r must be a number no larger than %g, not %r" % (row, period, MAX_NUMBER, str(v)[:40]))
    return float(v)


def _text(what: str, v, longest: int = MAX_TEXT) -> None:
    if not isinstance(v, str) or len(v) > longest or _CONTROL.search(v):
        raise InputError("%s must be plain text of at most %d characters" % (what, longest))


def _mapping(what: str, v) -> None:
    if not isinstance(v, dict) or not all(isinstance(k, str) for k in v):
        raise InputError("%s must be a table of names" % what)


def check_company(model: Model, company: Company) -> None:
    """Refuse a company whose shape is wrong, before anything reads it: every part is typed by someone."""
    _text("the company's name", company.name)
    _text("the unit's name", company.unit_name)
    _text("the basis", company.basis)
    for what, table in (("words", company.words), ("period labels", company.period_labels), ("constants", company.constants),
                        ("choices", company.choices), ("values", company.values), ("marks", company.marks)):
        _mapping("the " + what, table)
    for word, text in company.words.items():
        _text("the word %r" % word, text)
    names = {p.name for p in model.periods}
    for period, text in company.period_labels.items():
        if period not in names:
            raise InputError("a label is given for the unknown period %r" % period)
        _text("the label of %r" % period, text)
    for choice, picked in company.choices.items():
        _text("the choice %r" % choice, picked)
    if not isinstance(company.off, (tuple, list)) or not all(isinstance(s, str) for s in company.off):
        raise InputError("the segments switched off must be a list of names")
    rows = {r.key for r in model.data_rows()}
    for what, table in (("figure", company.values), ("mark", company.marks)):
        for key, by_period in table.items():
            if key not in rows:
                raise InputError("this model has no row called %r (a %s is given for it)" % (key, what))
            _mapping("the %ss of row %r" % (what, key), by_period)
            for period in by_period:
                if period not in names:
                    raise InputError("row %r has a %s for the unknown period %r" % (key, what, period))
    for key, by_period in company.marks.items():
        for period, mark in by_period.items():
            if not isinstance(mark, dict) or mark.get("level") not in MARKS:
                raise InputError("the mark on row %r in %r must have a level, amber or red" % (key, period))
            _text("the reason for the mark on row %r" % key, mark.get("why", ""), MAX_REASON)


def cells(model: Model, company: Company, allow_missing: bool = False) -> dict:
    """{(row, period): Cell} for every cell that is not empty.

    Refuses a company that leaves a choice or a constant out, misses an
    assumption (unless `allow_missing`: the cell is then an assumption with no
    value, as a blank yellow cell in a workbook not yet filled in), or types a
    figure where the model works one out.
    """
    check(model)
    check_company(model, company)
    choices = model.choices()
    for choice, picked in company.choices.items():
        if choice not in choices:
            raise InputError("this model has no choice called %r" % choice)
        if picked not in choices[choice]:
            raise InputError("the choice %r is one of %s, not %r" % (choice, ", ".join(sorted(choices[choice])), picked))
    for choice in choices:
        if choice not in company.choices:
            raise InputError("the choice %r is not made (%s)" % (choice, ", ".join(sorted(choices[choice]))))
    for name in model.constants:
        if name not in company.constants:
            raise InputError("the constant %r is not given" % name)
        _number(name, "every period", company.constants[name])
    for name in company.constants:
        if name not in model.constants:
            raise InputError("this model has no constant called %r" % name)
    for segment in company.off:
        if segment not in model.segments():
            raise InputError("this model has no segment called %r" % segment)
    words_for(model, company)

    rows = {r.key: r for r in model.data_rows()}
    out, missing = {}, []
    for row in rows.values():
        given = company.values.get(row.key, {})
        for period in model.periods:
            rule = rule_for(row, period)
            if isinstance(rule, Pick):
                rule = rule.options[company.choices[rule.choice]]
            if isinstance(rule, Typed):
                value = given.get(period.name)
                if value is not None:
                    value = _number(row.key, period.name, value)
                elif row.segment in company.off:
                    value = 0.0
                elif rule is ASSUME:
                    missing.append("%s[%s]" % (row.key, period.name))
                out[(row.key, period.name)] = Cell(typed=rule, value=value)
                continue
            if given.get(period.name) is not None:
                raise InputError("row %r is %s in %r: it takes no typed figure" % (
                    row.key, "empty" if rule is None else "worked out by the model", period.name))
            if rule is not None:
                out[(row.key, period.name)] = Cell(tree=formula.parse(rule))
    if missing and not allow_missing:
        raise InputError("assumptions not given: " + ", ".join(missing))
    return out


def missing(plan: dict) -> list:
    """The assumption cells of a plan with nothing typed in them, as (row, period)."""
    return [key for key, cell in plan.items() if cell.typed is ASSUME and cell.value is None]
