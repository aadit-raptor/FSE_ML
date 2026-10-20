"""The linking sentences of a results note (PLAN.md R8; 6.3's writer and its rule).

**The AI writes words, the app writes numbers.** Each section of the note
(``SECTIONS``) has slots: a figure of the model (``{rev_cy}``) or a label
(``{cy}``, the current year's name). The AI is sent each slot's name and a
plain meaning ("revenue, current year"), and the directions the code has
worked out ("revenue is higher in the current year than last year"); never
the company, its sector's words, a figure or anything from a document. So
on the free tier (``AI_FREE_TIER``) it receives only anonymised,
non-identifying text, and every figure is put in afterwards by the app.

**Its answer is checked** (``checked``): a section may name only its own
slots, hold no digit, no number word, no rating and no word of a call
(research/report.py ``words_problem``), and stay short. A section that fails,
is missing, or comes back while the AI is off, spent or busy is written by
the section's template instead (``template``), and the note says which.
The draft is checked again as a whole when it is built (``report.check``).
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from research.pieces import CY, LY
from research.report import (
    Sources, Text, changed_rows, comparison_figure, make_label, model_figure, words_problem,
)

SECTIONS = ("outlook", "results", "estimates", "valuation")
PROMPT_VERSION = "r8-1"
MAX_SENTENCE = 600              # characters a section's text may hold
SLOT = re.compile(r"\{([a-z][a-z0-9_]*)\}")

# What a slot means, in words that name no company (the row keys are the model format's own)
MEANS = {"rev": "revenue", "ebitda": "EBITDA", "ebitda_m": "EBITDA margin", "pat": "profit after tax",
         "pat_own": "profit for the owners", "eps": "earnings per share"}
RESULT_ROWS = ("rev", "ebitda", "pat", "pat_own")

# Why the template wrote a section (codes the screen words: report.source_*)
AI = "ai"
TEMPLATE = "template"
AI_OFF = "ai_off"
AI_FAILED = "ai_failed"
AI_REFUSED = "ai_refused"       # the AI's text broke a rule: a number, a call, an unknown slot


class SlotError(ValueError):
    pass


def parse(text: str, slots: dict) -> list:
    """`text` as segments: the words between slots as Text, each slot as the figure or label it stands for."""
    out, at = [], 0
    for m in SLOT.finditer(text):
        if m.start() > at:
            out.append(Text(text[at:m.start()]))
        if m.group(1) not in slots:
            raise SlotError(m.group(1))
        out.append(slots[m.group(1)])
        at = m.end()
    if at < len(text):
        out.append(Text(text[at:]))
    return out


def _higher(a: Optional[float], b: Optional[float]) -> Optional[str]:
    if a is None or b is None or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return "higher" if a > b else "lower"


def _forecast(s: Sources) -> Optional[str]:
    return next((p.name for p in s.model.periods if p.kind == "forecast"), None)


def facts(s: Sources) -> dict:
    """Per section: its slots (name -> (segment, meaning)) and the directions the code worked out."""
    have = {r.key for r in s.model.data_rows()}
    y1 = _forecast(s)
    out: dict = {}

    def fig(row, period, means):
        return model_figure(s, row, period), means

    # outlook: last year, this year and the first forecast year
    slots = {"ly": (make_label(s, "period:" + LY), "the name of last year"),
             "cy": (make_label(s, "period:" + CY), "the name of the current year")}
    directions = []
    for row in ("rev", "ebitda_m"):
        if row not in have:
            continue
        slots["%s_ly" % row] = fig(row, LY, "%s, last year" % MEANS[row])
        slots["%s_cy" % row] = fig(row, CY, "%s, current year" % MEANS[row])
        word = _higher(s.result.value(row, CY), s.result.value(row, LY))
        if word:
            directions.append("%s is %s in the current year than last year" % (MEANS[row], word))
        if y1:
            slots["%s_y1" % row] = fig(row, y1, "%s, first forecast year" % MEANS[row])
            word = _higher(s.result.value(row, y1), s.result.value(row, CY))
            if word:
                directions.append("%s is %s in the first forecast year than the current year" % (MEANS[row], word))
    if y1:
        slots["y1"] = (make_label(s, "period:" + y1), "the name of the first forecast year")
    out["outlook"] = (slots, directions)

    # results against the estimate, from the roll forward's stored comparison
    if s.comparison and s.comparison.get("lines"):
        slots = {"reported_period": (make_label(s, "comparison_period"), "the name of the period reported")}
        directions = []
        for i, line in enumerate(s.comparison["lines"]):
            row = line.get("row")
            if row not in RESULT_ROWS or line.get("kind") != "flow":
                continue
            slots["%s_reported" % row] = (comparison_figure(s, i, "actual"), "%s reported" % MEANS[row])
            slots["%s_estimate" % row] = (comparison_figure(s, i, "estimate"), "%s estimated" % MEANS[row])
            slots["%s_gap" % row] = (comparison_figure(s, i, "share"),
                                    "%s, the difference as a share of the estimate" % MEANS[row])
            word = _higher(line.get("actual"), line.get("estimate"))
            directions.append("%s reported is %s than estimated" % (MEANS[row], word) if word
                              else "%s reported is the same as estimated" % MEANS[row])
        out["results"] = (slots, directions)

    # changes to estimates against the version chosen
    moved = changed_rows(s)[:6]
    if moved:
        slots = {"version": (make_label(s, "version"), "the number of the earlier version")}
        directions = []
        for row, p, old in moved:
            base = "%s_%s" % (row, p)
            slots[base + "_before"] = (model_figure(s, row, old, "earlier"),
                                      "%s for that year in the earlier version" % MEANS[row])
            slots[base + "_now"] = (model_figure(s, row, p), "%s for that year now" % MEANS[row])
            slots[base + "_year"] = (make_label(s, "period:" + p), "the name of that year")
            word = _higher(s.result.value(row, p), s.earlier.result.value(row, old))
            directions.append("%s for {%s} is %s now than in the earlier version" % (MEANS[row], base + "_year", word))
        out["estimates"] = (slots, directions)

    # valuation
    if {"v_ps", "v_mps", "v_price", "v_date"} <= have:
        slots = {"dcf_value": fig("v_ps", LY, "value per share by the discounted cash flow"),
                 "multiple_value": fig("v_mps", LY, "value per share at the target multiple"),
                 "price": fig("v_price", LY, "the share price typed"),
                 "price_date": fig("v_date", LY, "the date of that price"),
                 "discount_rate": fig("v_rate", LY, "the discount rate"),
                 "growth": fig("v_g", LY, "growth after the forecast")}
        directions = []
        for name, row in (("discounted cash flow value", "v_ps"), ("target multiple value", "v_mps")):
            word = _higher(s.result.value(row, LY), s.result.value("v_price", LY))
            if word:
                directions.append("the %s is %s than the price typed" % (name, word))
        out["valuation"] = (slots, directions)
    return out


def slots_for(s: Sources, known: Optional[dict] = None) -> dict:
    """Every section's slots in one map (their names do not collide), for the draft builder."""
    out: dict = {}
    for slots, _ in (facts(s) if known is None else known).values():
        out.update({k: v[0] for k, v in slots.items()})
    return out


# ---------------------------------------------------------------------------------------------- templates
def template(section: str, s: Sources, known: Optional[dict] = None) -> Optional[str]:
    """The section's sentences when the AI does not write them: plain, and from the same slots. `known` is
    ``facts(s)`` when the caller has it already."""
    found = (facts(s) if known is None else known).get(section)
    if not found:
        return None
    slots, _ = found
    if section == "outlook":
        parts = []
        if "rev_cy" in slots:
            parts.append("Revenue was {rev_ly} in {ly}; the model has {rev_cy} for {cy}"
                         + (" and {rev_y1} for {y1}." if "rev_y1" in slots else "."))
        if "ebitda_m_cy" in slots:
            parts.append("The EBITDA margin is {ebitda_m_cy} for {cy}" +
                         (", against {ebitda_m_y1} for {y1}." if "ebitda_m_y1" in slots else "."))
        return " ".join(parts) or None
    if section == "results":
        parts = []
        for row in RESULT_ROWS:
            if "%s_reported" % row in slots:
                parts.append("%s was {%s_reported} against an estimate of {%s_estimate}, a difference of {%s_gap}."
                             % (MEANS[row][0].upper() + MEANS[row][1:], row, row, row))
        return ("For {reported_period}: " + " ".join(parts)) if parts else None
    if section == "estimates":
        parts = []
        for name in slots:
            if name.endswith("_now"):
                base = name[:-4]
                row = next(r for r in MEANS if base.startswith(r + "_") and base[len(r) + 1:] in
                           [p.name for p in s.model.periods])
                parts.append("%s for {%s_year} is now {%s_now}, from {%s_before} in version {version}."
                             % (MEANS[row][0].upper() + MEANS[row][1:], base, base, base))
        return " ".join(parts) or None
    if section == "valuation":
        return ("The discounted cash flow gives {dcf_value} a share at a discount rate of {discount_rate} and growth "
                "of {growth} after the forecast; the target multiple gives {multiple_value}. The price typed is "
                "{price}, on {price_date}.")
    return None


# ---------------------------------------------------------------------------------------------- the AI
def prompt(s: Sources) -> Optional[str]:
    """What the AI is sent: rules, then each section's slots and directions. No company, no figure."""
    sections = []
    for section, (slots, directions) in facts(s).items():
        sections.append({"section": section,
                         "slots": [{"slot": name, "means": means} for name, (_, means) in slots.items()],
                         "directions": directions})
    if not sections:
        return None
    rules = (
        "You write the linking sentences of an equity research results note. For each section below write one "
        "or two short sentences in plain English that use the section's slots, written exactly as {slot}; the "
        "application puts each figure or name in afterwards. Use only the slots listed for that section. Follow "
        "the directions; never guess one. Never write a digit, a number in words, a percentage, a rating (such "
        "as buy, sell or hold), a recommendation, a target price, upside or downside, or any view of whether the "
        "shares are worth owning. Do not mention a company. Answer with one JSON object whose keys are the "
        "section names and whose values are the sentences, and nothing else.")
    return rules + "\n\nSections:\n" + json.dumps(sections, indent=1, sort_keys=True)


def checked(section: str, text, s: Sources, known: Optional[dict] = None) -> Optional[str]:
    """The AI's text for a section if it keeps every rule, else None."""
    found = (facts(s) if known is None else known).get(section)
    if not found or not isinstance(text, str) or not text.strip() or len(text) > MAX_SENTENCE:
        return None
    slots, _ = found
    names = SLOT.findall(text)
    if not names or any(n not in slots for n in names):
        return None
    bare = SLOT.sub(" ", text)
    if "{" in bare or "}" in bare or words_problem(bare, strict=True):
        return None
    return " ".join(text.split())


def write(s: Sources, provider=None) -> dict:
    """Every section's sentences: the AI's when it keeps the rules, else the template's, saying why.
    Answers ``{section: {"text", "source"}}`` plus ``_model`` (the AI's model, if any)."""
    from research.reader.providers import ProviderError

    known = facts(s)
    sections = {name: template(name, s, known) for name in SECTIONS}
    out: dict = {}
    answer, reason, model = None, AI_OFF, None
    text = prompt(s)
    if provider is not None and text is not None:
        try:
            got = provider.write(text)
            answer, model = got.read, got.model
        except ProviderError:
            reason = AI_FAILED
    for name, fallback in sections.items():
        if fallback is None:
            continue
        mine = checked(name, answer.get(name), s, known) if isinstance(answer, dict) else None
        if mine is not None:        # kept with the directions it was written from (`stored`)
            out[name] = {"text": mine, "source": AI, "directions": known[name][1]}
        else:
            out[name] = {"text": fallback, "source": reason if answer is None else AI_REFUSED}
    return {"sections": out, "model": model, "prompt_version": PROMPT_VERSION}


def stored(saved: Optional[dict], s: Sources, known: Optional[dict] = None) -> dict:
    """Each section's text and who wrote it: the AI's stored sentence while it still fits the model (its slots, and
    the directions it was written from: a sentence that said "higher" must not stand beside figures now lower),
    else the template's. Answers ``{section: (text, source)}``."""
    known = facts(s) if known is None else known
    out = {}
    sections = (saved or {}).get("sections") or {}
    for name in SECTIONS:
        fallback = template(name, s, known)
        if fallback is None:
            continue
        mine = sections.get(name) or {}
        source = mine.get("source")
        if source == AI:
            fits = mine.get("text") and mine.get("directions") == known[name][1] \
                and checked(name, mine["text"], s, known) is not None
            out[name] = (mine["text"], AI) if fits else (fallback, TEMPLATE)
        else:
            out[name] = (fallback, source if source in (AI_OFF, AI_FAILED, AI_REFUSED) else TEMPLATE)
    return out


def stored_sentences(saved: Optional[dict], s: Sources, known: Optional[dict] = None) -> dict:
    """The sentences the note is drawn with (`stored`), by section."""
    return {name: text for name, (text, _) in stored(saved, s, known).items()}


__all__ = ["AI", "AI_FAILED", "AI_OFF", "AI_REFUSED", "PROMPT_VERSION", "SECTIONS", "SlotError", "checked", "facts",
           "parse", "prompt", "slots_for", "stored", "stored_sentences", "template", "write"]
