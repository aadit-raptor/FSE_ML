"""The linking sentences of an investment memo (PLAN.md 6.3; R8's rule, research/writer.py).

**The AI writes words, the app writes numbers.** Each section (``SECTIONS``)
has slots: a figure of the run (``{irr}``) or a label (``{weak_driver}``, a
driver's name). The AI is sent each slot's name, a plain meaning ("the deal's
internal rate of return") and the directions the code worked out ("the IRR is
above the hurdle"), and nothing else: **never the deal's name, the version's
label, its country, industry or currency, a figure, a date or an account**.
Slot names hold no digit, so the prompt holds none at all
(``tests/test_memo.py`` proves it through Gemini's own code). So on the free
tier (``AI_FREE_TIER``) the AI receives only anonymised, non-identifying
text, and every figure is put in afterwards by the app.

**Its answer is checked** (``checked``): a section may name only its own
slots and hold no digit, no number word, no rating and no word of a call
(research/report.py ``words_problem``), and stay short. A section that fails,
is missing, or comes back while the AI is off, spent or busy is written by
its template (the catalogue's ``memo.doc.template_*``), and the memo says
which. The AI writes English only: for a reader in another language the
templates, which are translated, write every section.
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from ml.memo import (
    SCENARIOS, Sources, _have, figure, label, lookup,
)
from research.report import Text, words_problem
from research.writer import AI, AI_FAILED, AI_OFF, AI_REFUSED, MAX_SENTENCE, TEMPLATE

SECTIONS = ("summary", "metrics", "drivers", "scenarios", "plan")
PROMPT_VERSION = "m1"
SLOT = re.compile(r"\{([a-z][a-z_]*)\}")
NOT_ENGLISH = "not_english"     # the reader's language is not English: the templates write (memo.source_*)


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


def _compare(a: Optional[float], b: Optional[float], more: str = "higher", less: str = "lower") -> Optional[str]:
    if a is None or b is None or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return more if a > b else less


def _value(s: Sources, path: str) -> Optional[float]:
    try:
        v = lookup(s.run, path)
    except (KeyError, IndexError, TypeError):
        return None
    return None if v is None else float(v)


def facts(s: Sources) -> dict:
    """Per section: its slots (name -> (segment, meaning)) and the directions the code worked out. Only figures
    the run holds (a figure that is nil gets no slot)."""
    out: dict = {}

    def add(slots: dict, name: str, path: str, means: str) -> None:
        if _have(s, path):
            slots[name] = (figure(s, path), means)

    # summary
    slots: dict = {}
    add(slots, "irr", "returns.irr", "the deal's internal rate of return")
    add(slots, "moic", "returns.moic", "the multiple of the equity invested that comes back")
    add(slots, "holding", "returns.holding_period", "the holding period, in years")
    add(slots, "entry_equity", "returns.entry_equity", "the equity invested at entry")
    add(slots, "exit_equity", "returns.net_exit_equity", "the equity at exit")
    add(slots, "total_gain", "bridge.total_gain", "the gain in equity over the holding period")
    directions = []
    word = _compare(_value(s, "returns.net_exit_equity"), _value(s, "returns.entry_equity"))
    if word:
        directions.append("the equity at exit is %s than the equity invested at entry" % word)
    out["summary"] = (slots, directions)

    # the full metric set
    slots = {}
    add(slots, "irr", "returns.irr", "the deal's internal rate of return")
    add(slots, "hurdle", "simulation.hurdle", "the hurdle rate of return")
    add(slots, "beat", "simulation.p_above_hurdle", "the share of simulated paths that beat the hurdle")
    add(slots, "loss", "simulation.p_loss", "the share of simulated paths that lose money")
    add(slots, "downside", "simulation.p5_irr", "the rate of return of the weakest simulated paths")
    add(slots, "median", "simulation.median_irr", "the median simulated rate of return")
    add(slots, "cover", "credit.coverage", "interest cover in the first year")
    add(slots, "default", "credit.default_pct", "the default risk over the holding period")
    if _have(s, "credit.rating"):
        slots["grade"] = (label(s, "rating"), "the implied credit grade")
    directions = []
    word = _compare(_value(s, "returns.irr"), _value(s, "simulation.hurdle"), "above", "below")
    if word:
        directions.append("the deal's internal rate of return is %s the hurdle" % word)
    word = _compare(_value(s, "simulation.median_irr"), _value(s, "returns.irr"))
    if word:
        directions.append("the median simulated rate of return is %s than the deal's" % word)
    speculative = (s.run.get("credit") or {}).get("speculative")
    if speculative is not None:
        directions.append("the implied credit grade is %s" % ("speculative" if speculative else "investment grade"))
    out["metrics"] = (slots, directions)

    # drivers
    ex = s.run.get("explanations")
    cases = {c["case"]: i for i, c in enumerate(ex["cases"])} if ex else {}
    if {"downside", "upside"} <= set(cases):
        slots = {"mean_case_irr": (figure(s, "explanations.base_irr"),
                                   "the rate of return at the mean assumptions")}
        directions = []
        for case, name, pick, verb in (("downside", "weak", min, "lowers"), ("upside", "strong", max, "raises")):
            c = cases[case]
            contributions = ex["cases"][c]["contributions"]
            j = contributions.index(pick(contributions, key=lambda x: x["irr"]))
            where = "weakest" if case == "downside" else "strongest"
            slots[name] = (figure(s, "explanations.cases.%d.irr" % c),
                           "the mean rate of return of the %s paths" % where)
            slots[name + "_driver"] = (label(s, "driver:%d:%d" % (c, j)),
                                       "the driver that moves the %s paths the most" % where)
            slots[name + "_effect"] = (figure(s, "explanations.cases.%d.contributions.%d.irr" % (c, j)),
                                       "how much that driver moves their rate of return")
            directions.append("in the %s paths the driver {%s_driver} %s the rate of return the most"
                              % (where, name, verb))
        out["drivers"] = (slots, directions)

    # scenarios
    names = [n for n in SCENARIOS if _have(s, "scenarios.%s.median_irr" % n)]
    if len(names) >= 2:
        slots = {"%s_median" % n: (figure(s, "scenarios.%s.median_irr" % n),
                                   "the median rate of return in the %s scenario" % n) for n in names}
        ranked = sorted(names, key=lambda n: _value(s, "scenarios.%s.median_irr" % n))
        slots["lowest_name"] = (label(s, "scenario:" + ranked[0]), "the name of the scenario with the lowest")
        slots["highest_name"] = (label(s, "scenario:" + ranked[-1]), "the name of the scenario with the highest")
        out["scenarios"] = (slots, ["the median rate of return is lowest in the {lowest_name} scenario and highest "
                                    "in the {highest_name} scenario"])

    # plan against actual, once there is an actual exit
    pa = s.run.get("plan_actual") or {}
    if pa.get("actual") and _have(s, "plan_actual.actual.irr", "plan_actual.plan.irr"):
        slots = {}
        add(slots, "plan_irr", "plan_actual.plan.irr", "the planned rate of return")
        add(slots, "actual_irr", "plan_actual.actual.irr", "the rate of return achieved")
        add(slots, "percentile", "plan_actual.actual.percentile",
            "the share of the plan's simulated paths below the rate achieved")
        word = _compare(_value(s, "plan_actual.actual.irr"), _value(s, "plan_actual.plan.irr"))
        out["plan"] = (slots, ["the rate of return achieved is %s than planned" % word] if word
                       else ["the rate of return achieved is the same as planned"])
    return out


def slots_for(s: Sources, known: Optional[dict] = None) -> dict:
    """Every section's slots in one map (a name means the same figure in every section), for the draft builder."""
    out: dict = {}
    for slots, _ in (facts(s) if known is None else known).values():
        out.update({k: v[0] for k, v in slots.items()})
    return out


# ---------------------------------------------------------------------------------------------- templates
TEMPLATES = {
    "summary": ("template_summary",),
    "metrics": ("template_metrics", "template_metrics_credit"),
    "drivers": ("template_drivers",),
    "scenarios": ("template_scenarios",),
    "plan": ("template_plan",),
}


def template(section: str, s: Sources, known: Optional[dict] = None) -> Optional[str]:
    """The section's sentences when the AI does not write them, in the reader's language: each of the section's
    template sentences whose slots the run has."""
    found = (facts(s) if known is None else known).get(section)
    if not found:
        return None
    slots, _ = found
    doc = s.words["doc"]
    parts = [doc[key] for key in TEMPLATES[section] if set(SLOT.findall(doc[key])) <= set(slots)]
    return " ".join(parts) or None


# ---------------------------------------------------------------------------------------------- the AI
RULES = (
    "You write the linking sentences of an investment memo about a leveraged buyout. For each section below write "
    "one or two short sentences in plain English that use the section's slots, written exactly as {slot}; the "
    "application puts each figure or name in afterwards. Use only the slots listed for that section. Follow the "
    "directions; never guess one. Never write a digit, a number in words, a percentage, a rating (such as buy, sell "
    "or hold), a recommendation, or any view of whether to invest. Do not name a company, a country or a currency. "
    "Answer with one JSON object whose keys are the section names and whose values are the sentences, and nothing "
    "else.")


def prompt(s: Sources) -> Optional[str]:
    """What the AI is sent: the rules, then each section's slot names, meanings and directions. Nothing from the
    deal but which way its figures point."""
    sections = []
    for section, (slots, directions) in facts(s).items():
        sections.append({"section": section,
                         "slots": [{"slot": name, "means": means} for name, (_, means) in slots.items()],
                         "directions": directions})
    if not sections:
        return None
    return RULES + "\n\nSections:\n" + json.dumps(sections, indent=1, sort_keys=True)


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


def write(s: Sources, provider=None, spent: Optional[list] = None) -> dict:
    """Every section's sentences: the AI's when it keeps the rules, else the template's, saying why. ``spent`` gets
    the AI's answer, so what it used can be counted (PLAN.md R9)."""
    from research.reader.providers import ProviderError

    known = facts(s)
    out: dict = {}
    answer, reason, model = None, AI_OFF, None
    english = s.language == "en"
    text = prompt(s)
    if not english:
        reason = NOT_ENGLISH
    elif provider is not None and text is not None:
        try:
            got = provider.write(text)
            answer, model = got.read, got.model
            if spent is not None:
                spent.append(got)
        except ProviderError:
            reason = AI_FAILED
    for name in SECTIONS:
        fallback = template(name, s, known)
        if fallback is None:
            continue
        mine = checked(name, answer.get(name), s, known) if isinstance(answer, dict) else None
        if mine is not None:        # kept with the directions it was written from (`stored`)
            out[name] = {"text": mine, "source": AI, "directions": known[name][1]}
        else:
            out[name] = {"text": fallback, "source": reason if answer is None else AI_REFUSED}
    return {"sections": out, "model": model, "prompt_version": PROMPT_VERSION}


def stored(saved: Optional[dict], s: Sources, known: Optional[dict] = None) -> dict:
    """Each section's text and who wrote it: the AI's stored sentence while it still fits the run and the reader
    reads English, else the template's. Answers ``{section: (text, source)}``."""
    known = facts(s) if known is None else known
    out = {}
    sections = (saved or {}).get("sections") or {}
    for name in SECTIONS:
        fallback = template(name, s, known)
        if fallback is None:
            continue
        mine = sections.get(name) or {}
        source = mine.get("source")
        if source == AI and s.language == "en":
            fits = mine.get("text") and mine.get("directions") == known[name][1] \
                and checked(name, mine["text"], s, known) is not None
            out[name] = (mine["text"], AI) if fits else (fallback, TEMPLATE)
        elif s.language != "en":
            out[name] = (fallback, NOT_ENGLISH)
        else:
            out[name] = (fallback, source if source in (AI_OFF, AI_FAILED, AI_REFUSED) else TEMPLATE)
    return out


__all__ = ["NOT_ENGLISH", "PROMPT_VERSION", "SECTIONS", "SlotError", "checked", "facts", "parse", "prompt",
           "slots_for", "stored", "template", "write"]
