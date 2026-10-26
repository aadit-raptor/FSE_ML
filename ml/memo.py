"""An investment memo drafted from a saved deal version (PLAN.md 6.3).

**Every number comes from the model run.** ``run_version`` runs the version's
inputs and Settings through the very endpoints the screens call (the deal
model, the Monte Carlo simulation with its driver explanations, the four
scenario presets and, when actual results are saved, plan against actual), on
the Monte Carlo screen's seed, and keeps what the memo reads (``trim``): that
is the memo's **run**, stored with its model stamp (``db/memos.py``).

A draft is R8's (research/report.py): blocks of segments, each saying where it
came from. Here a ``Figure`` has source ``run`` and its ``row`` is a dotted
path into the run (``returns.irr``, ``scenarios.bull.median_irr``); a
``Label`` is a name the run, the version or the catalogue holds (the deal's
name, a year's fiscal label, a driver's name); a ``Text`` is the catalogue's
words or a linking sentence, and may hold **no digit, no number word and no
rating** (``report.words_problem``). ``check`` works every segment out again
from the run and refuses the draft (``report.DraftRefused``) on any
difference; the Word and PDF writers (research/report_files.py) only ever
write a checked draft.

**In the reader's language**: every word of the memo is in
``web/messages/<language>.json`` under ``memo.doc`` (driver names under
``engine``, year names under ``fiscal``), chosen from the account's locale the
way the web chooses its catalogue (``language_for``); a language with no
catalogue reads English. Figures follow the account's digit grouping.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

from research.report import (
    Chart, Draft, DraftRefused, FIGURE_DIFFERS, Figure, Heading, LABEL_DIFFERS, Label, Paragraph, Table, Text,
    number, show as report_show, words_problem,
)

KIND = "memo.draft"
SEED = 42                       # the Monte Carlo screen's seed, so the memo's simulation is the screen's
SCENARIOS = ("recession", "stagflation", "base", "bull")
MESSAGES = Path(__file__).resolve().parent.parent / "web" / "messages"
DEFAULT_LANGUAGE = "en"
SELF_SOURCE = "deal_model"
NO_RUN = "no_run"               # a stored run without what a memo reads (refusal code)


# ---------------------------------------------------------------------------------------------- the run
def sim_inputs(inputs: dict, settings: dict) -> dict:
    """The Monte Carlo screen's inputs for a deal: its Settings' ``mc_*`` values, the deal's EBITDA, entry multiple
    and holding period (web/src/components/montecarlo/MonteCarloProvider.tsx ``simFromSettings``)."""
    from api.deps import resolve_settings
    from api.schemas import MCInputsIn

    cfg = resolve_settings(settings)
    defaults = MCInputsIn()
    out = {name: cfg.get("mc_" + name, getattr(defaults, name)) for name in (
        "n", "hurdle", "growth_mean", "growth_std", "exit_mean", "exit_std", "rate_mean", "rate_std", "gm_mean",
        "gm_std")}
    out["n"] = int(out["n"])
    deal = {**_input_defaults(), **inputs}
    return {**out, "ebitda": deal["ebitda"], "entry_mult": deal["entry_mult"], "hold": deal["hold"]}


def _input_defaults() -> dict:
    from api.schemas import DealInputsIn

    return DealInputsIn().model_dump()


def run_version(inputs: dict, settings: dict, actuals: Optional[dict] = None) -> dict:
    """What the screens' endpoints answer for this version (the endpoint functions themselves, the simulations
    without their slot wrapper: the caller holds the slot), trimmed to what a memo reads."""
    from fastapi import HTTPException

    from pydantic import TypeAdapter

    from api.routers import backtesting, deal, montecarlo
    from api.schemas import (
        DealRunRequest, DealRunResponse, MonteCarloRequest, MonteCarloResponse, PlanActualRequest,
        PlanActualResponse, ScenariosRequest, ScenariosResponse,
    )
    from jobs.kinds import _unwrapped

    def answered(endpoint, request, response):     # the JSON the endpoint answers, as a job keeps it
        adapter = TypeAdapter(response)
        return adapter.dump_python(adapter.validate_python(_unwrapped(endpoint)(request)), mode="json")

    mc = sim_inputs(inputs, settings)
    answer = answered(deal.post_run, DealRunRequest(inputs=inputs, settings=settings), DealRunResponse)
    simulated = answered(montecarlo.post_run, MonteCarloRequest(
        mc=mc, deal=inputs, settings=settings, seed=SEED, histogram_bins=10, scatter_points=0), MonteCarloResponse)
    scenarios = answered(montecarlo.post_scenarios, ScenariosRequest(
        mc=mc, deal=inputs, settings=settings, seed=SEED), ScenariosResponse)
    plan = None
    if actuals:
        try:
            plan = answered(backtesting.post_plan_vs_actual, PlanActualRequest(
                plan=inputs, settings=settings, actuals=actuals), PlanActualResponse)
        except (HTTPException, ValueError):
            plan = {"refused": True}        # another currency, more years than the plan: said in the memo
    return trim({**_input_defaults(), **inputs}, mc, answer, simulated, scenarios, plan)


def draft(owner: str, deal_id, version: int) -> dict:
    """The job (``memo.draft``): the caller's version run as its memo reads it, and kept. The version and the
    actuals are read with the owner matched (another account's deal is ``DealNotFound``)."""
    from db import deals, memos

    record = deals.get_version(owner, deal_id, version)
    actuals, _ = deals.get_actuals(owner, deal_id)
    run = run_version(record.inputs or {}, record.settings or {}, actuals)
    try:
        changed = memos.save_memo(owner, deal_id, version, run)
    except memos.MemosFull:
        from fastapi import HTTPException

        # another draft took the last place since the endpoint checked: the job says so, as the endpoint would
        raise HTTPException(409, "memos_full") from None
    return {"version": version, "changed": changed}


def trim(inputs: dict, mc: dict, answer: dict, simulated: dict, scenarios: dict, plan: Optional[dict]) -> dict:
    """The run a memo reads: the figures it shows, nothing else (a few kilobytes)."""
    r, ds, br = answer["returns"], answer["debt_schedule"], answer["equity_bridge"]
    sources: dict = {}
    for item in answer["risk_warnings"] + [answer["credit"]]:
        for src in item.get("sources") or []:
            sources[src["id"]] = {k: src.get(k) or "" for k in ("publisher", "title", "published")}
    explanations = simulated.get("explanations")
    out = {
        "inputs": {"ebitda": inputs["ebitda"], "entry_mult": inputs["entry_mult"], "hold": inputs["hold"],
                   "first_fiscal_year": inputs.get("first_fiscal_year"),
                   "fiscal_year_end_month": inputs.get("fiscal_year_end_month") or 12},
        "returns": {k: r[k] for k in ("irr", "moic", "entry_equity", "exit_ebitda", "exit_multiple", "exit_ev",
                                      "net_debt_at_exit", "net_exit_equity", "holding_period")},
        "debt": {"at_close": ds["total_beginning_debt"][0], "at_exit": ds["total_ending_debt"][-1]},
        "bridge": {k: br[k] for k in ("entry_costs", "ebitda_growth", "multiple_expansion", "deleveraging",
                                      "total_gain")},
        "credit": {k: answer["credit"][k] for k in ("coverage", "rating", "default_pct", "years", "speculative")},
        "warnings": [{"id": w["id"], "figures": w["figures"], "labels": w.get("labels") or {},
                      "sources": [s["id"] for s in w.get("sources") or []]} for w in answer["risk_warnings"]],
        "sources": sources,
        "simulation": {"n": simulated["n"], "seed": SEED, **{k: simulated["summary"][k] for k in (
            "hurdle", "mean_irr", "median_irr", "p5_irr", "p95_irr", "p_above_hurdle", "p_loss")}},
        "explanations": None if not explanations else {
            "base_irr": explanations["base_irr"], "share": explanations["share"],
            "cases": [{"case": c["case"], "irr": c["irr"],
                       "contributions": [{"driver": x["driver"], "irr": x["irr"]} for x in c["contributions"]]}
                      for c in explanations["cases"]]},
        "scenarios": {name: {k: scenarios["scenarios"][name][k] for k in ("median_irr", "p5_irr", "p95_irr",
                                                                         "p_above_hurdle")}
                      for name in SCENARIOS if name in scenarios["scenarios"]},
        "plan_actual": None,
        "money": {"currency": answer["money"]["currency"], "unit": answer["money"]["unit"]},
        "model": answer["model"],
    }
    if plan is not None:
        if plan.get("refused"):
            out["plan_actual"] = {"refused": True}
        else:
            actual = plan.get("actual")
            out["plan_actual"] = {
                "refused": False, "years_compared": plan["years_compared"],
                "plan": {k: plan["plan"][k] for k in ("irr", "moic", "irr_p5", "irr_p95")},
                "actual": None if actual is None else {k: actual[k] for k in ("irr", "moic", "percentile")},
                "attribution": None if not plan.get("attribution") else
                {k: plan["attribution"][k] for k in ("exit_ebitda", "exit_multiple", "net_debt")},
            }
    return json.loads(json.dumps(out))         # plain JSON, as it is stored


# ---------------------------------------------------------------------------------------------- the words
@lru_cache(maxsize=8)
def catalogue(language: str) -> dict:
    """``memo.doc``, ``engine`` and ``fiscal`` of a language's catalogue, English underneath any key it lacks."""
    def read(lang: str) -> dict:
        path = MESSAGES / ("%s.json" % lang)
        if not re.fullmatch(r"[a-z]{2,3}", lang) or not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"doc": (data.get("memo") or {}).get("doc") or {}, "engine": data.get("engine") or {},
                "fiscal": data.get("fiscal") or {}}

    base, own = read(DEFAULT_LANGUAGE), read(language)
    return {part: {**base.get(part, {}), **own.get(part, {})} for part in ("doc", "engine", "fiscal")}


def language_for(locale: Optional[str]) -> str:
    """The catalogue a locale reads (web/src/lib/i18n/config.ts ``languageFor``): its language's, else English."""
    primary = re.split(r"[-_]", (locale or "").strip())[0].lower()
    return primary if primary and (MESSAGES / ("%s.json" % primary)).is_file() else DEFAULT_LANGUAGE


def _camel(name: str) -> str:
    return "".join(p[0].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", name) if p)


# ---------------------------------------------------------------------------------------------- what a draft reads
@dataclass(frozen=True)
class Meta:
    deal_name: str
    version: int
    version_label: str = ""
    saved_on: str = ""              # ISO date the version was saved


@dataclass(frozen=True)
class Sources:
    run: dict
    meta: Meta
    language: str = DEFAULT_LANGUAGE
    grouping: str = "thousands"
    sentences: dict = field(default_factory=dict)       # section -> text with {slot}s (ml/memo_writer.py)

    @property
    def words(self) -> dict:
        return catalogue(self.language)


# ---------------------------------------------------------------------------------------------- figures
MONEY = {"ebitda", "entry_equity", "exit_ebitda", "exit_ev", "net_debt_at_exit", "net_exit_equity", "at_close",
         "at_exit", "entry_costs", "ebitda_growth", "multiple_expansion", "deleveraging", "total_gain", "unfunded",
         "unfunded_total", "repayment_due"}
SHARE = {"irr", "mean_irr", "median_irr", "p5_irr", "p95_irr", "p_above_hurdle", "p_loss", "hurdle", "base_irr",
         "irr_p5", "irr_p95", "share"}
TIMES = {"moic", "exit_multiple", "entry_mult", "leverage", "threshold"}
TIMES2 = {"coverage", "band_low", "band_high"}       # cover and its bands: their second decimal tells bands apart
POINTS = {"default_pct", "percentile"}          # already percentages
COUNT = {"holding_period", "hold", "years", "n", "seed", "count", "years_compared"}


def unit_of(path: str) -> str:
    """The unit a run figure is written in, by its path; an unknown figure has none (``KeyError``)."""
    last = path.rsplit(".", 1)[-1]
    if path.startswith("plan_actual.attribution."):
        return "money"                  # the exit equity gap's parts, money whatever their names
    for unit, names in (("money", MONEY), ("share", SHARE), ("times", TIMES), ("times2", TIMES2), ("pct", POINTS), ("count", COUNT)):
        if last in names:
            return unit
    raise KeyError(path)


def show(value: Optional[float], unit: str, grouping: str = "thousands") -> str:
    if unit in ("pct", "times2") and value is not None and math.isfinite(value):
        return number(value, 1 if unit == "pct" else 2, grouping) + ("%" if unit == "pct" else "x")
    return report_show(value, unit, grouping)


def lookup(run: dict, path: str):
    node = run
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def figure(s: Sources, path: str) -> Figure:
    unit = unit_of(path)
    value = lookup(s.run, path)
    value = None if value is None else float(value)
    return Figure("run", path, "", unit, value, show(value, unit, s.grouping))


# ---------------------------------------------------------------------------------------------- labels
UNITS = ("thousands", "millions", "billions", "lakhs", "crores")


def year_label(s: Sources, index: int) -> str:
    """The deal's year ``index`` (1 is the first year of the holding period) as the screens name it."""
    fiscal, inputs = s.words["fiscal"], s.run["inputs"]
    first = inputs.get("first_fiscal_year")
    if not first:
        return fiscal["yearShort"].replace("{number}", str(index))
    end_year, month = int(first) + index - 1, int(inputs.get("fiscal_year_end_month") or 12)
    if month == 12:
        return fiscal["labelSingle"].replace("{year}", str(end_year))
    return fiscal["labelSpan"].replace("{first}", str(end_year - 1)).replace("{second}", "%02d" % (end_year % 100))


def _label_text(s: Sources, source: str) -> str:
    kind, _, rest = source.partition(":")
    run, words = s.run, s.words
    if kind == "deal_name":
        return s.meta.deal_name
    if kind == "version":
        return str(s.meta.version)
    if kind == "version_label":
        return s.meta.version_label
    if kind == "saved_on":
        return s.meta.saved_on
    if kind == "money":
        unit = run["money"]["unit"]
        return "%s %s" % (run["money"]["currency"], words["doc"]["unit_" + unit] if unit in UNITS else unit)
    if kind == "engine":
        return str((run.get("model") or {}).get("engine_version") or "")
    if kind == "rating":
        return str(run["credit"]["rating"] or "")
    if kind == "warning":           # warning:<i>:<label>
        i, name = rest.split(":")
        return str(run["warnings"][int(i)]["labels"][name])
    if kind == "warning_year":      # warning_year:<i>:<figure>, a year index as the year's name
        i, name = rest.split(":")
        return year_label(s, int(run["warnings"][int(i)]["figures"][name]))
    if kind == "source":
        src = run["sources"][rest]
        return ", ".join(x for x in (src["publisher"], src["title"], src["published"]) if x)
    if kind == "driver":            # driver:<case>:<i>
        c, i = rest.split(":")
        name = run["explanations"]["cases"][int(c)]["contributions"][int(i)]["driver"]
        return words["engine"].get("driver" + _camel(name), name)
    if kind == "scenario":
        if rest not in SCENARIOS:
            raise KeyError(rest)
        return words["doc"]["scenario_" + rest]
    raise KeyError(source)


def label(s: Sources, source: str) -> Label:
    return Label(source, _label_text(s, source))


# ---------------------------------------------------------------------------------------------- the check
def _check(s: Sources, seg, where: str, strict: bool) -> None:
    if isinstance(seg, Figure):
        try:
            unit = unit_of(seg.row)
            value = lookup(s.run, seg.row)
            value = None if value is None else float(value)
        except (KeyError, IndexError, ValueError, TypeError):
            raise DraftRefused(FIGURE_DIFFERS, where) from None
        same = (value is None and seg.value is None) or (
            value is not None and seg.value is not None and math.isclose(value, seg.value, rel_tol=1e-12, abs_tol=1e-9))
        if seg.source != "run" or seg.period or not same or seg.unit != unit \
                or seg.text != show(value, unit, s.grouping):
            raise DraftRefused(FIGURE_DIFFERS, where)
    elif isinstance(seg, Label):
        try:
            expected = _label_text(s, seg.source)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            raise DraftRefused(LABEL_DIFFERS, where) from None
        if seg.text != expected:
            raise DraftRefused(LABEL_DIFFERS, where)
    elif isinstance(seg, Text):
        problem = words_problem(seg.text, strict)
        if problem:
            raise DraftRefused(problem, where)
    else:                           # a memo has no field the person types
        raise DraftRefused(LABEL_DIFFERS, where)


def check(draft: Draft, s: Sources) -> Draft:
    """The draft, if every segment is its source's; else ``DraftRefused`` naming the first that is not."""
    from research.report import segments

    if draft.grouping != s.grouping:
        raise DraftRefused(FIGURE_DIFFERS, "grouping")
    for seg, where, strict in segments(draft):
        _check(s, seg, where, strict)
    return draft


# ---------------------------------------------------------------------------------------------- building a draft
SLOT = re.compile(r"\{([a-z][a-z_]*)\}")


def words(s: Sources, key: str, slots: Optional[dict] = None) -> tuple:
    """A catalogue sentence as segments, each ``{slot}`` the figure or label it names."""
    from ml.memo_writer import parse

    return tuple(parse(s.words["doc"][key], slots or {}))


def _t(s: Sources, key: str) -> tuple:
    return words(s, key)


def _row(s: Sources, key: str, *paths: str) -> tuple:
    return ((_t(s, key)),) + tuple((figure(s, p),) for p in paths)


def _have(s: Sources, *paths: str) -> bool:
    try:
        return all(lookup(s.run, p) is not None for p in paths)
    except (KeyError, IndexError, TypeError):
        return False


def _header(s: Sources) -> list:
    meta = s.meta
    version = words(s, "version_named" if meta.version_label else "version",
                    {"version": label(s, "version"), "label": label(s, "version_label")})
    saved = words(s, "saved_on", {"date": label(s, "saved_on")}) if meta.saved_on else ()
    return [
        Paragraph(_t(s, "title"), kind="note"),
        Heading((label(s, "deal_name"),), 1),
        Paragraph(version + ((Text(" "),) + saved if saved else ()), kind="note"),
        Paragraph(words(s, "money", {"money": label(s, "money")}), kind="note"),
    ]


def _summary(s: Sources) -> list:
    rows = [("row_ebitda", "inputs.ebitda"), ("row_entry_mult", "inputs.entry_mult"),
            ("row_hold", "returns.holding_period"), ("row_entry_equity", "returns.entry_equity"),
            ("row_debt_close", "debt.at_close"), ("row_exit_ebitda", "returns.exit_ebitda"),
            ("row_exit_multiple", "returns.exit_multiple"), ("row_exit_ev", "returns.exit_ev"),
            ("row_net_debt_exit", "returns.net_debt_at_exit"), ("row_exit_equity", "returns.net_exit_equity")]
    body = tuple(_row(s, key, path) for key, path in rows)
    gain = [("row_entry_costs", "bridge.entry_costs"), ("row_growth", "bridge.ebitda_growth"),
            ("row_multiple", "bridge.multiple_expansion"), ("row_delever", "bridge.deleveraging"),
            ("row_total_gain", "bridge.total_gain")]
    return [Heading(_t(s, "h_summary"), 2), Table(((Text(""),), _t(s, "col_value")), body),
            Heading(_t(s, "h_gain"), 2), Table(((Text(""),), _t(s, "col_value")), tuple(_row(s, k, p) for k, p in gain))]


def _metrics(s: Sources) -> list:
    rows = [("row_irr", "returns.irr"), ("row_moic", "returns.moic"), ("row_p_loss", "simulation.p_loss"),
            ("row_downside", "simulation.p5_irr"), ("row_median", "simulation.median_irr"),
            ("row_hurdle", "simulation.hurdle"), ("row_beat", "simulation.p_above_hurdle"),
            ("row_cover", "credit.coverage")]
    body = [_row(s, key, path) for key, path in rows]
    if _have(s, "credit.rating"):
        body.append((_t(s, "row_grade"), (label(s, "rating"),)))
    body.append(_row(s, "row_default", "credit.default_pct"))
    note = words(s, "metrics_note", {"paths": figure(s, "simulation.n"), "seed": figure(s, "simulation.seed")})
    return [Heading(_t(s, "h_metrics"), 2), Table(((Text(""),), _t(s, "col_value")), tuple(body)),
            Paragraph(note, kind="note")]


WARNING_FIGURES = {"leverage_above_guidance": ("leverage", "threshold"),
                   "implied_rating": ("coverage", "band_low", "band_high", "years", "default_pct"),
                   "interest_exceeds_ebitda": ("count", "coverage"),
                   "unfunded_repayment": ("count", "unfunded", "repayment_due", "unfunded_total")}
WARNING_YEARS = ("year", "worst_year")


def warning_slots(s: Sources, i: int) -> dict:
    w = s.run["warnings"][i]
    slots = {name: figure(s, "warnings.%d.figures.%s" % (i, name))
             for name in WARNING_FIGURES.get(w["id"], ()) if w["figures"].get(name) is not None}
    slots.update({name: label(s, "warning_year:%d:%s" % (i, name)) for name in WARNING_YEARS
                  if w["figures"].get(name) is not None})
    slots.update({name: label(s, "warning:%d:%s" % (i, name)) for name in w["labels"]})
    return slots


def _risks(s: Sources) -> list:
    out: list = [Heading(_t(s, "h_risks"), 2)]
    warnings = s.run["warnings"]
    if not warnings:
        return out + [Paragraph(_t(s, "risks_none"), kind="note")]
    cited: list = []
    for i, w in enumerate(warnings):
        key = "warning_" + w["id"]
        if w["id"] == "implied_rating" and w["figures"].get("band_low") is None:
            key = "warning_implied_rating_lowest"
        if key not in s.words["doc"]:
            raise DraftRefused(LABEL_DIFFERS, "warning %s" % w["id"])
        out.append(Paragraph(words(s, key, warning_slots(s, i))))
        # "this deal's own model run" goes unsaid: the memo's closing says every figure is the run's
        cited += [x for x in w["sources"] if x in s.run["sources"] and x not in cited and x != SELF_SOURCE]
    for src in cited:
        out.append(Paragraph(words(s, "source", {"source": label(s, "source:" + src)}), kind="note"))
    return out


def _drivers(s: Sources) -> list:
    out: list = [Heading(_t(s, "h_drivers"), 2)]
    ex = s.run.get("explanations")
    if not ex:
        return out + [Paragraph(_t(s, "drivers_none"), kind="note")]
    cases = {c["case"]: i for i, c in enumerate(ex["cases"])}
    order = [cases[c] for c in ("downside", "upside") if c in cases]
    if not order:
        return out + [Paragraph(_t(s, "drivers_none"), kind="note")]
    first = order[0]
    drivers = ex["cases"][first]["contributions"]
    # each tail lists the same drivers; matched by name, in case an order ever differs
    per_case = {c: [figure(s, "explanations.cases.%d.contributions.%d.irr" % (c, next(
        k for k, x in enumerate(ex["cases"][c]["contributions"]) if x["driver"] == d["driver"])))
        for d in drivers] for c in order}
    names = [label(s, "driver:%d:%d" % (first, j)) for j in range(len(drivers))]
    header = (_t(s, "col_driver"),) + tuple(_t(s, "col_" + ex["cases"][c]["case"]) for c in order)
    body = [((names[j],),) + tuple((per_case[c][j],) for c in order) for j in range(len(drivers))]
    body.append((_t(s, "row_base"),) + tuple((figure(s, "explanations.base_irr"),) for _ in order))
    body.append((_t(s, "row_tail"),) + tuple((figure(s, "explanations.cases.%d.irr" % c),) for c in order))
    chart = Chart(_t(s, "chart_drivers"), tuple(names),
                  tuple((_t(s, "col_" + ex["cases"][c]["case"]), tuple(per_case[c])) for c in order), unit="share")
    note = words(s, "drivers_note", {"share": figure(s, "explanations.share")})
    return out + [Table(header, tuple(body)), chart, Paragraph(note, kind="note")]


def _scenarios(s: Sources) -> list:
    out: list = [Heading(_t(s, "h_scenarios"), 2)]
    names = [n for n in SCENARIOS if n in s.run["scenarios"]]
    header = (_t(s, "col_scenario"), _t(s, "col_median"), _t(s, "col_p5"), _t(s, "col_p95"), _t(s, "col_beat"))
    body = tuple(((label(s, "scenario:" + n),),) + tuple(
        (figure(s, "scenarios.%s.%s" % (n, k)),) for k in ("median_irr", "p5_irr", "p95_irr", "p_above_hurdle"))
        for n in names)
    return out + [Table(header, body)]


def _plan(s: Sources) -> list:
    out: list = [Heading(_t(s, "h_plan"), 2)]
    pa = s.run.get("plan_actual")
    if pa is None:
        return out + [Paragraph(_t(s, "plan_none"), kind="note")]
    if pa.get("refused"):
        return out + [Paragraph(_t(s, "plan_refused"), kind="note")]
    rows = [_row(s, "row_years_compared", "plan_actual.years_compared"),
            _row(s, "row_plan_irr", "plan_actual.plan.irr"), _row(s, "row_plan_moic", "plan_actual.plan.moic"),
            _row(s, "row_plan_p5", "plan_actual.plan.irr_p5"), _row(s, "row_plan_p95", "plan_actual.plan.irr_p95")]
    if pa.get("actual"):
        rows += [_row(s, "row_actual_irr", "plan_actual.actual.irr"),
                 _row(s, "row_actual_moic", "plan_actual.actual.moic"),
                 _row(s, "row_percentile", "plan_actual.actual.percentile")]
    if pa.get("attribution"):
        rows += [_row(s, "row_gap_ebitda", "plan_actual.attribution.exit_ebitda"),
                 _row(s, "row_gap_multiple", "plan_actual.attribution.exit_multiple"),
                 _row(s, "row_gap_net_debt", "plan_actual.attribution.net_debt")]
    out.append(Table(((Text(""),), _t(s, "col_value")), tuple(rows)))
    if not pa.get("actual"):
        out.append(Paragraph(_t(s, "plan_held"), kind="note"))
    return out


def _sentence(s: Sources, section: str, slots: dict) -> list:
    from ml.memo_writer import parse

    text = s.sentences.get(section)
    return [Paragraph(tuple(parse(text, slots)), kind="sentence")] if text else []


def build(s: Sources) -> Draft:
    """The memo for a version's run, checked."""
    from ml.memo_writer import slots_for

    try:
        lookup(s.run, "returns.irr")
    except (KeyError, IndexError, TypeError):
        raise DraftRefused(NO_RUN, "returns") from None
    slots = slots_for(s)
    blocks: list = _header(s)
    blocks += _summary(s) + _sentence(s, "summary", slots)
    blocks += _metrics(s) + _sentence(s, "metrics", slots)
    blocks += _risks(s)
    blocks += _drivers(s) + _sentence(s, "drivers", slots)
    blocks += _scenarios(s) + _sentence(s, "scenarios", slots)
    blocks += _plan(s) + _sentence(s, "plan", slots)
    blocks.append(Paragraph(words(s, "closing", {"version": label(s, "version"), "engine": label(s, "engine")}),
                            kind="note"))
    blocks.append(Paragraph(_t(s, "footer"), kind="note"))
    return check(Draft(tuple(blocks), s.grouping), s)


def with_sentences(s: Sources, sentences: dict) -> Sources:
    return replace(s, sentences=sentences)


__all__ = ["KIND", "Meta", "SCENARIOS", "SEED", "Sources", "build", "catalogue", "check", "figure", "label",
           "language_for", "lookup", "run_version", "show", "sim_inputs", "trim", "unit_of", "with_sentences",
           "year_label"]
