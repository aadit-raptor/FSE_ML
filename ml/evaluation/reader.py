"""The results reader's evaluation (research/reader/; PLAN.md R1, 5.1).

Question: when the reader shows a figure without amber, is it right; and is
it amber seldom enough to be worth using? R0's bar, kept as the card's:
**no wrong figure left unflagged, and under 2% of figures amber**.

- **Cases**: one per filing, from its recorded answers
  (``tests/fixtures/research/reader/<set>/``; ``research/reader/record.py``),
  replayed through the same code the API runs (``assemble``), so the card
  is the reader's and CI makes no AI call.
- **Sets**: ``development`` is R0's 30 filings, on which R0's prompt and
  checks were worked out; it can't show how the reader does on filings
  nobody tuned it on. ``held_out`` is filings read after the prompt and the
  checks were frozen (PLAN.md R1: "the card meets R0's bar on documents the
  prompts were not tuned on"); **it is the headline**, and until it holds
  ``MIN_CASES`` filings the card says "not enough data" and production
  keeps the reader closed (``research/reader/gate.py``). ``held_out_high``
  is the same filings' high-resolution trial (``READER_HIGH_RESOLUTION``),
  where recorded: the reader's setting follows whichever set does better.
- **Wrong figures** come three ways: a **planted** wrong digit (one figure
  of read A at a time, a digit changed, ``PLANTS`` a filing, seeded by the
  filing's name: what a misread looks like), the **truth file** (BSE's own
  figures for the filing's quarter, where saved: a figure shown without
  amber that is not BSE's, as one line or BSE's grouping of lines, is a
  wrong figure missed), and the reads' disagreements, amber by design.
- **Baseline**: one read with the sums checked (what a single AI read and
  arithmetic give); the reader beats it when it misses fewer planted digits.
- **Regions**: India only, for now.
"""
from __future__ import annotations

import datetime
import json
import random
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ml.evaluation.harness import MIN_CASES, Case, Metric, evaluate_set, metric_specs
from research.reader import assemble as asm
from research.reader import checks
from research.reader.pages import Info
from research.reader.providers import RECORDINGS

MODEL_ID = "reader"
# Each set: the folder its recordings are in and which pair of reads it scores
SOURCES = {"development": ("development", "reads"), "held_out": ("held_out", "reads"),
           "held_out_high": ("held_out", "high")}
SETS = tuple(SOURCES)
HEADLINE_SET = "held_out"
REGION = "india"
PLANTS = 12                       # planted digits a filing, each checked on its own
TODAY = datetime.date(2026, 10, 14)
BAR = {"amber_share_below": 0.02, "truth_missed": 0, "planted_missed_share": 0.0}
TO_MILLIONS = {"lakhs": 0.1, "crores": 10.0, "millions": 1.0, "thousands": 0.001, "units": 1e-6}
PER_SHARE = re.compile(r"EPS|Basic|Diluted")


def recordings(set_name: str, folder: Optional[Path] = None) -> list[dict]:
    folder = RECORDINGS if folder is None else folder
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / set_name).glob("*.json"))]


def reads(record: Mapping, key: str = "reads") -> tuple[dict, Optional[dict]]:
    """Read A and read B of a recording, unwrapped, B's pictures back on PDF pages."""
    a = asm.unwrap(record[key]["A"]["read"])
    b = record[key].get("B")
    if b is None:
        return a, None
    sent = [p for p in asm.table_pages(a) if p <= record["document"]["pages"]][:12]
    return a, asm.picture_pages(asm.unwrap(b["read"]), sent)


def _info(record: Mapping) -> Info:
    doc = record["document"]
    return Info(doc["pages"], tuple(doc["dpi"]))


def _quarter(record: Mapping) -> Optional[datetime.date]:
    q = record["document"].get("quarter_end")
    return datetime.date.fromisoformat(q) if q else None


def run(record: Mapping, a: dict, b: Optional[dict], *, read_twice: bool = True) -> dict:
    return asm.assemble(a, b, today=TODAY, info=_info(record), quarter=_quarter(record), read_twice=read_twice)


def _amber_cells(result: Mapping) -> dict[tuple, bool]:
    out = {}
    for t, table in enumerate(result["tables"]):
        for r, row in enumerate(table["rows"]):
            for c, cell in enumerate(row["cells"]):
                out[(t, r, c)] = cell["amber"] is not None
    return out


def _wrong_digit(value: float, rng: random.Random) -> Optional[float]:
    """The figure with one of its digits changed, as a misread would."""
    text = f"{abs(value):.2f}".rstrip("0").rstrip(".") if value != round(value) else str(int(abs(value)))
    places = [i for i, ch in enumerate(text) if ch.isdigit() and not (ch == "0" and i == 0)]
    if not places:
        return None
    at = rng.choice(places)
    digit = int(text[at])
    new = str((digit + rng.randint(1, 9)) % 10)
    if at == 0 and new == "0":
        new = str((digit % 9) + 1)
    changed = float(text[:at] + new + text[at + 1:])
    return -changed if value < 0 else changed


def planted(record: Mapping, a: dict, b: Optional[dict], *, read_twice: bool) -> tuple[int, int]:
    """``(planted, missed)``: wrong digits put into read A one at a time,
    each on a figure the clean reads show without amber, and how many of
    them the reader then shows without amber."""
    clean = run(record, a, b, read_twice=read_twice)
    amber = _amber_cells(clean)
    spots = [(t, r, c) for (t, r, c), is_amber in sorted(amber.items())
             if not is_amber and clean["tables"][t]["rows"][r]["cells"][c]["value"]]
    rng = random.Random(record["document"]["name"])
    chosen = rng.sample(spots, min(PLANTS, len(spots)))
    missed = count = 0
    for t, r, c in chosen:
        value = clean["tables"][t]["rows"][r]["cells"][c]["value"]
        wrong = _wrong_digit(value, rng)
        if wrong is None:
            continue
        changed = json.loads(json.dumps(a))
        changed["tables"][t]["rows"][r]["values"][c] = wrong
        result = run(record, changed, b, read_twice=read_twice)
        count += 1
        missed += result["tables"][t]["rows"][r]["cells"][c]["amber"] is None
    return count, missed


def truth(record: Mapping, result: Mapping) -> tuple[int, int, int]:
    """``(figures, confirmed, missed)`` against the truth file: a truth figure
    is confirmed when the reader shows it (as one line, or BSE's grouping of
    lines) without amber, and missed when it does not and nothing in its
    column is amber."""
    found_truth = record.get("truth")
    table = next((t for t in result["tables"] if t["statement"] == "results"), None)
    if not found_truth or table is None:
        return 0, 0, 0
    scale = TO_MILLIONS.get(result["unit"] or "", 1.0)
    grid = [[cell["value"] for cell in row["cells"]] for row in table["rows"]]
    unit = checks.step(grid) * scale
    labels = [row["label"] for row in table["rows"]]
    count = confirmed = missed = 0
    first = found_truth["figures"][0]
    for figure in found_truth["figures"]:
        column = next((i for i, col in enumerate(table["columns"]) if col.get("period_end") == figure["period_end"]
                       and col.get("months") == figure["months"]), None)
        if column is None and (figure["period_end"], figure["months"]) == (first["period_end"], first["months"]):
            column = 0
        if column is None:
            continue            # a period the filing does not print
        count += 1
        per_share = bool(PER_SHARE.search(figure["label"]))
        factor, slack = (1.0, 0.006) if per_share else (scale, 0.051 + 0.51 * unit)
        cells = [row["cells"][column] for row in table["rows"]]
        hits = [cell for cell in cells
                if cell["value"] is not None and abs(abs(cell["value"]) * factor - abs(figure["value"])) <= slack]
        ok = any(cell["amber"] is None for cell in hits)
        if not ok and figure.get("grouped"):
            # each line by its row, checked to still carry its label (a label can repeat)
            picks = [(row, sign) for label, sign, row in figure["grouped"]
                     if row < len(labels) and labels[row] == label]
            if len(picks) == len(figure["grouped"]) and all(cells[i]["value"] is not None for i, _ in picks):
                worked = sum(sign * abs(cells[i]["value"]) * factor for i, sign in picks)
                ok = abs(worked - abs(figure["value"])) <= slack * len(picks) \
                    and all(cells[i]["amber"] is None for i, _ in picks)
        if ok:
            confirmed += 1
        elif not any(cell["amber"] for cell in cells):
            missed += 1
    return count, confirmed, missed


def score(record: Mapping, key: str = "reads") -> tuple[dict, dict]:
    """The reader's counts on one filing, and the one-read baseline's."""
    a, b = reads(record, key)
    model_result, base_result = run(record, a, b), run(record, a, None, read_twice=False)
    out = []
    for result, read_twice in ((model_result, True), (base_result, False)):
        plants, plants_missed = planted(record, a, b if read_twice else None, read_twice=read_twice)
        figures, confirmed, missed = truth(record, result)
        out.append({"figures": result["summary"]["figures"], "amber": result["summary"]["amber"],
                    "planted": plants, "planted_missed": plants_missed, "truth": figures,
                    "truth_confirmed": confirmed, "truth_missed": missed,
                    "period_problems": result["summary"]["period_problems"]})
    return out[0], out[1]


def _sum(preds: Sequence[Mapping], key: str) -> int:
    return sum(p[key] for p in preds)


def amber_share(_cases: Sequence[Case], preds: Sequence[Mapping]) -> Optional[float]:
    figures = _sum(preds, "figures")
    return _sum(preds, "amber") / figures if figures else None


def planted_missed_share(_cases: Sequence[Case], preds: Sequence[Mapping]) -> Optional[float]:
    plants = _sum(preds, "planted")
    return _sum(preds, "planted_missed") / plants if plants else None


def truth_missed(_cases: Sequence[Case], preds: Sequence[Mapping]) -> Optional[float]:
    return float(_sum(preds, "truth_missed")) if _sum(preds, "truth") else None


def truth_confirmed_share(_cases: Sequence[Case], preds: Sequence[Mapping]) -> Optional[float]:
    figures = _sum(preds, "truth")
    return _sum(preds, "truth_confirmed") / figures if figures else None


METRICS = (
    Metric("planted_missed_share", "lower", planted_missed_share, 0.0,
           "Share of wrong digits planted in read A (one at a time) that the reader still shows without amber"),
    Metric("amber_share", "lower", amber_share, 0.002,
           "Share of figures shown amber (the analyst must look at each one)"),
    Metric("truth_missed", "lower", truth_missed, 0.0,
           "Figures of the truth file (BSE's own, where saved) shown without amber and not as printed"),
    Metric("truth_confirmed_share", "higher", truth_confirmed_share, 0.02,
           "Share of the truth file's figures the reader shows, without amber, as BSE gives them"),
)
HEADLINE = "planted_missed_share"


def meets_bar(group: Mapping) -> Optional[bool]:
    """R0's bar on a set's overall results; None when it has none."""
    model = group.get("model")
    if group.get("verdict") == "not_enough_data" or not model:
        return None
    return (model["amber_share"] is not None and model["amber_share"] < BAR["amber_share_below"]
            and model["planted_missed_share"] == BAR["planted_missed_share"]
            and (model["truth_missed"] in (None, 0.0)))


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    sets = {}
    for name in SETS:
        folder, key = SOURCES[name]
        rows = []
        for record in recordings(folder):
            # the trial is scored only beside a filing's ordinary reads, so the sets compare like with like
            if any("B" not in record.get(k, {}) for k in {"reads", key}):
                continue
            model, baseline = score(record, key)
            doc = record["document"]
            rows.append((Case(id=doc["name"], region=REGION, year=int(doc["quarter_end"][:4]) if doc.get(
                "quarter_end") else None, features={"prompt": record["prompt"]}, truth=None), model, baseline))
        result = evaluate_set(rows, METRICS, HEADLINE, regions=(REGION,))
        prompts = sorted({r[0].features["prompt"] for r in rows})
        described = {
            "development": "R0's 30 filings, on which the prompt and the checks were worked out (in-sample)",
            "held_out": "Filings read after the prompt and checks were frozen: none tuned on them",
            "held_out_high": "The held-out filings read again at high resolution (READER_HIGH_RESOLUTION=1)",
        }[name]
        sets[name] = {"description": f"{described}: {len(rows)} filings, "
                                     f"{sum(r[1]['figures'] for r in rows):,} figures, "
                                     f"prompts {', '.join(prompts) or '-'}",
                      "meets_bar": meets_bar(result["overall"]), **result}
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "regions_note": "India only for now: every filing is an Indian listed company's quarterly results",
        "split": {"kind": "none", "why": "nothing is trained; the held-out set is filings published or read after "
                  "the prompt and the checks were frozen, and the development set is labelled in-sample"},
        "baseline": {"id": "one_read", "description": "Read A alone, with the sums the table states checked "
                     "(what one AI read and arithmetic give)"},
        "bar": {"description": "R0's bar (PLAN.md R0): no wrong figure left unflagged and under 2% of figures "
                "amber; production reads filings only while the headline set meets it",
                **BAR},
        "data": {"plants_per_filing": PLANTS, "min_cases": MIN_CASES},
        "sets": sets,
    }


CARD = {
    "id": MODEL_ID,
    "name": "Results reader",
    "module": "research/reader/",
    "used_by": "Equity research -> Coverage -> Results (POST /api/research/reads, job research.read); "
               "production reads filings only while this card's headline set meets R0's bar",
    "estimates": "Every figure of a results filing's statements (results, segments, balance sheet, cash flow), "
                 "read twice by two different AI models, with each figure amber when the reads differ, a check "
                 "fails or the scan is poor",
    "training": {
        "data": "Nothing is trained: the prompts are research/reader/prompt.py and the checks "
                "research/reader/checks.py; the recorded answers are tests/fixtures/research/reader "
                "(python -m research.reader.record)",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "Two reads making the same mistake in a figure outside every sum is not seen: the planted digits test "
        "one read's misreads, and the truth file covers the quarter's headline figures of 17 filings only "
        "(ten development, seven held-out).",
        "The development set's prompt and checks were worked out on those very filings; only the held-out set "
        "says how the reader does on new ones.",
        "Six development filings were read by R0's first pass, the results table only; one (Yatharth Hospital) "
        "has a lite model's read A.",
        "Indian listed companies' results only; banks, insurers and lenders are left out.",
        "The held-out filings are eight June 2026 quarter or March 2026 year-end filings saved by hand, not a "
        "random sample; one hard scan (Larsen & Toubro's, its segment table read differently by the two models) "
        "holds most of their amber figures.",
        "The high-resolution trial (held_out_high) has fewer filings than the card needs: the free allowance "
        "ran out on the first day.",
        "A wrong year in every column heading is a period problem, not an amber figure: the analyst names the "
        "quarter, or the reads must agree on it.",
    ],
}
