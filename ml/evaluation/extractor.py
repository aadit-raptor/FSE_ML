"""The document reader's evaluation (ml/nlp_extractor.py, ml/document_reader.py; PLAN.md 6.2, 5.1).

Question: when the reader shows a value without amber, is it the document's? The card's bar, **set before any
document was read** (2026-10-21): **no wrong figure or label left unflagged, no planted misread left unflagged,
and at least 80% of the truth file's figures found without amber.** Production reads documents for people only
while the headline set meets it (``ml/document_reader.py`` ``open_to_people``).

- **Cases**: one per document, from its recorded answers (``tests/fixtures/extract/<set>/``, ``python -m
  ml.document_reader record``), replayed through ``assemble``, so the card is the reader's own and CI makes no
  AI call. Which quotes are on their page is recorded as hashes beside the reads (no page text is committed).
- **The set**: ``evaluation``, public annual reports and financial statements from six countries in five
  languages (German, Japanese, French, Spanish, English), read once the prompt (``x1``) was written and never
  tuned on them: the headline.
- **Wrong values** come two ways. The **truth file** (``tests/fixtures/extract/truth.json``): each document's
  headline figures and its labels, typed by hand from the document's own statements; a value shown without
  amber that is not the truth's is a wrong value missed. **Planted misreads**: one figure of read A at a time,
  a digit changed in its value *and* in its quote (a model that misreads a digit writes it everywhere),
  ``PLANTS`` a document, seeded by its name.
- **Baseline**: one read taken as it answers (what the old ``nlp_extractor`` did: a single AI answer, no second
  read and no check).
- **Regions**: S&P's (``library.base_rates.sp_region``), by the document's country.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ml import document_reader as dr
from ml import nlp_extractor as nx
from ml.evaluation.harness import MIN_CASES, Case, Metric, evaluate_set, metric_specs

MODEL_ID = "extractor"
SETS = ("evaluation",)
HEADLINE_SET = "evaluation"
PLANTS = 10
BAR = {"truth_missed": 0, "planted_missed_share": 0.0, "truth_found_share_at_least": 0.8}
REGIONS = ("us", "europe", "emerging", "other_developed")
TRUTH = dr.EXTRACTIONS / "truth.json"


def recordings(set_name: str, folder: Optional[Path] = None) -> list[dict]:
    folder = dr.EXTRACTIONS if folder is None else folder
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / set_name).glob("*.json"))]


def truth_file(path: Optional[Path] = None) -> dict:
    return json.loads((TRUTH if path is None else path).read_text(encoding="utf-8"))["documents"]


def run(record: Mapping, a: dict, b: Optional[dict]) -> dict:
    info = record["document"]
    dpi = info.get("dpi") or []

    def poor(page: int) -> bool:
        return 1 <= page <= len(dpi) and dpi[page - 1] is not None and dpi[page - 1] < 100

    return nx.assemble(a, b, quote_on_page=dr.recorded_lookup(record), poor=poor)


def one_read(a: dict) -> dict:
    """The baseline: read A's values as it gave them, nothing amber."""
    result = nx.assemble(a, None, quote_on_page=lambda page, quote: None, poor=lambda page: False)
    for found in result["labels"] + result["figures"]:
        found["amber"] = None
    return result


def _matches(value, truth) -> bool:
    wanted = truth if isinstance(truth, list) else [truth]
    return any(abs(value - w) <= 1e-6 * max(1.0, abs(w)) for w in wanted) if isinstance(value, (int, float)) \
        else value in wanted


def against_truth(result: Mapping, truth: Mapping) -> dict:
    """Counts of the truth file's figures and labels: found without amber, wrong without amber."""
    shown = {(f["key"], f["period_end"]): f for f in result["figures"]}
    labels = {i["key"]: i for i in result["labels"]}
    out = {"truth": 0, "found": 0, "missed": 0, "labels": 0, "labels_right": 0}
    for figure in truth["figures"]:
        out["truth"] += 1
        found = shown.get((figure["key"], figure["period_end"]))
        if found is None or found["amber"] is not None:
            continue
        if _matches(found["value"], figure["value"]):
            out["found"] += 1
        else:
            out["missed"] += 1
    for key, value in truth["labels"].items():
        out["labels"] += 1
        found = labels.get(key)
        if found is None or found["amber"] is not None:
            continue
        if found["value"] == value:
            out["labels_right"] += 1
        else:
            out["missed"] += 1
    return out




def _read_as(token: str, decimal: bool) -> Optional[float]:
    plain = re.sub(r"[ \u00a0\u202f]", "", token)
    last = max(plain.rfind("."), plain.rfind(","))
    if not decimal:
        return float(re.sub(r"[.,']", "", plain))
    if last <= 0 or not plain[last + 1:].isdigit():
        return None
    return float(re.sub(r"[.,']", "", plain[:last]) + "." + plain[last + 1:])


def _plant_one(found: dict, rng: random.Random) -> Optional[tuple[float, str]]:
    """A digit changed in the figure's printed number, in its quote and in its value alike."""
    quote = unicodedata.normalize("NFKC", found["quote"])
    magnitude = abs(found["value"])
    for match in list(nx.NUMBER.finditer(quote)) + list(nx.SPACED.finditer(quote)):
        token = match.group()
        for decimal in (False, True):
            reading = _read_as(token, decimal)
            if reading is None or abs(reading - magnitude) > 1e-9 * max(1.0, magnitude):
                continue
            places = [i for i, ch in enumerate(token) if ch.isdigit() and not (i == 0 and ch == "0")]
            at = rng.choice(places)
            digit = int(token[at])
            new = str((digit + rng.randint(1, 9)) % 10)
            if at == 0 and new == "0":
                new = str(digit % 9 + 1)
            changed = token[:at] + new + token[at + 1:]
            value = _read_as(changed, decimal)
            return (-value if found["value"] < 0 else value), quote[:match.start()] + changed + quote[match.end():]
    return None


def planted(record: Mapping, a: dict, b: Optional[dict], *, baseline: bool) -> tuple[int, int]:
    """``(planted, missed)``: misreads planted in read A one at a time on figures shown without amber, and how
    many of them are then shown without amber."""
    clean = one_read(a) if baseline else run(record, a, b)
    spots = [f for f in clean["figures"] if f["amber"] is None and f["value"]]
    rng = random.Random(record["document"]["name"])
    chosen = rng.sample(spots, min(PLANTS, len(spots)))
    count = missed = 0
    for spot in chosen:
        plant = _plant_one(spot, rng)
        if plant is None:
            continue
        value, quote = plant
        changed = json.loads(json.dumps(a))
        for found in changed["figures"]:
            if (found["key"], found["period_end"]) == (spot["key"], spot["period_end"]):
                found["value"], found["quote"] = value, quote
        result = one_read(changed) if baseline else run(record, changed, b)
        after = next(f for f in result["figures"] if (f["key"], f["period_end"]) == (spot["key"], spot["period_end"]))
        count += 1
        missed += after["amber"] is None
    return count, missed


def score(record: Mapping, truth: Mapping) -> tuple[dict, dict]:
    a, b = dr.recorded_reads(record)
    out = []
    for baseline in (False, True):
        result = one_read(a) if baseline else run(record, a, b)
        counts = against_truth(result, truth)
        plants, plants_missed = planted(record, a, b, baseline=baseline)
        every = result["labels"] + result["figures"]
        out.append({**counts, "values": len(every), "amber": sum(i["amber"] is not None for i in every),
                    "planted": plants, "planted_missed": plants_missed})
    return out[0], out[1]


def _sum(preds: Sequence[Mapping], key: str) -> int:
    return sum(p[key] for p in preds)


def _share(top: str, bottom: str):
    def compute(_cases, preds):
        below = _sum(preds, bottom)
        return _sum(preds, top) / below if below else None
    return compute


def truth_missed(_cases, preds) -> Optional[float]:
    return float(_sum(preds, "missed")) if _sum(preds, "truth") else None


METRICS = (
    Metric("planted_missed_share", "lower", _share("planted_missed", "planted"), 0.0,
           "Share of misreads planted in read A (a digit changed in the value and its quote, one at a time) still "
           "shown without amber"),
    Metric("truth_missed", "lower", truth_missed, 0.0,
           "Figures and labels of the truth file shown without amber and not as the document prints them"),
    Metric("truth_found_share", "higher", _share("found", "truth"), 0.03,
           "Share of the truth file's figures shown without amber, as the document prints them"),
    Metric("labels_right_share", "higher", _share("labels_right", "labels"), 0.0,
           "Share of currency, unit, standard, fiscal year end and scope read right without amber"),
    Metric("amber_share", "lower", _share("amber", "values"), 0.03,
           "Share of values shown amber (the person must look at each one)"),
)
HEADLINE = "planted_missed_share"


def meets_bar(group: Mapping) -> Optional[bool]:
    model = group.get("model")
    if group.get("verdict") == "not_enough_data" or not model:
        return None
    return (model["planted_missed_share"] == BAR["planted_missed_share"]
            and model["truth_missed"] == BAR["truth_missed"]
            and model["truth_found_share"] is not None
            and model["truth_found_share"] >= BAR["truth_found_share_at_least"])


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    from library.base_rates import sp_region

    truths = truth_file()
    sets = {}
    for name in SETS:
        rows = []
        for record in recordings(name):
            doc = record["document"]
            truth = truths.get(doc["name"])
            if truth is None or "B" not in record["reads"]:
                continue
            model, baseline = score(record, truth)
            rows.append((Case(id=doc["name"], region=sp_region(truth["country"]) or "other_developed",
                              year=None, features={"language": truth["language"], "prompt": record["prompt"]},
                              truth=None), model, baseline))
        result = evaluate_set(rows, METRICS, HEADLINE, regions=REGIONS)
        languages = sorted({r[0].features["language"] for r in rows})
        sets[name] = {"description": f"Public annual reports and financial statements read after the prompt was "
                                     f"written, none tuned on: {len(rows)} documents in {', '.join(languages)}, "
                                     f"{sum(r[1]['truth'] for r in rows)} truth figures, "
                                     f"{sum(r[1]['values'] for r in rows):,} values",
                      "meets_bar": meets_bar(result["overall"]), **result}
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "none", "why": "nothing is trained; every document was read after the prompt was "
                  "frozen, and none changed it"},
        "baseline": {"id": "one_read", "description": "One AI read taken as it answers, with no second read and "
                     "no check (what ml/nlp_extractor.py did before PLAN.md 6.2)"},
        "bar": {"description": "Set before any document was read: no wrong figure or label and no planted misread "
                "left unflagged, and at least 80% of the truth file's figures found without amber; production "
                "reads documents only while the headline set meets it", **BAR},
        "data": {"plants_per_document": PLANTS, "min_cases": MIN_CASES},
        "sets": sets,
    }


CARD = {
    "id": MODEL_ID,
    "name": "Document reader",
    "module": "ml/nlp_extractor.py",
    "used_by": "Deal -> From a document (POST /api/documents/reads, job document.extract); production reads "
               "documents only while this card's headline set meets its bar",
    "estimates": "A document's currency, unit, accounting standard, fiscal year end and scope, and up to fifteen "
                 "headline figures for its latest two years, each with its page and a verbatim quote, read "
                 "twice by two different AI models, with each value amber when the reads differ, its quote is "
                 "not on its page, its quote does not hold it or its page is a poor scan",
    "training": {
        "data": "Nothing is trained: the prompt and checks are ml/nlp_extractor.py; the recorded answers are "
                "tests/fixtures/extract (python -m ml.document_reader record), the truth file "
                "tests/fixtures/extract/truth.json",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "Eight documents: four German-language (three from Germany, one from Austria), one each in Japanese, "
        "French, Spanish and English (India); no US filer, no information pack (they are not public), no scan.",
        "The truth file holds each document's headline figures only (four to nine of them); a wrong figure "
        "outside it is seen only through the planted misreads and the quote check.",
        "The quote check needs a text layer: on a scanned page a quote can't be looked up, and two reads making "
        "the same mistake there would pass.",
        "Both reads may take the same line for a figure the document prints twice (an operating result before "
        "and after other items, profit before and after minorities): the truth file allows either.",
    ],
}
