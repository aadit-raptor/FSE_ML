"""Recorded answers for the reader's tests, its card and development (PLAN.md R1).

A recording is one filing's two reads as the AI answered them, with what
the evaluation needs beside them (never the PDF, which stays on the machine
that recorded it):

    tests/fixtures/research/reader/<set>/<name>.json
    {"document": {"name", "sha256", "pages", "dpi", "quarter_end", "set"},
     "prompt": "r0" | "r0-first" | "r1",
     "reads": {"A": {"model", "read", "usage", "seconds"}, "B": {...}},
     "high": {"A": ..., "B": ...},          # the high-resolution trial, where made
     "truth": {"source", "unit", "figures": [{"label", "period_end", "months", "value"}]}}

Two ways in:

    python -m research.reader.record r0 --samples research-samples
        R0's saved reads (research-samples/r0/out) as the ``development`` set,
        with BSE's tables (research-samples/xbrl) as truth. No call.

    python -m research.reader.record read --set held_out [--high] FILE.pdf [...]
        Reads filings with Gemini (``GEMINI_API_KEY`` in the environment, or
        ``--env-file``); ``--high`` makes the high-resolution trial's two reads
        (``high``) beside the ordinary ones. BSE's table for the filing, where
        saved in ``--truth`` (research-samples/xbrl), is joined as truth once
        the ordinary read A is in: running the command again joins it with no
        call. The trial waits for a filing's ordinary pair, which the card
        scores it beside. **Each answer is saved as it arrives and never asked
        for twice**, so a run stopped by the day's allowance carries on the
        next day where it left off: the free tier gives 20 requests a day per
        model, two a filing, so recording is spread over days. A model whose
        day is spent is skipped; when all are, the run stops and says so.

Filings are public results, so what is committed is public figures.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from research.reader import assemble as asm
from research.reader import pages, prompt
from research.reader.providers import RECORDINGS, SPENT, Document, Gemini, ProviderError, sha256

NAME = re.compile(r"^(?P<company>[a-z0-9-]+)_FY(?P<year>\d\d)-Q(?P<quarter>[1-4])")


def quarter_from_name(stem: str, year_end_month: int = 3) -> Optional[str]:
    """``company_FY27-Q1...`` is the quarter to June 2026 for a March year end."""
    found = NAME.match(stem)
    if not found:
        return None
    from research.reader.checks import back, month_end

    year_end = month_end(2000 + int(found["year"]), year_end_month)
    return back(year_end, 3 * (4 - int(found["quarter"]))).isoformat()


def short_name(stem: str) -> str:
    found = NAME.match(stem)
    return f"{found['company']}_FY{found['year']}-Q{found['quarter']}" if found else stem


# ------------------------------------------------------------------ truth

def bse_figures(path: Path) -> Optional[dict]:
    """The table BSE's results page saves (an HTML table named .xls), as truth:
    its amounts with their periods. None unless it is in millions."""
    text = path.read_text(errors="replace")
    rows = []
    for row in re.findall(r"<tr[^>]{0,200}>(.{0,20000}?)</tr>", text, flags=re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]{0,500}>", "", cell)).strip()
                 for cell in re.findall(r"<t[dh][^>]{0,500}>(.{0,5000}?)</t[dh]>", row, flags=re.S | re.I)]
        if cells:
            rows.append(cells)
    info, figures = {}, []
    for cells in rows:
        label, values = cells[0], cells[1:]
        if label in ("Type", "Date Begin", "Date End", "Description", "Period Ending", "No. of Months"):
            info[label] = values
            continue
        if "Margin" in label or label == "Notes":
            continue
        for index, value in enumerate(values):
            try:
                amount = float(value.replace(",", ""))
            except ValueError:
                continue
            if amount:
                figures.append((label, index, amount))

    def day(value: str) -> str:
        return str(datetime.datetime.strptime(value, "%d-%b-%y").date())

    if "million" not in (info.get("Description") or [""])[0].lower():
        return None
    if "Date End" in info:
        periods = [(day(info["Date End"][0]), 3)]
    else:
        periods = [(day(e), int(m)) for e, m in zip(info["Period Ending"], info["No. of Months"])]
    return {"source": "BSE's results page table for the filing, saved by hand (PLAN.md R0)",
            "unit": "millions",
            "figures": [{"label": label[:80], "period_end": periods[i][0], "months": periods[i][1], "value": v}
                        for label, i, v in figures if i < len(periods)]}


# --------------------------------------------------------------- from R0

def _first_pass(saved: dict) -> dict:
    """R0's first pass read only the results table, in its own shape."""
    table = dict(saved["table"])
    out = {"company": table.pop("company", ""), "currency": table.pop("currency", ""),
           "unit": table.pop("unit", None)}
    pages_ = table.get("pages") or ([table["page"]] if table.get("page") else [])
    out["tables"] = [{"statement": "results", "scope": table.get("scope"), "pages": pages_,
                      "columns": table.get("columns") or [], "rows": table.get("rows") or []}]
    return out


def _best(folder: Path, pattern: str) -> Optional[Path]:
    found = sorted(folder.glob(pattern), key=lambda p: ("lite" in p.name, p.name))
    return found[0] if found else None


def from_r0(samples: Path, out: Path = RECORDINGS / "development") -> list[str]:
    saved = samples / "r0" / "out"
    written = []
    for pdf in sorted(samples.glob("*.pdf")):
        a, b = _best(saved, f"{pdf.stem}.SA.*.json"), _best(saved, f"{pdf.stem}.SB.*.json")
        kind = "r0"
        if a is None or b is None:
            a, b, kind = _best(saved, f"{pdf.stem}.A.*.json"), _best(saved, f"{pdf.stem}.B.*.json"), "r0-first"
        if a is None or b is None:
            print(f"{pdf.name}: no pair of reads saved")
            continue

        def entry(path: Path) -> dict:
            data = json.loads(path.read_text())
            read = data["read"] if kind == "r0" else _first_pass(data)
            model = data.get("model") or path.name.rsplit(".", 2)[-2]
            return {"model": model, "read": read, "usage": data.get("usage") or {},
                    "seconds": data.get("seconds")}

        info = pages.info(pdf)
        record = {"document": {"name": short_name(pdf.stem), "sha256": sha256(pdf.read_bytes()),
                               "pages": info.pages, "dpi": list(info.dpi),
                               "quarter_end": quarter_from_name(pdf.stem), "set": "development"},
                  "prompt": kind, "reads": {"A": entry(a), "B": entry(b)}}
        ha, hb = _best(saved, f"{pdf.stem}.HA.*.json"), _best(saved, f"{pdf.stem}.HB.*.json")
        if ha and hb:
            record["high"] = {"A": entry(ha), "B": entry(hb)}
        bse = next(iter((samples / "xbrl").glob(f"{short_name(pdf.stem)}*.xls")), None)
        if bse is not None:
            truth = bse_figures(bse)
            if truth:
                record["truth"] = truth
                group_truth(record)
        write(out / f"{short_name(pdf.stem)}.json", record)
        written.append(short_name(pdf.stem))
    return written


TO_MILLIONS = {"lakhs": 0.1, "crores": 10.0, "millions": 1.0, "thousands": 0.001, "units": 1e-6}
PER_SHARE = re.compile(r"EPS|Basic|Diluted")
# Groupings checked on the page by hand in R0 (more lines than the search tries):
# Hindustan Aeronautics' "other expenses" on BSE's page is five of its lines
HAND_GROUPS = {("hindustan-aeronautics_FY27-Q1", "Other Expenses"): (
    ("other expenses", 1), ("impairment", 1), ("direct input", 1), ("provision", 1), ("capitalised", -1))}


def truth_column(table: dict, figure: dict, first: dict) -> Optional[int]:
    """The read's column for a truth figure's period; the filing's quarter is
    the first column whatever a read made of its heading."""
    for index, column in enumerate(table.get("columns") or []):
        if column.get("period_end") == figure["period_end"] and column.get("months") == figure["months"]:
            return index
    same = (figure["period_end"], figure["months"]) == (first["period_end"], first["months"])
    return 0 if same else None


def group_truth(record: dict) -> None:
    """For each truth figure read A does not print as one line, the two to
    four of A's lines (added or taken off) that make it, as labels: BSE
    groups lines its own way (other expenses with impairment, ...)."""
    import itertools

    from research.reader import checks

    truth = record.get("truth")
    if not truth or not truth["figures"]:
        return
    read = asm.unwrap(record["reads"]["A"]["read"])
    table = next((t for t in read["tables"] if t["statement"] == "results"), None)
    if table is None:
        return
    grid = checks.grid(table)
    scale = TO_MILLIONS.get(read["unit"] or "", 1.0)
    labels = [r["label"] for r in table["rows"]]
    for figure in truth["figures"]:
        column = truth_column(table, figure, truth["figures"][0])
        if column is None or PER_SHARE.search(figure["label"]):
            continue
        slack = 0.051 + 0.51 * checks.step(grid) * scale
        values = [None if row[column] is None else row[column] * scale for row in grid]
        if any(v is not None and abs(abs(v) - abs(figure["value"])) <= slack for v in values):
            continue
        hand = HAND_GROUPS.get((record["document"]["name"], figure["label"]))
        if hand:
            picks = []
            for word, sign in hand:
                index = next((i for i, label in enumerate(labels) if word in label.lower() and grid[i][column]), None)
                if index is not None:
                    picks.append((index, sign))
            if len(picks) == len(hand):
                figure["grouped"] = [[labels[i], s, i] for i, s in picks]
                figure["grouped_by"] = "hand"
            continue
        cells = [(i, v) for i, v in enumerate(values) if v not in (None, 0)]
        found = None
        for size in (2, 3, 4):
            for picks in itertools.combinations(cells, size):
                for signs in itertools.product((1, -1), repeat=size):
                    if abs(sum(s * abs(v) for s, (_, v) in zip(signs, picks)) - abs(figure["value"])) <= slack:
                        found = [[labels[i], s, i] for s, (i, _) in zip(signs, picks)]
                        break
                if found:
                    break
            if found:
                break
        if found:
            figure["grouped"] = found
            figure["grouped_by"] = "search"


def write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------- live reads

def join_truth(record: dict, truth_dir: Optional[Path]) -> bool:
    """BSE's table for the filing as its truth, once read A is in; whether
    one was joined. Matched by the file's short name, so a table saved for
    another quarter of the same company is never taken."""
    if truth_dir is None or record.get("truth") or "A" not in record["reads"]:
        return False
    bse = truth_dir / f"{record['document']['name']}.xls"
    found = bse_figures(bse) if bse.is_file() else None
    if not found:
        return False
    record["truth"] = found
    group_truth(record)
    return True


def read_one(provider: Gemini, pdf: Path, target: Path, set_name: str, quarter: Optional[str],
             *, key: str = "reads", truth_dir: Optional[Path] = None) -> str:
    """Read one filing, saving after each answer, into ``reads`` or the
    high-resolution trial's ``high``. What happened, in words."""
    record = json.loads(target.read_text()) if target.exists() else None
    data = pdf.read_bytes()
    if record is None:
        info = pages.info(pdf)
        record = {"document": {"name": target.stem, "sha256": sha256(data), "pages": info.pages,
                               "dpi": list(info.dpi), "quarter_end": quarter or quarter_from_name(pdf.stem),
                               "set": set_name},
                  "prompt": prompt.PROMPT_VERSION, "reads": {}}
    if record["prompt"] != prompt.PROMPT_VERSION:
        return f"recorded with prompt {record['prompt']}, not {prompt.PROMPT_VERSION}: left as it is"
    if key != "reads" and "B" not in record["reads"]:
        return "the ordinary reads first: the trial is scored only beside them"
    reads = record.setdefault(key, {})
    for which in ("A", "B"):
        if which in reads:
            continue
        if which == "A":
            document = Document(record["document"]["sha256"], pdf=data)
            exclude = frozenset()
        else:
            sent = [p for p in asm.table_pages(asm.unwrap(reads["A"]["read"]))
                    if p <= record["document"]["pages"]][:pages.MAX_PICTURES]
            if not sent:
                return "read A found no tables"
            document = Document(record["document"]["sha256"], pictures=pages.render(pdf, sent))
            exclude = frozenset({reads["A"]["model"]})
        answer = provider.read(which, document, exclude=exclude)
        reads[which] = {"model": answer.model, "read": answer.read, "usage": dict(answer.usage),
                        "seconds": answer.seconds}
        write(target, record)
    joined = join_truth(record, truth_dir)
    if joined:
        write(target, record)
    return f"A by {reads['A']['model']}, B by {reads['B']['model']}" + (", truth joined" if joined else "")


def _key(env_file: Optional[Path]) -> str:
    if env_file is not None:
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY, or pass --env-file with a GEMINI_API_KEY= line")
    return key


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.reader.record")
    sub = parser.add_subparsers(dest="command", required=True)
    r0 = sub.add_parser("r0")
    r0.add_argument("--samples", type=Path, default=Path("research-samples"))
    live = sub.add_parser("read")
    live.add_argument("--set", default="held_out")
    live.add_argument("--env-file", type=Path)
    live.add_argument("--quarter", help="the quarter end, when the file name does not say (YYYY-MM-DD)")
    live.add_argument("--high", action="store_true", help="the high-resolution trial's reads (READER_HIGH_RESOLUTION)")
    live.add_argument("--truth", type=Path, default=Path("research-samples") / "xbrl",
                      help="where BSE's tables are saved (company_FYyy-Qn.xls)")
    live.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "r0":
        for name in from_r0(args.samples):
            print(f"wrote {name}")
        return 0
    provider = Gemini(_key(args.env_file), high_resolution=args.high or None)
    for pdf in args.files:
        target = RECORDINGS / args.set / f"{short_name(pdf.stem)}.json"
        try:
            said = read_one(provider, pdf, target, args.set, args.quarter,
                            key="high" if args.high else "reads", truth_dir=args.truth)
            print(f"{pdf.name}: {said}", flush=True)
        except ProviderError as exc:
            print(f"{pdf.name}: {exc}", flush=True)
            if exc.code == SPENT:
                print("Every model's allowance for today is spent: run the same command again tomorrow.")
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
