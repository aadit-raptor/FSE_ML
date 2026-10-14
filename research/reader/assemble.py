"""Two reads of a filing made into checked statements (PLAN.md R1).

``assemble`` takes read A (the whole PDF) and read B (pictures of the pages
A found the tables on), as the AI answered them, and gives what the analyst
sees: read A's figures, each with its page and place in the table, its
period column and whether the table is standalone or consolidated, the
standard line its row matches, and **why it is amber**, if it is:

- the reads give it differently, or only one read has its row (or table);
- a sum the table states does not add up, the balance sheet does not
  balance, the cash flow's change in cash does not reach its closing cash;
- it sits on a poorly scanned page and no sum that holds confirms it;
- it differs from what is already stored for that line and period.

Column headings are checked apart from the figures (``checks.periods``):
their problems are listed under ``problems``, with a unit or currency the
reads disagree on, a scope that differs, or a table only one read found.

Pure: no provider, no file, no clock (``today`` is passed in), so the
recorded answers give the same result everywhere (tests, the model card).
"""
from __future__ import annotations

import datetime
from typing import Any, Mapping, Optional, Sequence

from research.reader import checks, lines
from research.reader.pages import Info

UNITS = ("lakhs", "crores", "millions", "thousands", "units")
STATEMENTS = ("results", "segments", "balance_sheet", "cash_flow")
SCOPES = ("consolidated", "standalone")
MAX_TABLES = 12
MAX_ROWS = 400
MAX_COLUMNS = 16
MAX_LABEL = 200


class Unreadable(ValueError):
    """The answer is not the shape asked for."""


def _text(value: Any, limit: int = MAX_LABEL) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def unwrap(answer: Any) -> dict:
    """The answer as the one object asked for (some models put it in a list;
    R0 lesson 8), cut to sane sizes. Raises ``Unreadable`` otherwise."""
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    if not isinstance(answer, dict) or not isinstance(answer.get("tables"), list):
        raise Unreadable("not the object asked for")
    tables = []
    for table in answer["tables"][:MAX_TABLES]:
        if not isinstance(table, dict):
            continue
        columns = [c for c in (table.get("columns") or []) if isinstance(c, dict)][:MAX_COLUMNS]
        rows = []
        for row in (table.get("rows") or [])[:MAX_ROWS]:
            if not isinstance(row, dict):
                continue
            values = row.get("values") if isinstance(row.get("values"), list) else []
            page = row.get("page")
            rows.append({"label": _text(row.get("label")), "kind": _text(row.get("kind"), 40),
                         "page": page if isinstance(page, int) and not isinstance(page, bool) else None,
                         "values": [checks.number(v) for v in values[:len(columns)]]})
        pages = [p for p in (table.get("pages") or []) if isinstance(p, int) and not isinstance(p, bool)]
        tables.append({
            "statement": table.get("statement") if table.get("statement") in STATEMENTS else "results",
            "scope": table.get("scope") if table.get("scope") in SCOPES else "consolidated",
            "pages": pages[:20],
            "columns": [{"label": _text(c.get("label")), "period_end": _text(c.get("period_end"), 10),
                         "months": c.get("months") if isinstance(c.get("months"), int) else None,
                         "audited": bool(c.get("audited"))} for c in columns],
            "rows": rows,
        })
    unit = _text(answer.get("unit"), 20).lower()
    return {"company": _text(answer.get("company")), "currency": _text(answer.get("currency"), 3).upper(),
            "unit": unit if unit in UNITS else None, "tables": tables}


def picture_pages(read: dict, sent: Sequence[int]) -> dict:
    """Read B numbers its pictures 1, 2 ...: give them back their PDF pages."""
    def page(n):
        return sent[n - 1] if isinstance(n, int) and 1 <= n <= len(sent) else None

    for table in read["tables"]:
        table["pages"] = [p for p in (page(n) for n in table["pages"]) if p is not None]
        for row in table["rows"]:
            row["page"] = page(row["page"])
    return read


def table_pages(read: Mapping) -> list[int]:
    """Every page read A found a table on, in order: what read B is shown."""
    found: list[int] = []
    for table in read["tables"]:
        for page in table["pages"]:
            if page >= 1 and page not in found:
                found.append(page)
    return sorted(found)


def _partner(read_b: Optional[Mapping], table: Mapping) -> Optional[Mapping]:
    """Read B's table for the same statement, of the same scope when B has
    one (a filing may hold standalone and consolidated statements both)."""
    if read_b is None:
        return None
    same = [t for t in read_b["tables"] if t["statement"] == table["statement"]]
    return next((t for t in same if t["scope"] == table["scope"]), same[0] if same else None)


def _row_page(table: Mapping, row: Mapping) -> Optional[int]:
    if row.get("page") in table["pages"]:
        return row["page"]
    return table["pages"][0] if len(table["pages"]) == 1 else None


def _table(table: dict, other: Optional[Mapping], read_twice: bool, info: Optional[Info],
           stored: Sequence[Mapping], problems: list) -> dict:
    found = checks.flags(table, other)
    amber = dict(found["amber"])
    if read_twice and other is None:
        # Read B did not find this table: nothing confirms its figures
        problems.append({"statement": table["statement"], "column": None, "code": "table_in_one_read",
                         "detail": "A"})
        for cell in found["figures"]:
            amber.setdefault(cell, (checks.ONE_READ_ONLY, None))
    if other is not None and other["scope"] != table["scope"]:
        problems.append({"statement": table["statement"], "column": None, "code": "scope_differs",
                         "detail": other["scope"]})
    if found["agreement"]:
        for item in found["agreement"]["headers"]:
            column = item[0] if isinstance(item[0], int) else None
            problems.append({"statement": table["statement"], "column": column, "code": "reads_differ_period",
                             "detail": str(item[-1])[:20]})
    assigned = lines.assign(table["statement"], table["rows"])
    for row, (line, source) in zip(table["rows"], assigned):
        row["line"], row["line_source"] = line, source
    # the statements' own arithmetic, once its lines are known
    if table["statement"] == "balance_sheet":
        for column, *rows in checks.balance(table):
            for r in rows:
                amber.setdefault((r, column), (checks.UNBALANCED, None))
    if table["statement"] == "cash_flow":
        for column, rows in checks.cash(table):
            for r in rows:
                if r is not None:
                    amber.setdefault((r, column), (checks.CASH_MISMATCH, None))
    for r, c, known in checks.against_stored(table, table["scope"], stored):
        amber.setdefault((r, c), (checks.STORED_DIFFERS, known))
    if info is not None:
        for r, c in found["figures"]:
            page = _row_page(table, table["rows"][r])
            pages = [page] if page else table["pages"]
            if any(info.poor(p) for p in pages) and (r, c) not in found["checked"]:
                amber.setdefault((r, c), (checks.POOR_SCAN, None))
    grid = checks.grid(table)
    rows = []
    for r, row in enumerate(table["rows"]):
        cells = []
        for c, value in enumerate(grid[r]):
            reason = amber.get((r, c))
            cells.append({"value": value, "amber": reason[0] if reason else None,
                          "other": reason[1] if reason else None, "in_sum": (r, c) in found["checked"]})
        rows.append({"label": row["label"], "line": row["line"], "line_source": row["line_source"],
                     "page": _row_page(table, row), "cells": cells})
    sums = found["sums"]
    return {
        "statement": table["statement"], "scope": table["scope"], "pages": table["pages"],
        "columns": table["columns"], "rows": rows,
        "summary": {"figures": len(found["figures"]), "amber": sum(1 for k in amber if k in found["figures"]),
                    "in_sums": len(found["figures"] & found["checked"]), "sums_found": len(sums["found"]),
                    "sums_failed": len(sums["failed"]),
                    "agreed": len(found["agreement"]["agreed"]) if found["agreement"] else 0},
    }


def assemble(read_a: Mapping, read_b: Optional[Mapping], *, today: datetime.date,
             info: Optional[Info] = None, quarter: Optional[datetime.date] = None,
             stored: Sequence[Mapping] = (), read_twice: bool = True) -> dict:
    """The checked statements from read A and read B (both ``unwrap``\\ ped;
    B's pages already PDF pages). ``read_twice=False`` with no read B is the
    one-read baseline the model card compares with."""
    problems: list[dict] = []
    if read_b is not None:
        for field in ("unit", "currency"):
            if read_a[field] != read_b[field]:
                problems.append({"statement": None, "column": None, "code": f"{field}_differs",
                                 "detail": str(read_b[field])})
        names_a = {t["statement"] for t in read_a["tables"]}
        for table in read_b["tables"]:
            if table["statement"] not in names_a:
                problems.append({"statement": table["statement"], "column": None, "code": "table_in_one_read",
                                 "detail": "B"})
    if read_a["unit"] is None:
        problems.append({"statement": None, "column": None, "code": "unit_unknown", "detail": ""})
    source = "given" if quarter else None
    if quarter is None:
        quarter = checks.filing_quarter(read_a["tables"], read_b["tables"] if read_b else [], today)
        source = "reads" if quarter else None
    if quarter is None:
        problems.append({"statement": None, "column": None, "code": "quarter_unknown", "detail": ""})
    year_end = checks.year_end_month(quarter, read_a["tables"]) if quarter else None
    for statement, column, code, detail in checks.periods(read_a["tables"], quarter, year_end):
        problems.append({"statement": statement, "column": column, "code": code, "detail": detail})
    tables = [_table(dict(t, rows=[dict(r) for r in t["rows"]]), _partner(read_b, t),
                     read_twice and read_b is not None, info, stored, problems) for t in read_a["tables"]]
    figures = sum(t["summary"]["figures"] for t in tables)
    amber = sum(t["summary"]["amber"] for t in tables)
    return {
        "company": read_a["company"], "currency": read_a["currency"] or None, "unit": read_a["unit"],
        "quarter_end": quarter.isoformat() if quarter else None, "quarter_source": source,
        "year_end_month": year_end, "tables": tables, "problems": problems,
        "summary": {"tables": len(tables), "figures": figures, "amber": amber,
                    "amber_share": round(amber / figures, 4) if figures else None,
                    "in_sums": sum(t["summary"]["in_sums"] for t in tables),
                    "period_problems": sum(1 for p in problems if p["code"] not in (
                        "unit_differs", "currency_differs", "table_in_one_read", "scope_differs",
                        "unit_unknown"))},
    }
