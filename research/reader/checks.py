"""The checks on two reads of a results filing (PLAN.md R1).

Ported from R0's ``score.py``, whose rules were measured on 30 filings and
7,899 figures (PLAN.md R0, "Lessons for R1"):

- **sums** (``identities``): arithmetic the table itself states, found from
  the numbers and not from an assumed layout -- a row that is the sum of the
  rows above it, one row less the rows between, a heading that carries the
  sum of the rows under it. A formula is chosen once per row for every
  column, so it cannot bend to hide one wrong cell; a "sum" that is only a
  copy of one line proves nothing.
- **agreement** (``agreement``): read A against read B, rows lined up by
  their numbers, not their labels (one read keeps a heading the other
  drops). A dash read as 0 against a blank is no difference; the same digits
  with the other sign are settled by a sum that holds, else amber.
- **periods** (``periods``): the column headings against the quarter the
  filing is for and against each other. A model wrote 2024 for 2026 in every
  heading of four reads in R0, on figures that were right.
- **statements**: the balance sheet balances, the cash flow's change in cash
  is its closing less its opening cash, and this document's comparative
  columns agree with what is already stored (``against_stored``).

Everything here is pure arithmetic on the reads; nothing calls a provider.
"""
from __future__ import annotations

import calendar
import datetime
import difflib
import re
from typing import Iterable, Mapping, Optional, Sequence

SUM, LESS, DIFF, BELOW = "sum", "sum_less", "difference", "sum_below"
MAX_ENTRIES = 40

# Why a figure is amber: codes, worded on the screen (web/messages/en.json reader.amber_*)
READS_DIFFER = "reads_differ"
SIGN_DIFFERS = "sign_differs"
ONE_READ_ONLY = "one_read_only"
SUM_FAILS = "sum_fails"
POOR_SCAN = "poor_scan"
UNBALANCED = "unbalanced"
CASH_MISMATCH = "cash_mismatch"
STORED_DIFFERS = "stored_differs"
AMBER_REASONS = (READS_DIFFER, SIGN_DIFFERS, ONE_READ_ONLY, SUM_FAILS, POOR_SCAN, UNBALANCED,
                 CASH_MISMATCH, STORED_DIFFERS)


def number(value) -> Optional[float]:
    """A cell as a float, or None for a blank. Text the model left in is a blank."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value and abs(value) != float("inf") else None
    text = str(value).strip().replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    try:
        found = float(text.strip("()"))
    except ValueError:
        return None
    if found != found or abs(found) == float("inf"):
        return None
    return -found if negative else found


def zero(value: Optional[float]) -> float:
    return 0.0 if value is None else value


def grid(table: Mapping) -> list[list[Optional[float]]]:
    width = len(table.get("columns") or [])
    rows = []
    for row in table.get("rows") or []:
        values = [number(v) for v in (row.get("values") or [])][:width]
        rows.append(values + [None] * (width - len(values)))
    return rows


def empty(values: Iterable[Optional[float]]) -> bool:
    return all(v is None or v == 0 for v in values)


def step(rows: Sequence[Sequence[Optional[float]]]) -> float:
    """The smallest amount the table prints: 1, or 0.01 when it shows decimals."""
    for values in rows:
        for value in values:
            if value is not None and abs(value - round(value)) > 1e-9:
                return 0.01
    return 1.0


def is_less(label: str) -> bool:
    return bool(re.match(r"\s*(\(?[a-z0-9ivx]{1,4}[).]\s*)?(less|\(-\)|-)\s*[:\-]?", label.lower())) \
        and "less" in label.lower()[:14]


RATIO = re.compile(r"per share|\beps\b|\bbasic\b|\bdiluted\b|\bratios?\b|\bmargin\b|%|\(times\)|face value")


# ------------------------------------------------------------- identities

def _holds(rows, target: int, parts: list, signs: list, width: int, unit: float) -> list:
    tolerance = unit * (1.01 if len(parts) <= 5 else 2.01)
    columns = []
    for column in range(width):
        printed = rows[target][column]
        if printed is None:
            columns.append(None)
            continue
        worked = sum(sign * zero(rows[part][column]) for part, sign in zip(parts, signs))
        columns.append(abs(printed - worked) <= tolerance + 1e-9 * abs(printed))
    return columns


def identities(table: Mapping) -> dict:
    """The sums the table states: ``{"found": [...], "checked": {(row, column)
    in a sum that holds}, "failed": [(row, column, printed, worked out)]}``."""
    rows = grid(table)
    labels = [str(row.get("label") or "") for row in table.get("rows") or []]
    width = len(table.get("columns") or [])
    unit = step(rows)
    is_ratio = [bool(RATIO.search(label.lower())) for label in labels]
    span: dict[int, int] = {}   # subtotal row -> first row of what it adds up
    children: set[int] = set()  # rows that a heading above them already adds up
    found = []

    def entries(target: int) -> list:
        out, at = [], target - 1
        while at >= 0 and len(out) < MAX_ENTRIES:
            if at not in children:
                out.append(at)
            at = span[at] - 1 if at in span else at - 1
        return out

    def candidates(target: int):
        above = entries(target)
        for count in range(2, len(above) + 1):
            parts = above[:count]
            if sum(not empty(rows[p]) for p in parts) < 2:
                continue
            yield SUM, parts, [1] * count
            if any(is_less(labels[p]) for p in parts):
                yield LESS, parts, [-1 if is_less(labels[p]) else 1 for p in parts]
            if not empty(rows[parts[-1]]):
                yield DIFF, parts, [-1] * (count - 1) + [1]
        for count in range(2, 8):
            parts = list(range(target + 1, target + 1 + count))
            if parts[-1] >= len(rows) or sum(not empty(rows[p]) for p in parts) < 2:
                continue
            yield BELOW, parts, [1] * count

    for target in range(len(rows)):
        printed = [v for v in rows[target] if v is not None]
        if not printed or empty(rows[target]) or is_ratio[target] or target in children:
            continue
        best, support = None, 0
        for form, parts, signs in candidates(target):
            if any(is_ratio[p] for p in parts):
                continue
            columns = _holds(rows, target, parts, signs, width, unit)
            # where all but one part is nil the "sum" is only a copy of a line,
            # and a copy proves no formula: only real sums count as support
            real = [sum(1 for p in parts if zero(rows[p][c]) != 0) >= 2 for c in range(width)]
            good = sum(ok is True and r and zero(rows[target][c]) != 0
                       for c, (ok, r) in enumerate(zip(columns, real)))
            shown = sum(c is not None for c in columns)
            if all(c is not False for c in columns) and good >= 1:
                best = (form, parts, signs, columns)
                break
            # A formula that fails somewhere is believed when it is a real sum
            # in at least two columns and fails in at most a third of them (one
            # at least): the columns it fails in are then real failures. Of
            # several, the one holding in the most columns.
            failing = sum(c is False for c in columns)
            if good >= 2 and failing <= max(1, shown // 3) and shown >= 3 and good > support:
                best, support = (form, parts, signs, columns), good
            # a balance sheet or cash flow has two columns: there a plain sum of
            # three or more lines that holds in one column is believed too (two
            # lines could match by chance)
            if shown == 2 and good == 1 and best is None and form in (SUM, LESS) \
                    and sum(not empty(rows[p]) for p in parts) >= 3 \
                    and any(ok and rows[target][c] for c, ok in enumerate(columns)):
                best, support = (form, parts, signs, columns), 1
        if best is None:
            continue
        form, parts, signs, columns = best
        if form == BELOW:
            children.update(parts)
        else:
            span[target] = min(span.get(p, p) for p in parts)
        found.append({"row": target, "form": form, "parts": parts, "signs": signs, "columns": columns})

    checked, failed = set(), []
    for item in found:
        for column, ok in enumerate(item["columns"]):
            if ok is True:
                checked.add((item["row"], column))
                checked.update((part, column) for part in item["parts"] if rows[part][column] is not None)
            elif ok is False:
                worked = sum(s * zero(rows[p][column]) for p, s in zip(item["parts"], item["signs"]))
                failed.append((item["row"], column, rows[item["row"]][column], round(worked, 2)))
    return {"found": found, "checked": checked, "failed": failed}


# -------------------------------------------------------------- agreement

def same(a: Optional[float], b: Optional[float]) -> bool:
    return abs(zero(a) - zero(b)) < 1e-6


def likeness(a: Sequence, b: Sequence) -> float:
    # by size, not sign: a "less" row one read gives as negative is the same row
    cells = [(x, y) for x, y in zip(a, b) if not (x is None and y is None)]
    return sum(same(abs(zero(x)), abs(zero(y))) for x, y in cells) / len(cells) if cells else 0.0


def _pair_block(a, b, left: list, right: list) -> list:
    """Rows that are not identical in the two reads: the most alike pair is
    joined, then the rows before it and the rows after it, each on their own."""
    if not left or not right:
        return [(x, None) for x in left] + [(None, y) for y in right]
    score, _, i, j = max((likeness(a[x], b[y]), -abs(m - n), m, n)
                         for m, x in enumerate(left) for n, y in enumerate(right))
    if score < 0.34:
        return [(x, None) for x in left] + [(None, y) for y in right]
    return (_pair_block(a, b, left[:i], right[:j]) + [(left[i], right[j])]
            + _pair_block(a, b, left[i + 1:], right[j + 1:]))


def align(a: Sequence, b: Sequence) -> list:
    """Pairs (row of A, row of B) over rows that hold a number, lined up by the
    numbers. A row with no partner comes back as (index, None) or (None, index)."""
    rows_a = [i for i, values in enumerate(a) if not empty(values)]
    rows_b = [i for i, values in enumerate(b) if not empty(values)]

    def key(values):
        return tuple(round(abs(zero(v)), 4) for v in values)

    matcher = difflib.SequenceMatcher(a=[key(a[i]) for i in rows_a], b=[key(b[i]) for i in rows_b],
                                      autojunk=False)
    pairs = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            pairs += [(rows_a[a0 + k], rows_b[b0 + k]) for k in range(a1 - a0)]
            continue
        pairs += _pair_block(a, b, list(rows_a[a0:a1]), list(rows_b[b0:b1]))
    return pairs


def agreement(table_a: Mapping, table_b: Mapping) -> dict:
    a, b = grid(table_a), grid(table_b)
    headers = []
    columns_a, columns_b = table_a.get("columns") or [], table_b.get("columns") or []
    if len(columns_a) != len(columns_b):
        headers.append(("count", len(columns_a), len(columns_b)))
    for index, (x, y) in enumerate(zip(columns_a, columns_b)):
        for field in ("period_end", "months"):
            if x.get(field) != y.get(field):
                headers.append((index, field, x.get(field), y.get(field)))
    width = min(len(columns_a), len(columns_b))
    cells = agree = 0
    differences, signs, only_a, only_b, agreed, partner = [], [], [], [], [], {}
    for i, j in align([r[:width] for r in a], [r[:width] for r in b]):
        if j is None:
            only_a.append(i)
            continue
        if i is None:
            only_b.append(j)
            continue
        partner[i] = j
        for column in range(width):
            x, y = a[i][column], b[j][column]
            if x is None and y is None:
                continue
            if zero(x) == 0 and zero(y) == 0:
                continue                      # a dash read as 0 against a blank
            cells += 1
            if same(x, y):
                agree += 1
                agreed.append((i, column))
            elif same(abs(zero(x)), abs(zero(y))):
                signs.append((i, column, x, y))     # same digits, other sign
            else:
                differences.append((i, column, x, y))
    return {"cells": cells, "agree": agree, "differences": differences, "signs": signs, "agreed": agreed,
            "only_a": only_a, "only_b": only_b, "headers": headers, "partner": partner}


# ------------------------------------------------------------------ flags

def figures(table: Mapping) -> set:
    """Every (row, column) of the table that holds a number other than a zero dash."""
    return {(r, c) for r, values in enumerate(grid(table)) for c, v in enumerate(values)
            if v is not None and v != 0}


def flags(table_a: Mapping, table_b: Optional[Mapping]) -> dict:
    """What is amber in read A, and why: ``{"amber": {(row, column): (code,
    other read's value or the worked-out sum)}, "checked": cells in a sum that
    holds, "agreement": ..., "sums": ...}``."""
    sums = identities(table_a)
    cells = figures(table_a)
    amber: dict[tuple, tuple] = {}
    for row, column, _printed, worked in sums["failed"]:
        amber[(row, column)] = (SUM_FAILS, worked)
    both = None
    if table_b is not None:
        both = agreement(table_a, table_b)
        for row, column, _x, y in both["differences"]:
            amber[(row, column)] = (READS_DIFFER, y)
        # a sign the reads disagree on is settled by a sum that holds; else it is amber
        for row, column, _x, y in both["signs"]:
            if (row, column) not in sums["checked"]:
                amber[(row, column)] = (SIGN_DIFFERS, y)
        for row in both["only_a"]:
            for cell in cells:
                if cell[0] == row:
                    amber.setdefault(cell, (ONE_READ_ONLY, None))
    return {"amber": amber, "checked": sums["checked"], "figures": cells, "sums": sums, "agreement": both}


# ---------------------------------------------------------------- periods

def month_end(year: int, month: int) -> datetime.date:
    return datetime.date(year, month, calendar.monthrange(year, month)[1])


def back(day: datetime.date, months: int) -> datetime.date:
    """The month end a number of months before a month end."""
    index = day.year * 12 + day.month - 1 - months
    return month_end(index // 12, index % 12 + 1)


def expected_periods(quarter: datetime.date, year_end_month: int = 3) -> dict:
    """What a results filing for a quarter may carry, as {months: set of period
    ends}: the quarter, the one before and the same one a year ago; the year so
    far and a year ago; the last full year and the one before; balance sheets
    (months 0) at the quarter end, the last year end and a year ago."""
    so_far = (quarter.month - year_end_month - 1) % 12 + 1      # months of the year so far
    last_year_end = quarter if so_far == 12 else back(quarter, so_far)
    allowed = {3: {quarter, back(quarter, 3), back(quarter, 12)},
               12: {last_year_end, back(last_year_end, 12)},
               0: {quarter, last_year_end, back(quarter, 12)}}
    allowed.setdefault(so_far, set()).update({quarter, back(quarter, 12)})
    return allowed


def column_date(column: Mapping) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(str(column.get("period_end")))
    except ValueError:
        return None


def year_end_month(quarter: datetime.date, tables: Sequence[Mapping]) -> int:
    """The month the company's year ends in: the end of a 12-month column at
    or before the quarter (results filings print the last full year), else
    March, Indian companies' usual year."""
    for table in tables:
        for column in table.get("columns") or []:
            end = column_date(column)
            if column.get("months") == 12 and end is not None and end <= quarter:
                return end.month
    return 3


def periods(tables: Sequence[Mapping], quarter: Optional[datetime.date],
            year_end: Optional[int] = None) -> list[tuple]:
    """Problems with the column headings of one read of a filing, as
    ``(statement, column or None, code, detail)``. Empty when the headings
    fit the filing's quarter and each other."""
    problems = []
    month = year_end or (year_end_month(quarter, tables) if quarter else 3)
    allowed = expected_periods(quarter, month) if quarter else None
    for table in tables:
        name = table.get("statement") or "results"
        seen, ends = set(), []
        for index, column in enumerate(table.get("columns") or []):
            months = column.get("months")
            end = column_date(column)
            if end is None:
                problems.append((name, index, "no_date", str(column.get("period_end"))[:20]))
                continue
            ends.append(end)
            if end != month_end(end.year, end.month):
                problems.append((name, index, "not_month_end", str(end)))
            if months not in (0, 3, 6, 9, 12):
                problems.append((name, index, "odd_length", str(months)[:10]))
            if (end, months) in seen:
                problems.append((name, index, "repeated", str(end)))
            seen.add((end, months))
            if allowed is None:
                continue
            if end > quarter:
                problems.append((name, index, "after_quarter", str(end)))
            elif end not in allowed.get(months, set()):
                problems.append((name, index, "unexpected_period", str(end)))
        if allowed is not None and ends and quarter not in ends:
            problems.append((name, None, "no_quarter_column", str(quarter)))
        # an earlier year stands beside the later one, never in its place
        newest = max(allowed[12]) if allowed else None
        if newest and any(m == 12 for _, m in seen) and (newest, 12) not in seen:
            problems.append((name, None, "no_latest_year", str(newest)))
    # each other: the results and the segment table of one filing carry the same periods
    by_name = {t.get("statement"): sorted(str((c.get("period_end"), c.get("months")))
                                         for c in t.get("columns") or []) for t in tables}
    if "results" in by_name and "segments" in by_name and by_name["results"] != by_name["segments"]:
        problems.append(("segments", None, "segment_periods", ""))
    return problems


def filing_quarter(tables_a: Sequence[Mapping], tables_b: Sequence[Mapping],
                   today: datetime.date) -> Optional[datetime.date]:
    """The quarter a filing is for, when nobody said: the latest three-month
    column end of the results table, if both reads give the same and it is
    not after today. None when the reads disagree, so the periods are amber
    until the analyst names the quarter."""
    def latest(tables):
        ends = [column_date(c) for t in tables if t.get("statement") == "results"
                for c in t.get("columns") or [] if c.get("months") == 3]
        ends = [e for e in ends if e is not None]
        return max(ends) if ends else None

    a, b = latest(tables_a), latest(tables_b)
    if a is None or (tables_b and a != b) or a > today:
        return None
    return a


# ------------------------------------------------------------- statements

def _row_of(table: Mapping, line: str) -> Optional[int]:
    for index, row in enumerate(table.get("rows") or []):
        if row.get("line") == line:
            return index
    return None


def _close(a: float, b: float, unit: float) -> bool:
    return abs(a - b) <= unit * 2.01 + 1e-9 * max(abs(a), abs(b))


def balance(table: Mapping) -> list[tuple]:
    """A balance sheet's columns where total assets differ from total equity
    and liabilities: ``[(column, assets row, liabilities row)]``."""
    rows = grid(table)
    assets, liabilities = _row_of(table, "total_assets"), _row_of(table, "total_equity_and_liabilities")
    if assets is None or liabilities is None:
        return []
    unit, out = step(rows), []
    for column in range(len(table.get("columns") or [])):
        x, y = rows[assets][column], rows[liabilities][column]
        if x is not None and y is not None and not _close(x, y, unit):
            out.append((column, assets, liabilities))
    return out


def cash(table: Mapping) -> list[tuple]:
    """A cash flow's columns where the closing cash is not the opening cash
    plus the change, nor the change the three activities (with an exchange
    difference row where the statement prints one): ``[(column, rows)]``."""
    rows = grid(table)
    ids = {line: _row_of(table, line) for line in (
        "cash_from_operations", "cash_from_investing", "cash_from_financing",
        "net_change_in_cash", "opening_cash", "closing_cash")}
    unit, out = step(rows), []
    known = {i for i in ids.values() if i is not None}

    def holds(target: float, worked: float, involved: list, column: int) -> bool:
        """Whether ``worked`` reaches ``target``, by itself or with one or
        two of the unnamed rows printed among the lines involved (an
        exchange difference, cash of a business bought or sold, overdrafts
        netted off: statements print these between the lines, each its own
        way)."""
        if _close(worked, target, unit):
            return True
        lo, hi = min(involved), max(involved)
        between = [zero(rows[i][column]) for i in range(lo, hi + 1) if i not in known and zero(rows[i][column])]
        return any(_close(worked + x, target, unit) for x in between) or any(
            _close(worked + x + y, target, unit) for k, x in enumerate(between) for y in between[k + 1:])

    for column in range(len(table.get("columns") or [])):
        def value(line):
            row = ids[line]
            return None if row is None else rows[row][column]

        change, opening, closing = value("net_change_in_cash"), value("opening_cash"), value("closing_cash")
        if change is not None and opening is not None and closing is not None:
            involved = [ids["net_change_in_cash"], ids["opening_cash"], ids["closing_cash"]]
            if not holds(closing, opening + change, involved, column):
                out.append((column, involved))
                continue
        parts = [value(k) for k in ("cash_from_operations", "cash_from_investing", "cash_from_financing")]
        if change is not None and None not in parts:
            involved = [ids["net_change_in_cash"], ids["cash_from_operations"],
                        ids["cash_from_investing"], ids["cash_from_financing"]]
            if not holds(change, sum(parts), involved, column):
                out.append((column, involved))
    return out


def against_stored(table: Mapping, scope: str, stored: Sequence[Mapping]) -> list[tuple]:
    """This document's figures that differ from what is already stored for
    the same statement, scope, standard line and period (a comparative
    column restated, or a misread): ``[(row, column, stored value)]``.
    ``stored`` holds ``{statement, scope, line, period_end, months, value}``
    in this document's unit; only standard lines are compared."""
    rows = grid(table)
    unit = step(rows)
    index = {(s.get("statement"), s.get("scope"), s.get("line"), str(s.get("period_end")), s.get("months")):
             number(s.get("value")) for s in stored}
    out = []
    for r, row in enumerate(table.get("rows") or []):
        line = row.get("line")
        if not line:
            continue
        for c, column in enumerate(table.get("columns") or []):
            key = (table.get("statement"), scope, line, str(column.get("period_end")), column.get("months"))
            known, value = index.get(key), rows[r][c]
            if known is not None and value is not None and not _close(known, value, unit):
                out.append((r, c, known))
    return out
