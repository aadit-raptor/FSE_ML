"""Quarterly actuals feeding a yearly model (PLAN.md R4): the past columns of
a sector model filled from the periods the analyst accepted in coverage (R2).

How far into its fiscal year the newest filing reaches comes from the fiscal
year end the reader stored with it (``year_end_month``), else from the
longest column it prints ending at its own date.

The model's year in progress is **the quarters reported plus the rest
assumed**. Its reported part (``ytd``) is read from the newest accepted
filing's year-to-date column (a half year's filing prints the six months);
where a filing prints only its quarter, the quarters accepted for the same
fiscal year are added up, and the part is left blank if one is missing. Last
year (``ly``) is the twelve-month column ending the day before the year in
progress starts, from whichever accepted filing prints it. Balance sheet
lines come from the balance sheet at those two dates, when a filing carried
one (a first or third quarter's does not).

Only lines the reader matches to a standard line are used, so the fill is
partial by design: the drivers (beds, occupancy, price) and anything the
filing does not split (interest on leases inside finance costs) stay for the
analyst to type, and the answer says which rows it filled and from where.
Money is converted to the newest filing's unit; a filing in another currency
is left out.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

from core.money import UNIT_SCALE

SCALE = {"units": 1.0, **UNIT_SCALE}

# model row -> the standard lines it adds up (research/reader/lines.py)
FLOWS = {
    "rev": ("revenue_from_operations",),
    "mat": ("cost_of_materials", "purchases_stock_in_trade", "changes_in_inventories"),
    "emp": ("employee_benefits",),
    "oth": ("other_expenses",),
    "dep": ("depreciation_amortisation",),
    "othinc": ("other_income",),
    "int_debt": ("finance_costs",),
    "tax": ("tax_total",),
    "r_pbt": ("profit_before_tax",),
    "r_pat": ("net_profit",),
}
TAX_PARTS = ("current_tax", "deferred_tax")         # added up when the filing prints no total
STOCKS = {
    "inv": ("inventories",),
    "rec": ("trade_receivables",),
    "cash": ("cash_and_cash_equivalents",),
    "pay": ("trade_payables",),
}
UNIT_WORDS = {"units": "", "thousands": "thousand", "lakhs": "lakh", "millions": "million", "crores": "crore",
              "billions": "billion"}

# Why a part was left blank (words: web/messages/en.json research.fill_*)
NOTHING_ACCEPTED = "nothing_accepted"
YEAR_COMPLETE = "year_complete"
QUARTER_MISSING = "quarter_missing"
NO_LAST_YEAR = "no_last_year"
NO_BALANCE_SHEET = "no_balance_sheet"
LEASES_IN_FINANCE_COSTS = "leases_in_finance_costs"
OTHER_CURRENCY = "other_currency"


@dataclass
class Fill:
    values: dict = field(default_factory=dict)          # row -> {"ly"|"ytd": number}
    sources: dict = field(default_factory=dict)         # row -> {"ly"|"ytd": "page 3"}
    labels: dict = field(default_factory=dict)          # "ly"/"ytd" -> the period in words
    unit: str | None = None
    currency: str | None = None
    unit_rupees: float | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"values": self.values, "sources": self.sources, "labels": self.labels, "unit": self.unit,
                "currency": self.currency, "unit_rupees": self.unit_rupees, "notes": self.notes}


def _months_back(day: date, months: int) -> date:
    """The last day of the month `months` before the month `day` ends."""
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    return date(year, month + 1, calendar.monthrange(year, month + 1)[1])


def _day(text) -> date | None:
    try:
        return date.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def _columns(periods: list, statement: str):
    """(period, table, column index, end, months) for every column of `statement`, newest filing first."""
    for period in periods:
        for table in period["figures"].get("tables") or []:
            if table.get("statement") != statement:
                continue
            for c, column in enumerate(table.get("columns") or []):
                end = _day(column.get("period_end"))
                if end is not None:
                    yield period, table, c, end, column.get("months")


def _near(a: date, b: date) -> bool:
    return abs((a - b).days) <= 7            # a 52/53-week year ends within a week of its month end


def _sum(table: dict, column: int, lines: tuple) -> tuple[float | None, int | None]:
    total, page, found = 0.0, None, False
    for row in table.get("rows") or []:
        if row.get("line") in lines:
            values = row.get("values") or []
            value = values[column] if column < len(values) else None
            if value is not None:
                total += value
                found = True
                page = page or row.get("page")
    return (total if found else None), page


def _flows(table: dict, column: int) -> dict:
    out = {}
    for key, lines in FLOWS.items():
        value, page = _sum(table, column, lines)
        if value is None and key == "tax":
            value, page = _sum(table, column, TAX_PARTS)
        if value is not None:
            out[key] = (value, page)
    return out


def _stocks(periods: list, end: date) -> dict | None:
    for period, table, c, at, _ in _columns(periods, "balance_sheet"):
        if _near(at, end):
            out = {}
            for key, lines in STOCKS.items():
                value, page = _sum(table, c, lines)
                if value is not None:
                    out[key] = (value * period["scale"], page)
            return out
    return None


def fill(stored: list, scope: str = "consolidated") -> Fill:
    """The model's past (``ly`` and ``ytd``) from a company's accepted periods
    (`db.coverage.company_figures`), for one scope."""
    out = Fill()
    periods = [p for p in stored if p.get("scope") == scope and p.get("figures")]
    if not periods:
        out.notes.append(NOTHING_ACCEPTED)
        return out
    periods.sort(key=lambda p: max((end for *_, end, _ in _columns([p], "results")), default=date.min), reverse=True)
    newest = periods[0]
    out.unit, out.currency = newest.get("unit"), newest.get("currency")
    kept = []
    for p in periods:
        if p.get("currency") and out.currency and p["currency"] != out.currency:
            if OTHER_CURRENCY not in out.notes:
                out.notes.append(OTHER_CURRENCY)
            continue
        kept.append({**p, "scale": SCALE[p["unit"]] / SCALE[out.unit]})
    periods = kept
    out.unit_rupees = SCALE[out.unit]

    latest = max(end for *_, end, _ in _columns(periods[:1], "results"))
    year_end_month = newest["figures"].get("year_end_month")
    if isinstance(year_end_month, int) and 1 <= year_end_month <= 12:
        months = (latest.month - year_end_month) % 12 or 12      # how far into its fiscal year the filing reaches
    else:                                                        # not known: the longest column ending there says
        months = max((m for *_, end, m in _columns(periods[:1], "results")
                      if end == latest and isinstance(m, int) and 0 < m <= 12), default=None)
    if months is None:
        out.notes.append(NOTHING_ACCEPTED)
        return out
    if months == 12:                         # the year is reported whole: it is last year, and nothing is to date
        out.notes.append(YEAR_COMPLETE)
        start = latest + timedelta(days=1)
    else:
        start = _months_back(latest, months) + timedelta(days=1)
        part = _year_to_date(periods, latest, months, start)
        if part is None:
            out.notes.append(QUARTER_MISSING)
        else:
            _put(out, "ytd", part, (latest - start).days + 1)
            out.labels["ytd"] = "%d months to %s" % (months, latest.isoformat())
            stocks = _stocks(periods, latest)
            if stocks:
                _put(out, "ytd", stocks)
    year_end = start - timedelta(days=1)
    year = next(((p, t, c) for p, t, c, end, m in _columns(periods, "results") if m == 12 and _near(end, year_end)), None)
    if year is None:
        out.notes.append(NO_LAST_YEAR)
    else:
        p, t, c = year
        _put(out, "ly", {k: (v * p["scale"], page) for k, (v, page) in _flows(t, c).items()},
             (year_end - _months_back(year_end, 12)).days)
        out.labels["ly"] = "12 months to %s" % year_end.isoformat()
        stocks = _stocks(periods, year_end)
        if stocks:
            _put(out, "ly", stocks)
        else:
            out.notes.append(NO_BALANCE_SHEET)
    if "int_debt" in out.values:
        out.notes.append(LEASES_IN_FINANCE_COSTS)
    return out


def _year_to_date(periods: list, latest: date, months: int, start: date) -> dict | None:
    """The flows from the start of the year to `latest`: a column printing exactly that, else its quarters added."""
    for p, t, c, end, m in _columns(periods, "results"):
        if end == latest and m == months:
            return {k: (v * p["scale"], page) for k, (v, page) in _flows(t, c).items()}
    total: dict = {}
    for q in range(months // 3):
        quarter_end = _months_back(latest, months - 3 * (q + 1))
        found = next(((p, t, c) for p, t, c, end, m in _columns(periods, "results") if m == 3 and _near(end, quarter_end)),
                     None)
        if found is None:
            return None
        p, t, c = found
        for key, (value, page) in _flows(t, c).items():
            was, first = total.get(key, (0.0, page))
            total[key] = (was + value * p["scale"], first)
    return total


def _put(out: Fill, period: str, figures: dict, days: int | None = None) -> None:
    for key, (value, page) in figures.items():
        out.values.setdefault(key, {})[period] = round(value, 6)
        out.sources.setdefault(key, {})[period] = page
    if days is not None:
        out.values.setdefault("days", {})[period] = days
