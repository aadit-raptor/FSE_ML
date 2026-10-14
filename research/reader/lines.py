"""The company's own row labels matched to standard lines (PLAN.md R1).

A results filing names its lines its own way ("Revenue from operations",
"a) Sales / Income from operations", "Turnover"). Each statement here has a
**fixed list** of standard lines with the wordings Ind AS filings use
(Schedule III and SEBI's results format), tried first; a row the list does
not place keeps the line the AI gave it (``kind`` in the read), **for the
few lines the prompt asks for** and nothing else; the company's own wording
is always kept beside it (``label``). A line is given to at most one row a
table: the first row that matches.

``line_source`` says which did it: ``list``, ``ai`` or None (no standard
line: most rows, which keep only their own label).
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

# The lines the prompt asks the AI to mark (research/reader/prompt.py), per statement
AI_LINES = {
    "results": ("revenue_from_operations", "other_income", "total_income", "total_expenses",
                "profit_before_tax", "tax_total", "net_profit"),
    "balance_sheet": ("total_assets", "total_equity", "total_equity_and_liabilities",
                      "cash_and_cash_equivalents"),
    "cash_flow": ("cash_from_operations", "cash_from_investing", "cash_from_financing",
                  "net_change_in_cash", "opening_cash", "closing_cash"),
    "segments": ("segment_revenue_total", "segment_result_total"),
}

# (line, pattern on the cleaned label). Order matters: a narrower wording before a wider one.
LISTS: dict[str, tuple[tuple[str, str], ...]] = {
    "results": (
        ("revenue_from_operations", r"^(revenue|income|sales)( from| of)? operations?\b|^net sales\b|^turnover\b"),
        ("other_income", r"^other (operating )?income\b"),
        ("total_income", r"^total (revenue|income)\b(?!.*comprehensive)"),
        ("cost_of_materials", r"^cost of (raw )?materials? consumed\b"),
        ("purchases_stock_in_trade", r"^purchases? of (stock|traded goods)"),
        ("changes_in_inventories", r"^changes? in inventor"),
        ("employee_benefits", r"^employee(s)? benefits? expenses?\b|^employee (costs?|expenses)\b"),
        ("finance_costs", r"^finance (costs?|charges)\b"),
        ("depreciation_amortisation", r"^depreciation\b"),
        ("other_expenses", r"^other expenses\b"),
        ("total_expenses", r"^total expenses\b"),
        ("exceptional_items", r"^exceptional items?\b"),
        ("current_tax", r"^current tax\b"),
        ("deferred_tax", r"^deferred tax\b"),
        ("tax_total", r"^total tax( expense)?\b|^tax expense\b"),
        ("total_comprehensive_income", r"^total comprehensive income\b"),
        ("other_comprehensive_income", r"^(total )?other comprehensive income\b"),
        ("paid_up_equity", r"^paid[- ]up equity share capital\b|^equity share capital\b"),
        ("other_equity", r"^other equity\b|^reserves (excluding|excl)"),
        ("eps_basic", r"^(\(?a\)? )?basic\b"),
        ("eps_diluted", r"^(\(?b\)? )?diluted\b"),
    ),
    "balance_sheet": (
        ("total_non_current_assets", r"^total non[- ]current assets\b"),
        ("total_current_assets", r"^total current assets\b"),
        ("total_assets", r"^total assets\b"),
        ("equity_share_capital", r"^(equity )?share capital\b"),
        ("other_equity", r"^other equity\b"),
        ("total_equity_and_liabilities", r"^total equity and liabilities\b"),
        ("total_equity", r"^total equity\b"),
        ("total_non_current_liabilities", r"^total non[- ]current liabilities\b"),
        ("total_current_liabilities", r"^total current liabilities\b"),
        ("inventories", r"^inventories\b"),
        ("trade_receivables", r"^trade receivables\b"),
        ("cash_and_cash_equivalents", r"^cash and cash equivalents\b"),
        ("trade_payables", r"^trade payables\b"),
    ),
    "cash_flow": (
        ("cash_from_operations", r"^net cash (generated |inflow |flows? )?(from|used in|\(used in\)|generated from)"
                                 r"( / \(used in\))? operating"),
        ("cash_from_investing", r"^net cash .*investing activities"),
        ("cash_from_financing", r"^net cash .*financing activities"),
        ("net_change_in_cash", r"^net (increase|decrease|change)( / \(decrease\)| \(decrease\))? in cash"),
        ("opening_cash", r"^cash and cash equivalents at (the )?(beginning|start)"),
        ("closing_cash", r"^cash and cash equivalents at (the )?end"),
    ),
    "segments": (),
}

_NUMBERING = re.compile(r"^\s*(\(?([0-9]{1,2}|[a-z]|[ivx]{1,5})[).:]\s+|[0-9]{1,2}\s+(?=[a-z]))", re.I)


def clean(label: str) -> str:
    """A label in lower case, without its numbering ("a)", "1.", "(ii)") or
    a leading "less:" and with single spaces."""
    text = " ".join(str(label or "").split()).lower()
    for _ in range(2):
        text = _NUMBERING.sub("", text)
        text = re.sub(r"^(less|add)\s*[:\-]\s*", "", text)
    return text.strip()


def match(statement: str, label: str) -> Optional[str]:
    """The standard line the fixed list gives a label, or None."""
    text = clean(label)
    for line, pattern in LISTS.get(statement, ()):
        if re.search(pattern, text):
            return line
    return None


def assign(statement: str, rows: Sequence[Mapping]) -> list[tuple[Optional[str], Optional[str]]]:
    """``(line, source)`` for each row of a table: the fixed list first, the
    AI's ``kind`` for the rest (only the lines the prompt asks for), each
    line once."""
    used: set[str] = set()
    out: list[tuple[Optional[str], Optional[str]]] = [(None, None)] * len(rows)
    for index, row in enumerate(rows):
        line = match(statement, str(row.get("label") or ""))
        if line and line not in used and any(v is not None for v in row.get("values") or []):
            used.add(line)
            out[index] = (line, "list")
    allowed = AI_LINES.get(statement, ())
    for index, row in enumerate(rows):
        kind = row.get("kind")
        if out[index][0] is None and kind in allowed and kind not in used:
            used.add(kind)
            out[index] = (kind, "ai")
    return out
