"""What the reader asks the AI (PLAN.md R1).

One request reads every statement of a filing (R0 lesson 7: the free
allowance counts requests). Read A is given the whole PDF; read B pictures
of the pages read A found the tables on, by another model (R0 lesson 1: a
wrong read can still add up, so the second read must differ in what it is
given and in the model).

``PROMPT_VERSION`` names the wording. ``r0`` is R0's (``statements.py``),
which the 30 development filings' recorded answers were read with;
``r1`` adds one thing, the page each row is printed on, so a figure's place
on the page can be shown beside it (R2). **Changing the wording is a new
version**, and the held-out filings must be read again with it before its
card can say anything (ml/evaluation/reader.py).
"""
from __future__ import annotations

PROMPT_VERSION = "r1"
VERSIONS = ("r0", "r1")

_INTRO = "This is the quarterly results filing of an Indian listed company."
_INTRO_PICTURES = ("These are pictures of the pages of the quarterly results filing of an Indian\n"
                   "listed company that hold its financial statement tables. Number the pictures\n"
                   "1, 2, ... in 'pages'.")

_R0 = """This is the quarterly results filing of an Indian listed company.

Read every CONSOLIDATED financial statement table in it:
- "results": the statement of financial results (profit and loss, with columns
  for quarters and years), down to earnings per share;
- "segments": the segment revenue, results, assets and liabilities table;
- "balance_sheet": the statement of assets and liabilities (usually only in
  half-year and year-end filings);
- "cash_flow": the statement of cash flows (likewise).
Leave out a statement the filing does not have. If the filing has no
consolidated statements at all, read the standalone ones. If one table holds
standalone and consolidated columns side by side, take only the consolidated
columns. Never mix the two.

Answer with JSON only, in exactly this shape:
{
 "company": str,
 "currency": str,            // ISO code, e.g. "INR"
 "unit": "lakhs" | "crores" | "millions" | "thousands" | "units",
 "tables": [{
   "statement": "results" | "segments" | "balance_sheet" | "cash_flow",
   "scope": "consolidated" | "standalone",
   "pages": [int],           // every 1-based PDF page the table runs over, in order
   "columns": [{"label": str, "period_end": "YYYY-MM-DD", "months": int, "audited": bool}],
   "rows": [{"label": str, "kind": str, "values": [number | null]}]   // one value per column
 }]
}

"kind" is "other" except for these lines, each used at most once per table:
- results: "revenue_from_operations", "other_income", "total_income",
  "total_expenses", "profit_before_tax" (the last profit before tax line),
  "tax_total" (total tax expense), "net_profit" (profit for the period after
  tax, before other comprehensive income);
- balance_sheet: "total_assets", "total_equity", "total_equity_and_liabilities",
  "cash_and_cash_equivalents";
- cash_flow: "cash_from_operations", "cash_from_investing",
  "cash_from_financing", "net_change_in_cash", "opening_cash", "closing_cash";
- segments: "segment_revenue_total" (total segment revenue before
  inter-segment elimination), "segment_result_total".

Rules:
- Every row of each table, in order, sub-rows and headings too.
- Copy each number exactly as printed. No rounding, no unit conversion.
- A number in brackets is negative. A dash or "Nil" is 0. An empty cell is null.
- For "months" give the length of the period the column covers (3, 6, 9, 12);
  for a balance sheet column give 0.
- Read the digits from the page itself. If a digit is unclear, give your best
  reading; do not skip the row.
"""

_R1 = _R0.replace(
    '"rows": [{"label": str, "kind": str, "values": [number | null]}]   // one value per column',
    '"rows": [{"label": str, "kind": str, "page": int, "values": [number | null]}]\n'
    '                             // one value per column; "page" is the 1-based page the row is on')


def text(version: str = PROMPT_VERSION, *, pictures: bool = False) -> str:
    """The prompt of a version, for the whole PDF or for pictures of pages."""
    if version not in VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}")
    wording = _R0 if version == "r0" else _R1
    return wording.replace(_INTRO, _INTRO_PICTURES) if pictures else wording
