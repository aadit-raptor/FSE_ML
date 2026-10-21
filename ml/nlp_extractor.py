"""A document, any language, read for a deal (PLAN.md 6.2).

An annual report or an information pack (a PDF the person uploaded, scanned clean: PLAN.md 6.1) is read
**twice by two different Gemini models** through the results reader's provider switch
(``research/reader/providers.py``; the user's decision of 2026-10-21): read A is given the whole PDF, read B
pictures of the pages read A cited. Each read answers what the document says it is counted in (**currency,
unit, accounting standard, fiscal year end, consolidated or not**) and a short list of figures for the latest
year and the one before (``FIGURES``), each with **the page it is on and a verbatim quote** holding it.

``assemble`` makes the two reads one list for the person to review. A value is **amber** when

- the reads give it differently (``reads_differ``), or only one read gives it (``one_read_only``);
- its quote is not on the page it cites, where that page has text to look it up in (``quote_not_on_page``):
  a model that invents a quote, or cites the wrong page, is caught here;
- its quote does not hold its number (``number_not_in_quote``);
- its page is a poor scan (``poor_scan``, the reader's rule: under 100 dpi).

``deal_inputs`` turns **only the values the person accepted** into a deal's inputs, by the engine's own
definitions (EBIT = gross profit - opex; EBITDA = EBIT + D&A, unless the document prints an EBITDA); an input
whose figures are not all accepted is left out and says which it needs, and money needs its currency and unit
accepted too. Nothing here calls a provider or opens a file: ``ml/document_reader.py`` is the job.

The prompt's wording is ``PROMPT_VERSION``: a change is a new version, and the evaluation set must be read again
with it before the card (``ml/evaluation/extractor.py``) says anything.
"""
from __future__ import annotations

import datetime
import re
import unicodedata
from typing import Any, Callable, Mapping, Optional, Sequence

from research.reader import checks

PROMPT_VERSION = "x1"
MAX_QUOTE = 160
MAX_FIGURES = 60
MAX_PICTURES = 12

# What each read is asked for, in this order (the prompt lists them with what they mean)
FIGURES = ("revenue", "gross_profit", "operating_income", "depreciation_amortization", "ebitda",
           "interest_expense", "profit_before_tax", "income_tax", "net_income", "capital_expenditures",
           "cash_from_operations", "cash_and_equivalents", "total_borrowings", "total_assets", "total_equity")
LABELS = ("currency", "unit", "accounting_standard", "fiscal_year_end", "scope")
UNITS = ("units", "thousands", "lakhs", "millions", "crores", "billions")
STANDARDS = ("ifrs", "us_gaap", "ind_as", "jgaap", "uk_gaap", "hgb", "local", "other")
SCOPES = ("consolidated", "standalone")
DOCUMENTS = ("annual_report", "information_pack", "results", "financial_statements", "other")
DEAL_STANDARDS = ("ifrs", "us_gaap")             # what a deal's accounting_standard takes (PLAN.md 2.6)
AMBER = ("reads_differ", "one_read_only", "quote_not_on_page", "number_not_in_quote", "poor_scan")
UNIT_SCALE = {"units": 1.0, "thousands": 1e3, "lakhs": 1e5, "millions": 1e6, "crores": 1e7, "billions": 1e9}

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_LANGUAGE = re.compile(r"^[a-z]{2}$")

_MEANINGS = {
    "revenue": "revenue / turnover / net sales (the top line)",
    "gross_profit": "gross profit, only where the income statement prints one",
    "operating_income": "operating profit (EBIT, result from operating activities)",
    "depreciation_amortization": "depreciation, amortisation and impairment of the year (income statement or "
                                 "cash flow statement)",
    "ebitda": "EBITDA, only where the document prints a figure called EBITDA",
    "interest_expense": "finance costs / interest expense",
    "profit_before_tax": "profit before tax",
    "income_tax": "income tax expense (a tax credit is negative)",
    "net_income": "profit for the year, before other comprehensive income",
    "capital_expenditures": "purchases of property, plant and equipment and intangible assets (cash flow "
                            "statement), as a positive amount",
    "cash_from_operations": "net cash from operating activities",
    "cash_and_equivalents": "cash and cash equivalents at the year end (balance sheet)",
    "total_borrowings": "total borrowings / financial debt, current and non-current, where the balance sheet or "
                        "a note prints a total",
    "total_assets": "total assets",
    "total_equity": "total equity",
}

_PROMPT = """This is a company document: an annual report, an information pack or a set of financial statements.
It may be in any language. Read its CONSOLIDATED primary financial statements (or the standalone ones if it has
no consolidated statements), for the latest financial year and the year before it.

Answer with JSON only, in exactly this shape:
{
 "company": str,
 "language": str,              // ISO 639-1 code of the document's language, e.g. "de", "ja", "en"
 "document": "annual_report" | "information_pack" | "results" | "financial_statements" | "other",
 "labels": {
   "currency": {"value": str, "page": int, "quote": str},       // ISO 4217 code of the statements, e.g. "EUR"
   "unit": {"value": "units" | "thousands" | "lakhs" | "millions" | "crores" | "billions", "page": int, "quote": str},
   "accounting_standard": {"value": "ifrs" | "us_gaap" | "ind_as" | "jgaap" | "uk_gaap" | "hgb" | "local" | "other",
                           "page": int, "quote": str},
   "fiscal_year_end": {"value": "YYYY-MM-DD", "page": int, "quote": str},   // the latest year's last day
   "scope": {"value": "consolidated" | "standalone", "page": int, "quote": str}
 },
 "figures": [{"key": str, "period_end": "YYYY-MM-DD", "value": number, "page": int, "quote": str}]
}

"key" is one of these, each at most once per year:
%s

Rules:
- "page" is the 1-based page of the PDF file (count from its first page), not the number printed on the page.
- "quote" is copied verbatim from that page, in the document's own language and script: the row's own words
  and the number exactly as printed, at most 120 characters. For a label, the words that say it ("in EUR
  million", "単位：百万円", "Konzernabschluss nach IFRS").
- "value" is the number as printed in the statements' unit: no rounding, no conversion. A number in brackets
  or with a minus sign is negative. Leave out a figure the document does not print; never work one out.
- Every figure from the same unit as "unit". Give "unit" "units" when amounts are printed in full.
"""

_PICTURES = """These are pictures of pages of a company document (an annual report, an information pack or a set of
financial statements), which may be in any language. Number the pictures 1, 2, ... and give that number as
"page". Read what they show of its CONSOLIDATED primary financial statements (or the standalone ones if there
are no consolidated statements), for the latest financial year and the year before it.
"""


class Unreadable(ValueError):
    """The answer is not the shape asked for."""


def prompt_text(*, pictures: bool = False) -> str:
    keys = "\n".join(f'- "{k}": {_MEANINGS[k]}' for k in FIGURES)
    text = _PROMPT % keys
    if pictures:
        first = text.index("\nAnswer with JSON")
        text = _PICTURES + text[first:].replace(
            '- "page" is the 1-based page of the PDF file (count from its first page), not the number printed on '
            'the page.', '- "page" is the number of the picture the figure is on (1, 2, ...).')
    return text


# ---------------------------------------------------------------- the answer

def _text(value: Any, limit: int = MAX_QUOTE) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def _page(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _label(key: str, found: Any) -> dict:
    found = found if isinstance(found, dict) else {}
    raw = found.get("value")
    value = None
    if isinstance(raw, str):
        if key == "currency":
            value = raw.strip().upper() if _CURRENCY.match(raw.strip().upper()) and len(raw.strip()) == 3 else None
        elif key == "unit":
            value = raw if raw in UNITS else None
        elif key == "accounting_standard":
            value = raw if raw in STANDARDS else None
        elif key == "fiscal_year_end":
            value = raw if _DATE.match(raw) and _day(raw) else None
        elif key == "scope":
            value = raw if raw in SCOPES else None
    return {"value": value, "page": _page(found.get("page")), "quote": _text(found.get("quote"))}


def _day(text: str) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def unwrap(answer: Any) -> dict:
    """The answer as the one object asked for, cut to sane sizes; ``Unreadable`` otherwise."""
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    if not isinstance(answer, dict) or not isinstance(answer.get("figures"), list) \
            or not isinstance(answer.get("labels"), dict):
        raise Unreadable("not the object asked for")
    figures, seen = [], set()
    for found in answer["figures"][:MAX_FIGURES * 2]:
        if not isinstance(found, dict) or found.get("key") not in FIGURES:
            continue
        period = found.get("period_end")
        if not isinstance(period, str) or not _DATE.match(period) or not _day(period):
            continue
        if (found["key"], period) in seen:
            continue                                  # a repeated figure: the first is the one read
        seen.add((found["key"], period))
        figures.append({"key": found["key"], "period_end": period, "value": checks.number(found.get("value")),
                        "page": _page(found.get("page")), "quote": _text(found.get("quote"))})
    language = _text(answer.get("language"), 8).lower()
    return {"company": _text(answer.get("company"), 200),
            "language": language if _LANGUAGE.match(language) else "",
            "document": answer.get("document") if answer.get("document") in DOCUMENTS else "other",
            "labels": {k: _label(k, answer["labels"].get(k)) for k in LABELS},
            "figures": figures[:MAX_FIGURES]}


def cited_pages(read: Mapping) -> list[int]:
    """What read B is shown: the pages read A cites figures on, then those of the labels."""
    figures = sorted({f["page"] for f in read["figures"] if f["page"]})
    labels = sorted({v["page"] for v in read["labels"].values() if v["page"]} - set(figures))
    return (figures + labels)[:MAX_PICTURES]


def picture_pages(read: dict, sent: Sequence[int]) -> dict:
    """Read B numbers its pictures 1, 2 ...: give them back their PDF pages."""
    def page(n):
        return sent[n - 1] if isinstance(n, int) and 1 <= n <= len(sent) else None

    for found in read["figures"]:
        found["page"] = page(found["page"])
    for found in read["labels"].values():
        found["page"] = page(found["page"])
    return read


# ---------------------------------------------------------------- the checks

def _plain(text: str) -> str:
    """For comparing a quote with its page: full-width forms folded, case and every space gone."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.replace("−", "-").replace("–", "-").replace("—", "-")
    return "".join(ch for ch in folded if not ch.isspace())


def quote_on_text(quote: str, text: str) -> bool:
    plain = _plain(quote)
    return bool(plain) and plain in _plain(text)


_NUMBER = re.compile(r"\d(?:[\d.,']*\d)?")
_SPACED = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?")


def _readings(token: str) -> set[float]:
    """What a printed number can mean: its separators as grouping, or the last one as the decimal point."""
    plain = re.sub(r"[   ]", "", token)
    found = {float(re.sub(r"[.,']", "", plain))}
    last = max(plain.rfind("."), plain.rfind(","))
    if last > 0:
        whole, fraction = re.sub(r"[.,']", "", plain[:last]), plain[last + 1:]
        if fraction.isdigit():
            found.add(float(f"{whole}.{fraction}"))
    return found


def number_in_quote(value: Optional[float], quote: str) -> bool:
    """Whether the quote prints the value, however it groups, points and signs it (1.627,9; 1,627.9; 1 627,9;
    (1,234); full-width digits)."""
    if value is None or not quote:
        return False
    text = unicodedata.normalize("NFKC", quote)
    magnitude = abs(value)
    tokens = _NUMBER.findall(text) + _SPACED.findall(text)
    return any(abs(reading - magnitude) <= 1e-9 * max(1.0, magnitude)
               for token in tokens for reading in _readings(token))


def _same(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _amber(value, other_given: bool, other_value, page, quote, *, number: bool,
           quote_on_page: Callable[[int, str], Optional[bool]], poor: Callable[[int], bool]) -> Optional[str]:
    if not other_given:
        return "one_read_only"
    if (not _same(value, other_value)) if number else value != other_value:
        return "reads_differ"
    if page is not None and quote and quote_on_page(page, quote) is False:
        return "quote_not_on_page"
    if number and not number_in_quote(value, quote):
        return "number_not_in_quote"
    if page is not None and poor(page):
        return "poor_scan"
    return None


def assemble(read_a: Mapping, read_b: Optional[Mapping], *,
             quote_on_page: Callable[[int, str], Optional[bool]], poor: Callable[[int], bool]) -> dict:
    """Read A's values with read B's beside them and each one's amber reason (``None`` when it needs no look).
    ``quote_on_page(page, quote)``: True or False where the page has text, None where it has none."""
    labels = []
    for key in LABELS:
        mine = read_a["labels"][key]
        if mine["value"] is None:
            continue
        theirs = (read_b or {}).get("labels", {}).get(key) or {}
        given = theirs.get("value") is not None
        labels.append({"id": f"label:{key}", "key": key, "value": mine["value"], "page": mine["page"],
                       "quote": mine["quote"], "other": theirs.get("value") if given else None,
                       "amber": _amber(mine["value"], given, theirs.get("value"), mine["page"], mine["quote"],
                                       number=False, quote_on_page=quote_on_page, poor=poor)})
    partner = {(f["key"], f["period_end"]): f for f in (read_b or {}).get("figures", [])}
    figures = []
    for found in read_a["figures"]:
        if found["value"] is None:
            continue
        theirs = partner.get((found["key"], found["period_end"]))
        given = theirs is not None and theirs["value"] is not None
        figures.append({"id": f"figure:{found['key']}:{found['period_end']}", "key": found["key"],
                        "period_end": found["period_end"], "value": found["value"], "page": found["page"],
                        "quote": found["quote"], "other": theirs["value"] if given else None,
                        "amber": _amber(found["value"], given, theirs["value"] if given else None, found["page"],
                                        found["quote"], number=True, quote_on_page=quote_on_page, poor=poor)})
    order = {k: i for i, k in enumerate(FIGURES)}
    figures.sort(key=lambda f: (order[f["key"]], f["period_end"]))
    every = labels + figures
    return {"company": read_a["company"], "language": read_a["language"], "document_kind": read_a["document"],
            "labels": labels, "figures": figures,
            "summary": {"figures": len(figures), "labels": len(labels),
                        "amber": sum(i["amber"] is not None for i in every)}}


# ---------------------------------------------------------------- the deal

def _accepted(result: Mapping, accepted: set) -> tuple[dict, dict]:
    labels = {i["key"]: i for i in result.get("labels", []) if i["id"] in accepted}
    figures: dict = {}
    for found in result.get("figures", []):
        if found["id"] in accepted:
            figures.setdefault(found["key"], {})[found["period_end"]] = found
    return labels, figures


def _latest_year(result: Mapping) -> Optional[str]:
    end = next((i["value"] for i in result.get("labels", []) if i["key"] == "fiscal_year_end"), None)
    if end:
        return end
    periods = sorted({f["period_end"] for f in result.get("figures", [])})
    return periods[-1] if periods else None


def _prior(latest: str, periods) -> Optional[str]:
    day = _day(latest)
    if day is None:
        return None
    earlier = [p for p in periods if (_day(p) or day) < day and 300 <= (day - _day(p)).days <= 430]
    return max(earlier) if earlier else None


def deal_inputs(result: Mapping, accepted) -> dict:
    """The deal inputs the accepted values give (``inputs``), the values each was made from (``used``), the
    inputs left out and the values they need (``left_out``) and ``notes``. Money is in the document's currency
    and unit (a deal unit: amounts printed in full become thousands)."""
    accepted = set(accepted)
    labels, figures = _accepted(result, accepted)
    every = {i["id"] for i in result.get("labels", []) + result.get("figures", [])}
    latest = _latest_year(result)
    inputs: dict = {}
    used: dict = {}
    left_out: list = []
    notes: list = []

    def fig(key: str, period: Optional[str] = None):
        period = period or latest
        return (figures.get(key) or {}).get(period)

    def need(key: str, period: Optional[str] = None) -> str:
        return f"figure:{key}:{period or latest}"

    def offer(name: str, needs: list, make: Callable[[], Optional[float]], *, note: Optional[str] = None):
        if not all(n in every for n in needs):
            return                                  # the document doesn't print what it needs
        if not all(n in accepted for n in needs):
            left_out.append({"input": name, "needs": [n for n in needs if n not in accepted]})
            return
        value = make()
        if value is None:
            if note:
                notes.append(note)
            return
        inputs[name], used[name] = value, needs

    unit = (labels.get("unit") or {}).get("value")
    money_needs = ["label:currency", "label:unit"]
    if "currency" in labels and unit:
        inputs["currency"] = labels["currency"]["value"]
        inputs["unit"] = "thousands" if unit == "units" else unit
        used["currency"], used["unit"] = ["label:currency"], ["label:unit"]
    to_deal_unit = 1e-3 if unit == "units" else 1.0

    if latest:
        if need("ebitda") in every:
            offer("ebitda", [need("ebitda")] + money_needs,
                  lambda: _positive(fig("ebitda")["value"] * to_deal_unit), note="ebitda_not_positive")
        else:
            offer("ebitda", [need("operating_income"), need("depreciation_amortization")] + money_needs,
                  lambda: _positive((fig("operating_income")["value"] + abs(fig("depreciation_amortization")["value"]))
                                    * to_deal_unit), note="ebitda_not_positive")
        revenue = need("revenue")

        def share(key: str, *, absolute: bool = False):
            def make():
                top = fig("revenue")["value"]
                if top <= 0:
                    return None
                part = fig(key)["value"]
                return _pct((abs(part) if absolute else part) / top * 100)
            return make

        offer("gross_margin", [revenue, need("gross_profit")], share("gross_profit"))
        offer("opex", [revenue, need("gross_profit"), need("operating_income")],
              lambda: _pct((fig("gross_profit")["value"] - fig("operating_income")["value"])
                           / fig("revenue")["value"] * 100) if fig("revenue")["value"] > 0 else None)
        offer("da", [revenue, need("depreciation_amortization")], share("depreciation_amortization", absolute=True))
        offer("capex", [revenue, need("capital_expenditures")], share("capital_expenditures", absolute=True))
        # many statements print expenses negative, so the tax is taken by its size (a credit is rare in a deal)
        offer("tax", [need("profit_before_tax"), need("income_tax")],
              lambda: _pct(abs(fig("income_tax")["value"]) / fig("profit_before_tax")["value"] * 100)
              if fig("profit_before_tax")["value"] > 0 else None)
        prior = _prior(latest, (figures.get("revenue") or {}).keys() | {
            f["period_end"] for f in result.get("figures", []) if f["key"] == "revenue"})
        if prior:
            offer("growth", [revenue, need("revenue", prior)],
                  lambda: _bounded((fig("revenue")["value"] / fig("revenue", prior)["value"] - 1) * 100, -50, 100)
                  if fig("revenue", prior)["value"] > 0 else None)

    standard = labels.get("accounting_standard")
    if standard:
        if standard["value"] in DEAL_STANDARDS:
            inputs["accounting_standard"], used["accounting_standard"] = standard["value"], [standard["id"]]
        else:
            notes.append("standard_not_in_deal")
    end = labels.get("fiscal_year_end")
    if end and _day(end["value"]):
        day = _day(end["value"])
        # a 52/53-week year ending in a month's first week counts as the month before (PLAN.md 2.3a)
        month = (day.month - 2) % 12 + 1 if day.day <= 7 else day.month
        year = day.year if not (day.day <= 7 and day.month == 1) else day.year - 1
        inputs["fiscal_year_end_month"], inputs["first_fiscal_year"] = month, year + 1
        used["fiscal_year_end_month"] = used["first_fiscal_year"] = [end["id"]]
    return {"inputs": inputs, "used": used, "left_out": left_out, "notes": sorted(set(notes))}


def _pct(value: float) -> Optional[float]:
    return _bounded(value, 0, 100)


def _bounded(value: float, low: float, high: float) -> Optional[float]:
    return value if low <= value <= high else None


def _positive(value: float) -> Optional[float]:
    return value if value > 0 else None
