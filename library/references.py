"""Reference transactions: real buyouts with every figure read from a filing (PLAN.md 4.5b).

A reference transaction is one sponsor-led buyout: its target, country,
sector, sponsors and closing date; its entry figures (what was paid, the
EBITDA, the debt and the equity raised, and the fees where the filings give
them); and its outcome (an exit, distress, or still held). **Every figure
names its source**: the filing (filer, form, date, address), where in it,
and a few of its words where they help a reviewer find it. A figure the
filings give only in pieces (EBITDA as operating income plus D&A, debt as
its facilities) lists the pieces, each with its own source, and must equal
their sum.

A transaction joins the library only through Library -> Review, when two
administrators other than the one who proposed it approve it
(``library/review.py``). The repository proposes the ones in
``reference_deals.json``; an administrator may propose more in the same
format.

**Inclusion rules** (``problems``) refuse a proposal that cannot be checked
or does not describe a buyout: a required figure missing, a source that is
not a filing on a recognised filing host, pieces that do not add up, a
multiple or fee no deal has, an outcome before the closing. **Balance
rules** (``balance``) say where the library is lopsided -- a region, size,
sector, era or outcome holding more than ``BALANCE_SHARE`` of it -- and
which empty buckets a proposal fills; reviewers see them beside the
evidence. Money is in millions of the deal's currency; ``tags`` sizes deals
in US dollars, so a deal in another currency also needs the filing's own
US dollar value.

Nothing in the deal model, the simulation or the risk warnings reads these
deals. Their fees and amortisation become *offered* Settings
(``library/fees.py``), applied only when the user asks.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

from library import base_rates, coverage

DATA = Path(__file__).with_name("reference_deals.json")

# The figures a reference transaction may carry, in the order shown
FIGURES = ("transaction_value", "transaction_value_usd", "ebitda", "debt", "equity",
           "transaction_fees", "financing_fees", "senior_amort_pct")
REQUIRED = ("transaction_value", "ebitda", "debt")
PERCENT_FIGURES = ("senior_amort_pct",)
VALUE_BASES = ("stated", "equity_value", "uses_less_fees", "funds_needed")
EBITDA_BASES = ("reported", "adjusted", "stated", "projection")
KINDS = ("take_private", "carve_out", "recapitalization", "secondary")
OUTCOMES = coverage.OUTCOMES
EVENTS = {
    "success": ("ipo", "sale", "relisted"),
    "distress": ("missed_payment", "bankruptcy", "restructuring"),
    "held": ("held",),
}
# GICS sectors, as the coverage page counts them
SECTORS = ("communication_services", "consumer_discretionary", "consumer_staples", "energy", "financials",
           "health_care", "industrials", "information_technology", "materials", "real_estate", "utilities")

# Hosts that publish filings themselves: securities regulators, company
# registers and exchanges' disclosure systems. A source elsewhere (a news
# report, a database) is refused: a reviewer must be able to open the filing.
FILING_HOSTS = (
    "www.sec.gov",                                          # US SEC EDGAR
    "find-and-update.company-information.service.gov.uk",   # UK Companies House
    "disclosure2.edinet-fsa.go.jp",                         # Japan EDINET
    "filings.xbrl.org",                                     # ESEF reports (EU, UK)
    "www.sedarplus.ca",                                     # Canada SEDAR+
    "www.cninfo.com.cn",                                    # China, the CSRC's disclosure site
    "www.hkexnews.hk",                                      # Hong Kong exchange filings
    "www.bseindia.com", "www.nseindia.com",                 # India's exchanges
    "www.asx.com.au",                                       # Australia's exchange announcements
)

# What a buyout looks like: an entry multiple of EBITDA, the debt against the
# value paid, and fees as a share of what they are charged on
MULTIPLE_RANGE = (2.0, 40.0)
FEE_RANGE_PCT = (0.0, 10.0)
AMORT_RANGE_PCT = (0.0, 100.0)
# Pieces must add up to the figure to within rounding of one decimal
PARTS_TOLERANCE = 0.051

# Balance: no bucket should hold more than this share of the library once it
# has at least BALANCE_MIN deals (below that every bucket is "most")
BALANCE_SHARE = 0.5
BALANCE_MIN = 6
BALANCED = ("region", "size", "sector", "era", "outcome")
# Every finding ``problems`` can report (api/schemas.py RuleCode lists the same, tested)
RULE_CODES = ("no_sponsor", "missing_figure", "unknown_figure", "missing_source", "not_a_filing", "unused_source",
              "parts_dont_add_up", "not_positive", "multiple_out_of_range", "debt_exceeds_value",
              "fee_out_of_range", "amortisation_out_of_range", "closed_in_future", "outcome_before_close",
              "event_doesnt_match_outcome")


@lru_cache(maxsize=1)
def _file() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def repository_deals() -> list[dict]:
    """The transactions the repository proposes (``reference_deals.json``)."""
    return json.loads(json.dumps(_file()["deals"]))


def _canonical(x):
    """Numbers as floats and no empty fields, so the repository's JSON and the
    API's validated copy of the same deal (where 33000 becomes 33000.0) agree."""
    if isinstance(x, dict):
        return {k: _canonical(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_canonical(v) for v in x]
    if isinstance(x, int) and not isinstance(x, bool):
        return float(x)
    return x


def content_hash(deal: dict) -> str:
    """A fingerprint of a proposal's content: the same deal proposed twice,
    from the repository or through the API, is one proposal."""
    text = json.dumps(_canonical(deal), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _figure(deal: dict, name: str) -> Optional[dict]:
    return (deal.get("figures") or {}).get(name)


def value(deal: dict, name: str) -> Optional[float]:
    found = _figure(deal, name)
    return None if found is None else float(found["value"])


def closed_year(deal: dict) -> int:
    return date.fromisoformat(deal["closed"]["date"]).year


def _evidence(deal: dict) -> Iterable[tuple[str, dict]]:
    """Every place a source is named: (what, the cited object)."""
    yield "closed", deal["closed"]
    yield "outcome", deal["outcome"]
    for name, fig in (deal.get("figures") or {}).items():
        if fig.get("parts"):
            for i, part in enumerate(fig["parts"]):
                yield f"{name}.parts.{i}", part
        else:
            yield name, fig


def is_filing_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in FILING_HOSTS and not parsed.username


def _pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator * 100, 2)


def derived(deal: dict) -> dict:
    """What the figures say together: the entry multiple, leverage, the fees
    as Settings measure them (transaction fees per cent of the value paid,
    financing fees per cent of the debt) and the senior amortisation."""
    tv, ebitda, debt = value(deal, "transaction_value"), value(deal, "ebitda"), value(deal, "debt")
    return {
        "entry_multiple": round(tv / ebitda, 2) if tv and ebitda else None,
        "leverage": round(debt / ebitda, 2) if debt is not None and ebitda else None,
        "debt_share_pct": _pct(debt, tv),
        "tx_fee_pct": _pct(value(deal, "transaction_fees"), tv),
        "fin_fee_pct": _pct(value(deal, "financing_fees"), debt),
        "def_senior_amort": value(deal, "senior_amort_pct"),
    }


def _in_range(x: Optional[float], bounds: tuple[float, float]) -> bool:
    return x is None or bounds[0] <= x <= bounds[1]


def problems(deal: dict, *, today: Optional[date] = None) -> list[dict]:
    """The inclusion rules: why this proposal can't join the library, as
    codes (the screen words them). Empty when it may be approved."""
    out: list[dict] = []

    def add(code: str, at: Optional[str] = None) -> None:
        assert code in RULE_CODES, code
        out.append({"code": code, **({"at": at} if at else {})})

    figures = deal.get("figures") or {}
    sources = deal.get("sources") or {}
    if not deal.get("sponsors"):
        add("no_sponsor")
    for name in REQUIRED:
        if name not in figures:
            add("missing_figure", name)
    if deal.get("currency") != "USD" and "transaction_value_usd" not in figures:
        add("missing_figure", "transaction_value_usd")
    for name in figures:
        if name not in FIGURES:
            add("unknown_figure", name)
    for at, cited in _evidence(deal):
        source = sources.get(cited.get("source"))
        if source is None:
            add("missing_source", at)
        elif not is_filing_url(source.get("url", "")):
            add("not_a_filing", at)
    used = {cited.get("source") for _, cited in _evidence(deal)}
    for sid in sources:
        if sid not in used:
            add("unused_source", sid)
    for name, fig in figures.items():
        parts = fig.get("parts")
        if parts and not math.isclose(sum(float(p["value"]) for p in parts), float(fig["value"]),
                                      abs_tol=PARTS_TOLERANCE):
            add("parts_dont_add_up", name)
        if name not in PERCENT_FIGURES and name not in ("transaction_fees", "financing_fees") \
                and float(fig["value"]) <= 0:
            add("not_positive", name)
    d = derived(deal)
    if not _in_range(d["entry_multiple"], MULTIPLE_RANGE):
        add("multiple_out_of_range", "ebitda")
    tv, debt = value(deal, "transaction_value"), value(deal, "debt")
    if tv is not None and debt is not None and debt > tv:
        add("debt_exceeds_value", "debt")
    for name, key in (("transaction_fees", "tx_fee_pct"), ("financing_fees", "fin_fee_pct")):
        if not _in_range(d[key], FEE_RANGE_PCT):
            add("fee_out_of_range", name)
    if not _in_range(d["def_senior_amort"], AMORT_RANGE_PCT):
        add("amortisation_out_of_range", "senior_amort_pct")
    closed = date.fromisoformat(deal["closed"]["date"])
    if closed > (today or date.today()):
        add("closed_in_future", "closed")
    outcome = deal["outcome"]
    if outcome["kind"] != "held" and outcome["year"] < closed.year:
        add("outcome_before_close", "outcome")
    if outcome["event"] not in EVENTS.get(outcome["kind"], ()):
        add("event_doesnt_match_outcome", "outcome")
    return out


def usd_value(deal: dict) -> Optional[float]:
    if deal.get("currency") == "USD":
        return value(deal, "transaction_value")
    return value(deal, "transaction_value_usd")


def tags(deal: dict) -> dict:
    """The deal's bucket in each dimension the coverage page counts."""
    return {
        "region": base_rates.sp_region(deal["country"]),
        "size": coverage.size(usd_value(deal)),
        "sector": deal["sector"],
        "era": coverage.era(closed_year(deal)),
        "outcome": deal["outcome"]["kind"],
    }


def balance(library: list[dict], deal: dict) -> dict:
    """The balance rules for adding ``deal`` to the approved ``library``:
    the buckets it would push over ``BALANCE_SHARE`` (once the library holds
    ``BALANCE_MIN`` deals) and the empty buckets it fills."""
    held = [tags(d) for d in library]
    mine = tags(deal)
    after = len(held) + 1
    over, fills = [], []
    for dim in BALANCED:
        same = sum(1 for t in held if t[dim] == mine[dim]) + 1
        if same == 1:
            fills.append({"dimension": dim, "bucket": mine[dim]})
        if after >= BALANCE_MIN and same / after > BALANCE_SHARE:
            over.append({"dimension": dim, "bucket": mine[dim], "share_pct": round(same / after * 100, 1)})
    return {"over": over, "fills": fills}


def summary(deal: dict) -> dict:
    """A transaction as the screens show it: its content, what its figures
    say together and its buckets."""
    return {**deal, "derived": derived(deal), "tags": tags(deal)}
