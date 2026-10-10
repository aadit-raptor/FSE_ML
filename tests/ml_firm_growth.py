"""The companies' revenue the growth calibrator rests on (PLAN.md 5.5).

    python -m tests.ml_firm_growth [--cache DIR]   # rewrites ml/evaluation/data/firm_growth.json and ml/growth_ranges.json

The growth calibrator (ml/growth_calibrator.py) reads how listed companies'
revenue really grew, by region and sector. The source is the SEC's XBRL
data -- the same filings the company search reads (companies/sec.py) --
through its **frames** API, which answers one concept for one calendar
year for every filer at once: each revenue concept the deal model maps
(``core.accounting``: US GAAP and IFRS) in every currency a filer reports
it in, every year from ``FIRST_YEAR``. Each company's business address
(the frame's ``loc``) gives its country, and its filings index
(``SUBMISSIONS_URL``) its SIC industry code. The ECB's yearly average
reference rates turn each company's revenue into US dollars for the size
floor, and the IMF's growth and inflation history (benchmarks/history.py,
the 24 economies the app covers) gives each country's nominal growth.

Only what the calibrator reads is kept: a company's id (its CIK), country,
SIC code and revenue by concept, currency and year, in millions. Companies
with no span of ``MIN_SPAN`` years in one concept are left out. Writing
it calls the SEC a few thousand times (paced at its limit) and takes
about half an hour; ``--cache`` keeps the answers so a rerun reads them
from disk. The file changes once a year, when the new year's 10-Ks are in;
ml/growth_ranges.json is rewritten from it in the same run, and
tests/test_growth_calibrator.py fails when the two disagree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ml" / "evaluation" / "data" / "firm_growth.json"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/CY{year}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ECB_FX_URL = "https://data-api.ecb.europa.eu/service/data/EXR/A..EUR.SP00.A"
FIRST_YEAR = 2008
MIN_SPAN = 3
# Currencies probed: the ECB's reference currencies and the larger others
CURRENCIES = (
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "HKD", "INR", "BRL", "MXN", "ILS", "SEK",
    "NOK", "DKK", "KRW", "TWD", "SGD", "ZAR", "RUB", "CLP", "COP", "ARS", "PHP", "IDR", "TRY", "THB", "MYR",
    "PLN", "CZK", "HUF", "RON", "PEN", "NGN", "EGP", "KZT", "VND", "AED", "SAR", "ISK", "BGN",
)
# Years a concept and currency is probed in before every year is read
PROBE_YEARS = (2012, 2018, 2023)
# Calls in flight at once
WORKERS = 6


def concepts() -> list[tuple[str, str]]:
    from core.accounting import IFRS_ITEMS, US_GAAP_ITEMS
    return ([("us-gaap", c) for c in US_GAAP_ITEMS["revenue"]]
            + [("ifrs-full", c) for c in IFRS_ITEMS["revenue"]])


class Cache:
    """Answers kept on disk by URL, so a rerun calls nobody it already has."""

    def __init__(self, directory: Optional[Path]):
        self.dir = directory
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def json(self, source: str, url: str, **kwargs):
        from companies import http
        path = self.dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json") if self.dir else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        found = http.get_json(source, url, allow_404=True, **kwargs)
        if path:
            path.write_text(json.dumps(found), encoding="utf-8")
        return found


def _country(loc: Optional[str]) -> Optional[str]:
    code = (loc or "").split("-")[0].strip().upper()
    return code if len(code) == 2 and code.isalpha() else None


def fetch_all(cache: Cache, urls: list[str], pick=lambda answer: answer) -> dict[str, object]:
    """``pick`` of every URL's answer, ``WORKERS`` at a time (the SEC
    answers in about half a second; companies/http.py keeps the calls
    within its pace)."""
    with ThreadPoolExecutor(WORKERS) as pool:
        return dict(zip(urls, pool.map(lambda u: pick(cache.json("sec", u)), urls)))


def frames(cache: Cache, last_year: int, log=print) -> dict[int, dict]:
    """``{cik: {"country", "series": {"taxonomy:concept|CUR": {year: millions}}}}``."""
    firms: dict[int, dict] = {}
    years = range(FIRST_YEAR, last_year + 1)
    url = lambda t, c, u, y: FRAMES_URL.format(taxonomy=t, concept=c, unit=u, year=y)  # noqa: E731
    combos = [(t, c, u) for t, c in concepts() for u in CURRENCIES]
    probes = fetch_all(cache, [url(t, c, u, y) for t, c, u in combos for y in PROBE_YEARS])
    found = [(t, c, u) for t, c, u in combos
             if any((p := probes[url(t, c, u, y)]) and p.get("data") for y in PROBE_YEARS)]
    for taxonomy, concept, unit in found:
        # One concept and currency at a time: a year of every filer is a few megabytes parsed
        answers = fetch_all(cache, [url(taxonomy, concept, unit, y) for y in years])
        key = f"{taxonomy}:{concept}|{unit}"
        n = 0
        for year in years:
            data = answers[url(taxonomy, concept, unit, year)]
            for row in (data or {}).get("data") or []:
                value = row.get("val")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                firm = firms.setdefault(int(row["cik"]), {"country": None, "country_year": 0, "series": {}})
                country = _country(row.get("loc"))
                if country and year >= firm["country_year"]:
                    firm["country"], firm["country_year"] = country, year
                firm["series"].setdefault(key, {})[year] = round(value / 1e6, 4)
                n += 1
        log(f"{key}: {n} company-years")
    return firms


def has_span(series: dict[int, float]) -> bool:
    years = sorted(y for y, v in series.items() if v > 0)
    return any(b - a >= MIN_SPAN for a in years for b in years)


def sic_codes(cache: Cache, ciks: list[int], log=print) -> dict[int, Optional[int]]:
    log(f"industry codes for {len(ciks)} companies")
    answers = fetch_all(cache, [SUBMISSIONS_URL.format(cik=cik) for cik in ciks],
                        lambda d: str((d or {}).get("sic") or "").strip())
    return {cik: int(sic) if (sic := answers[SUBMISSIONS_URL.format(cik=cik)]).isdigit() else None for cik in ciks}


def usd_rates(first: int, last: int) -> dict[str, dict[str, float]]:
    """US dollars per unit of each currency, by year: the ECB's yearly
    averages of its euro reference rates (USD per euro over units per euro)."""
    from companies import http
    resp = http.get("ecb", ECB_FX_URL, params={"startPeriod": str(first), "endPeriod": str(last),
                                                 "format": "csvdata", "detail": "dataonly"})
    per_euro: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))):
        try:
            per_euro.setdefault(r["CURRENCY"], {})[r["TIME_PERIOD"]] = float(r["OBS_VALUE"])
        except (KeyError, ValueError):
            continue
    per_euro["EUR"] = {str(y): 1.0 for y in range(first, last + 1)}
    usd = per_euro.get("USD", {})
    return {cur: {y: round(usd[y] / v, 6) for y, v in sorted(years.items()) if y in usd and v}
            for cur, years in sorted(per_euro.items())}


def nominal_growth(today: date) -> dict[str, dict[str, float]]:
    """Each covered economy's nominal growth by year, in %: (1 + real
    growth)(1 + inflation) - 1, from the IMF (benchmarks/history.py)."""
    from benchmarks import history
    out = {}
    for country, series in sorted(history._imf(today).items()):
        real, inflation = series.get("real_growth", {}), series.get("inflation", {})
        out[country] = {y: round(((1 + real[y] / 100) * (1 + inflation[y] / 100) - 1) * 100, 4)
                        for y in sorted(real) if y in inflation}
    return out


def build(today: date, cache: Cache, log=print) -> dict:
    last_year = today.year - 1
    firms = frames(cache, last_year, log)
    kept = {}
    for cik, f in firms.items():
        series = {k: s for k, s in f["series"].items() if has_span(s)}
        if series and f["country"]:
            kept[cik] = {"country": f["country"], "series": series}
    log(f"{len(kept)} of {len(firms)} companies have a span of {MIN_SPAN} years")
    sics = sic_codes(cache, sorted(kept), log)
    companies = [{"id": cik, "country": f["country"], "sic": sics.get(cik),
                  "series": {k: {str(y): v for y, v in sorted(s.items())} for k, s in sorted(f["series"].items())}}
                 for cik, f in sorted(kept.items())]
    return {
        "about": "Written by python -m tests.ml_firm_growth from the SEC's XBRL frames (revenue, every filer), "
                 "its filings index (SIC codes), the ECB's yearly reference rates and the IMF's World Economic "
                 "Outlook; do not edit by hand.",
        "written_on": today.isoformat(),
        "years": [FIRST_YEAR, last_year],
        "sources": {
            "revenue": {"publisher": "U.S. Securities and Exchange Commission", "title": "XBRL frames API",
                        "url": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"},
            "fx": {"publisher": "European Central Bank", "title": "Euro foreign exchange reference rates, "
                   "yearly averages (EXR)", "url": "https://data.ecb.europa.eu/data/datasets/EXR"},
            "growth": {"publisher": "International Monetary Fund", "title": "World Economic Outlook (real GDP "
                       "growth and inflation)", "url": "https://www.imf.org/external/datamapper/datasets/WEO"},
        },
        "usd_per_unit": usd_rates(FIRST_YEAR, last_year),
        "nominal_growth": nominal_growth(today),
        "companies": companies,
    }


def main(argv=None) -> int:
    from companies import http
    parser = argparse.ArgumentParser(prog="python -m tests.ml_firm_growth")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    http.use_transport(None)
    today = datetime.now(timezone.utc).date()
    data = build(today, Cache(args.cache))
    OUT.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['companies'])} companies")
    from ml import growth_calibrator
    growth_calibrator.write_ranges()
    print(f"wrote {growth_calibrator.RANGES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
