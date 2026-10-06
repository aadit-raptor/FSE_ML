"""The six economic data sources, each read into ``Series`` (PLAN.md 4.2).

All of them go through companies/http.py: paced per host, a User-Agent
naming the product, no URL or key ever in a message. Each function makes
one or a few calls covering every economy at once, so a whole refresh is
about a dozen requests:

- ``imf``: the World Economic Outlook through the DataMapper API (growth,
  inflation; history and projections), one call per indicator.
- ``worldbank``: World Development Indicators (the latest actual years of
  growth and inflation).
- ``bis``: central bank policy rates (daily).
- ``oecd``: long-term government bond yields and 3-month interbank rates
  (monthly averages).
- ``ecb``: €STR and 3-month EURIBOR; ``ecb_fx`` the euro reference rates.
- ``fred``: SOFR, SONIA and the US 10-year Treasury yield (needs
  ``FRED_API_KEY``).

Sources answer CSV or JSON; a row that can't be read (a missing value, a
non-finite number) is skipped rather than guessed.
"""
from __future__ import annotations

import csv
import io
import math
import os
from datetime import date
from typing import Optional

from companies import http
from economy.catalogue import AREAS, COUNTRIES, EURO_MEMBERS
from economy.model import FxDay, Series

IMF_API = "https://www.imf.org/external/datamapper/api/v1"
WORLDBANK_API = "https://api.worldbank.org/v2"
BIS_API = "https://stats.bis.org/api/v1/data"
OECD_API = "https://sdmx.oecd.org/public/rest/data"
OECD_FLOW = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0"
ECB_API = "https://data-api.ecb.europa.eu/service/data"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY_ENV = "FRED_API_KEY"

IMF_INDICATORS = {"gdp_growth": "NGDP_RPCH", "inflation": "PCPIPCH"}
WORLDBANK_INDICATORS = {"gdp_growth": "NY.GDP.MKTP.KD.ZG", "inflation": "FP.CPI.TOTL.ZG"}
OECD_MEASURES = {"IRLT": "bond_yield_10y", "IR3TIB": "short_rate_3m"}
# (indicator, area, flow, key) at the ECB
ECB_RATES = (
    ("reference_rate", "ESTR", "EST", "B.EU000A2X2A25.WT", "D"),
    ("reference_rate", "EURIBOR", "FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "M"),
)
# (indicator, area, FRED series id, frequency)
FRED_SERIES = (
    ("reference_rate", "SOFR", "SOFR", "D"),
    ("reference_rate", "SONIA", "IUDSOIA", "D"),
    ("bond_yield_10y", "US", "DGS10", "D"),
)
# Years of the WEO kept: recent history and the projections ahead
IMF_YEARS_BACK, IMF_YEARS_AHEAD = 6, 5


def _number(text) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _csv(source: str, resp) -> list[dict]:
    try:
        return list(csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error):
        raise http.SourceError(source, "unreadable") from None


# ---------------------------------------------------------------------------
# IMF
# ---------------------------------------------------------------------------
def imf(today: date) -> list[Series]:
    by_iso3 = {AREAS[c].iso3: c for c in COUNTRIES}
    years = range(today.year - IMF_YEARS_BACK, today.year + IMF_YEARS_AHEAD + 1)
    out = []
    for indicator, code in IMF_INDICATORS.items():
        data = http.get_json("imf", f"{IMF_API}/{code}") or {}
        values = (data.get("values") or {}).get(code) or {}
        for iso3, country in by_iso3.items():
            row = values.get(iso3) or {}
            obs = [(str(y), v) for y in years if (v := _number(row.get(str(y)))) is not None]
            if obs:
                out.append(Series(indicator, country, "imf", code, "A",
                                  f"https://www.imf.org/external/datamapper/{code}@WEO/{iso3}", tuple(obs)))
    return out


# ---------------------------------------------------------------------------
# World Bank
# ---------------------------------------------------------------------------
def worldbank() -> list[Series]:
    countries = ";".join(AREAS[c].iso3 for c in COUNTRIES)
    out = []
    for indicator, code in WORLDBANK_INDICATORS.items():
        data = http.get_json("worldbank", f"{WORLDBANK_API}/country/{countries}/indicator/{code}",
                             params={"format": "json", "mrv": "6", "per_page": "1000"})
        rows = data[1] if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list) else []
        found: dict[str, list] = {}
        for r in rows:
            country = (r.get("country") or {}).get("id", "")
            value = _number(r.get("value"))
            if country in AREAS and value is not None and str(r.get("date", "")).isdigit():
                found.setdefault(country, []).append((str(r["date"]), value))
        out += [Series(indicator, c, "worldbank", code, "A",
                       f"https://data.worldbank.org/indicator/{code}?locations={c}", tuple(obs))
                for c, obs in found.items()]
    return out


# ---------------------------------------------------------------------------
# BIS
# ---------------------------------------------------------------------------
def bis() -> list[Series]:
    areas = [c for c in AREAS if c not in EURO_MEMBERS]          # members share XM's
    resp = http.get("bis", f"{BIS_API}/WS_CBPOL/D.{'+'.join(areas)}/all",
                    params={"lastNObservations": "10", "format": "csv", "detail": "dataonly"})
    found: dict[str, list] = {}
    for r in _csv("bis", resp):
        value = _number(r.get("OBS_VALUE"))
        if r.get("REF_AREA") in AREAS and value is not None:
            found.setdefault(r["REF_AREA"], []).append((r.get("TIME_PERIOD", ""), value))
    return [Series("policy_rate", area, "bis", f"WS_CBPOL/D.{area}", "D",
                   f"https://data.bis.org/topics/CBPOL/BIS%2CWS_CBPOL%2C1.0/D.{area}", tuple(obs))
            for area, obs in found.items()]


# ---------------------------------------------------------------------------
# OECD
# ---------------------------------------------------------------------------
def oecd() -> list[Series]:
    by_iso3 = {a.iso3: code for code, a in AREAS.items()}
    resp = http.get("oecd", f"{OECD_API}/{OECD_FLOW}/{'+'.join(by_iso3)}.M.{'+'.join(OECD_MEASURES)}.PA.....",
                    params={"lastNObservations": "24", "format": "csvfile"})
    found: dict[tuple, list] = {}
    for r in _csv("oecd", resp):
        area, indicator = by_iso3.get(r.get("REF_AREA", "")), OECD_MEASURES.get(r.get("MEASURE", ""))
        value = _number(r.get("OBS_VALUE"))
        if area and indicator and value is not None:
            found.setdefault((indicator, area, r["MEASURE"]), []).append((r.get("TIME_PERIOD", ""), value))
    link = "https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_FINMARK&df[ag]=OECD.SDD.STES"
    return [Series(indicator, area, "oecd", f"DF_FINMARK/{AREAS[area].iso3}.M.{measure}", "M", link, tuple(obs))
            for (indicator, area, measure), obs in found.items()]


# ---------------------------------------------------------------------------
# ECB
# ---------------------------------------------------------------------------
def ecb() -> list[Series]:
    out = []
    for indicator, area, flow, key, freq in ECB_RATES:
        resp = http.get("ecb", f"{ECB_API}/{flow}/{key}",
                        params={"lastNObservations": "24", "format": "csvdata", "detail": "dataonly"})
        obs = [(r.get("TIME_PERIOD", ""), v) for r in _csv("ecb", resp)
               if (v := _number(r.get("OBS_VALUE"))) is not None]
        if obs:
            out.append(Series(indicator, area, "ecb", f"{flow}.{key}", freq,
                              f"https://data.ecb.europa.eu/data/datasets/{flow}/{flow}.{key}", tuple(obs)))
    return out


def ecb_fx(since: date) -> list[FxDay]:
    """The euro reference rates for every day from ``since``; a currency the
    ECB has stopped quoting has no rows in that window, so it drops out."""
    resp = http.get("ecb", f"{ECB_API}/EXR/D..EUR.SP00.A",
                    params={"startPeriod": since.isoformat(), "format": "csvdata", "detail": "dataonly"})
    days: dict[str, dict] = {}
    for r in _csv("ecb", resp):
        currency, value = r.get("CURRENCY", ""), _number(r.get("OBS_VALUE"))
        if len(currency) == 3 and currency.isalpha() and value and value > 0:
            days.setdefault(r.get("TIME_PERIOD", ""), {})[currency.upper()] = value
    out = []
    for day, rates in sorted(days.items()):
        try:
            out.append(FxDay(date.fromisoformat(day), {"EUR": 1.0, **rates}))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def fred_configured() -> bool:
    return bool(os.environ.get(FRED_KEY_ENV))


def fred() -> list[Series]:
    key = os.environ.get(FRED_KEY_ENV)
    if not key:
        raise http.NotConfigured("fred")
    out = []
    for indicator, area, series_id, freq in FRED_SERIES:
        data = http.get_json("fred", FRED_API, params={
            "series_id": series_id, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": "24"},
            secret_params=("api_key",)) or {}
        obs = [(o.get("date", ""), v) for o in data.get("observations") or []
               if (v := _number(o.get("value"))) is not None]
        if obs:
            out.append(Series(indicator, area, "fred", series_id, freq,
                              f"https://fred.stlouisfed.org/series/{series_id}", tuple(obs)))
    return out

