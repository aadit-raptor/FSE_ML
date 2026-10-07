"""Risk ranges, correlations and scenarios by region (PLAN.md 4.4).

The Monte Carlo simulation draws a growth rate, an exit multiple, an
interest rate, a gross margin and a one-year EBITDA shock for every path,
around means with spreads, correlated, and shifts the means for its
scenario presets. Every one of those Settings is worked out here from
published history (benchmarks/history.py), each with its source, the group
or country it was read for, its sample and its years:

- **Means**: the deal's own sourced starting figures (benchmarks/starting.py):
  the country's nominal growth, the industry's EV/EBITDA, the currency's
  benchmark plus spread, the industry's gross margin.
- **Spreads** (standard deviations), from real year-to-year variation:
  the **exit multiple** and **gross margin** are the spread of the
  industry's own EV/EBITDA and gross margin across the years of
  Damodaran's archive, in the closest group with ``MIN_YEARS`` usable years
  (country file, region, global, as for the starting figures); **growth**
  is the spread of the country's nominal GDP growth year to year (IMF; no
  free source gives an industry's revenue growth by year); the **rate** is
  the spread of the year-to-year change in the country's policy rate (BIS).
- **Correlations**, per group: rank correlations of the yearly changes,
  pooled over the group's industries and years (Spearman, turned into the
  normal correlation the simulation draws with), the economy's figures being
  the group's economies' median. A matrix that is not a valid correlation
  matrix is shrunk toward no correlation until it is (``valid_matrix``).
- **Scenarios**, from the country's own past (IMF, ``WINDOW_START`` to the
  last full year): a **recession** is the weakest fifth of its years by real
  GDP growth, **stagflation** its fifth with the highest inflation, **bull**
  its strongest fifth. Each preset moves growth by the gap between nominal
  growth in those years and its average, the interest rate by the policy
  rate's average change in them, and the exit multiple and gross margin by
  the industry's figure in those years against its average. The years are
  listed as periods.

Nothing here is split by company size (no free source does), and nothing
changes the engine: these are Settings values, applied when the user asks,
stored with the deal like typed ones.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from datetime import date
from typing import Callable, Mapping, Optional

import numpy as np
from scipy.stats import spearmanr

from benchmarks import history, starting
from benchmarks.catalogue import ALL_INDUSTRIES_ID, MIN_FIRMS, REGIONS, SOURCE, chain
from benchmarks.damodaran import Table
from benchmarks.starting import Figure, Skipped
from economy.catalogue import COUNTRIES
from economy.model import Series

# Years of usable figures a spread is measured on, at least
MIN_YEARS = 6
# A scenario takes this share of a country's years
SHARE = 0.2
# Pooled industry-years a correlation is measured on, at least; fewer leaves it at 0
MIN_PAIR_OBSERVATIONS = 30
# The smallest eigenvalue a correlation matrix keeps, so rounding its
# entries to CORR_DECIMALS can't make it invalid (ten entries off by at most
# 0.005 move an eigenvalue by at most 0.02)
EIGEN_FLOOR = 0.05
SHRINK_STEP = 0.01
CORR_DECIMALS = 2
MAX_MULTIPLE = starting.MAX_MULTIPLE

VARIABLES = ("growth", "exit_multiple", "interest", "gross_margin", "ebitda_shock")
CORR_KEYS: Mapping[tuple[int, int], str] = {
    (0, 1): "corr_g_em", (0, 2): "corr_g_ir", (0, 3): "corr_g_gm", (0, 4): "corr_g_sh",
    (1, 2): "corr_em_ir", (1, 3): "corr_em_gm", (1, 4): "corr_em_sh",
    (2, 3): "corr_ir_gm", (2, 4): "corr_ir_sh", (3, 4): "corr_gm_sh",
}
MEANS = {"growth": "mc_growth_mean", "exit_mult": "mc_exit_mean", "base_rate": "mc_rate_mean",
         "gross_margin": "mc_gm_mean"}
SPREADS = ("mc_growth_std", "mc_exit_std", "mc_rate_std", "mc_gm_std")
# Scenario presets: the rule choosing their years, and their Settings keys
SCENARIOS = {
    "recession": {"rule": "weakest_growth", "prefix": "rec"},
    "stagflation": {"rule": "highest_inflation", "prefix": "stag"},
    "bull": {"rule": "strongest_growth", "prefix": "bull"},
}
SCENARIO_SETTINGS = tuple(
    f"{p}_{k}" for p, keys in (("bull", ("growth_mult", "exit_mult", "rate_mult", "margin_mult")),
                               ("rec", ("growth_adj", "growth_floor", "exit_mult", "rate_mult", "margin_mult")),
                               ("stag", ("growth_adj", "growth_floor", "exit_mult", "rate_mult", "margin_mult")))
    for k in keys)
# Every Setting this module produces (the defaults registry's "sourced" ones)
SETTINGS = (*MEANS.values(), *SPREADS, *CORR_KEYS.values(), *SCENARIO_SETTINGS)
NOTES = ("size_not_split", "growth_economy_wide", "rate_policy_only", "industry_aggregates")


def _multiple_usable(v: float) -> bool:
    return 0 < v <= MAX_MULTIPLE


def _margin_usable(v: float) -> bool:
    return 0 < v < 1


def _window(series: Mapping[int, float], today: date) -> dict[int, float]:
    return {y: v for y, v in series.items() if history.WINDOW_START <= y < today.year}


def _period(years) -> Optional[str]:
    years = sorted(years)
    return f"{years[0]}-{years[-1]}" if years else None


def periods(years) -> list[dict]:
    """Consecutive years as periods: [2008, 2009, 2020] -> 2008-2009 and 2020."""
    out: list[dict] = []
    for y in sorted(years):
        if out and out[-1]["end"] == y - 1:
            out[-1]["end"] = y
        else:
            out.append({"start": y, "end": y})
    return out


# ---------------------------------------------------------------------------
# An industry's history, in the closest group with enough of it
# ---------------------------------------------------------------------------
def industry_series(tables: Mapping[str, Table], group: str, industry: str, metric: str,
                    usable: Callable[[float], bool]) -> dict[int, float]:
    """The industry's yearly figure in ``group``, for years with at least
    ``MIN_FIRMS`` companies and a usable value."""
    return {y: f[metric] for y, f in history.industry_years(tables, group, industry).items()
            if f.get("firms", 0) >= MIN_FIRMS and isinstance(f.get(metric), (int, float)) and usable(f[metric])}


def _pick(tables, country, industry, metric, usable) -> tuple[Optional[tuple[str, dict]], list]:
    skipped = []
    for area in chain(country):
        if industry not in history.industries_in(tables, area):
            skipped.append(Skipped(area, "missing"))
            continue
        series = industry_series(tables, area, industry, metric, usable)
        if len(series) < MIN_YEARS:
            skipped.append(Skipped(area, "short", len(series)))
            continue
        return (area, series), skipped
    return None, skipped


def _latest_firms(tables, area, industry) -> Optional[int]:
    years = history.industry_years(tables, area, industry)
    return years[max(years)].get("firms") if years else None


def _industry_figure(name: str, value: float, tables, area: str, industry: str, series: dict, skipped: list,
                     **detail) -> Figure:
    current = tables.get(f"margins.{area}")
    return Figure(name, value, SOURCE["id"], history.table_name(area), area, REGIONS[area].level,
                  _latest_firms(tables, area, industry), "companies",
                  current.published.isoformat() if current and current.published else None,
                  history.ARCHIVE_PAGE, tuple(skipped),
                  {"years": len(series), "period": _period(series), **detail})


# ---------------------------------------------------------------------------
# Economy-wide history: a country's own, else its group's median
# ---------------------------------------------------------------------------
def nominal_growth(tables: Mapping[str, Table], area: str) -> dict[int, float]:
    """(1 + real growth)(1 + inflation) - 1, in %, by year."""
    real = history.macro_series(tables, area, "real_growth")
    inflation = history.macro_series(tables, area, "inflation")
    return {y: ((1 + real[y] / 100) * (1 + inflation[y] / 100) - 1) * 100 for y in real if y in inflation}


def rate_changes(tables: Mapping[str, Table], country: str) -> dict[int, float]:
    """The year-to-year change in the yearly average policy rate, in points."""
    rates = history.macro_series(tables, history.policy_area(country), "policy_rate")
    return {y: rates[y] - rates[y - 1] for y in rates if y - 1 in rates}


def members(group: str) -> list[str]:
    """The catalogue's economies whose figures stand for a group."""
    return list(COUNTRIES) if group == "global" else [c for c in COUNTRIES if group in chain(c)]


def _median_by_year(per_country: list[dict[int, float]]) -> dict[int, float]:
    years = sorted({y for s in per_country for y in s})
    return {y: statistics.median(vals) for y in years if (vals := [s[y] for s in per_country if y in s])}


def _economy_spread(name: str, tables, country: str, series_of: Callable, today: date, source: str,
                    dataset: str, url: str, **detail) -> tuple[Optional[Figure], list]:
    """The spread of a yearly economic figure: the country's own, else the
    median of its region's (then the world's) economies' spreads."""
    own = _window(series_of(tables, country), today)
    if len(own) >= MIN_YEARS:
        return Figure(name, round(statistics.stdev(own.values()), 2), source, dataset, country, "country", 1,
                      "economies", None, url, (), {"years": len(own), "period": _period(own), **detail}), []
    skipped = [Skipped(country, "short" if own else "missing", len(own) or None)]
    region = next(g for g in chain(country) if REGIONS[g].level != "country")
    for area in dict.fromkeys((region, "global")):
        spreads, spans = [], []
        for c in members(area):
            s = _window(series_of(tables, c), today)
            if len(s) >= MIN_YEARS:
                spreads.append(statistics.stdev(s.values()))
                spans += list(s)
        if spreads:
            return Figure(name, round(statistics.median(spreads), 2), source, dataset, area, REGIONS[area].level,
                          len(spreads), "economies", None, url, tuple(skipped),
                          {"statistic": "median", "period": _period(spans), **detail}), skipped
        skipped.append(Skipped(area, "missing"))
    return None, skipped


def _economy_years(tables, country: str, today: date) -> tuple[Optional[dict], list]:
    """The economy a country's scenario years are chosen in: its own
    history, else its region's median by year."""
    real = _window(history.macro_series(tables, country, "real_growth"), today)
    inflation = _window(history.macro_series(tables, country, "inflation"), today)
    if len(real) >= MIN_YEARS and len(inflation) >= MIN_YEARS:
        return {"area": country, "level": "country", "economies": 1, "real": real, "inflation": inflation,
                "nominal": _window(nominal_growth(tables, country), today),
                "rate_change": _window(rate_changes(tables, country), today)}, []
    skipped = [Skipped(country, "short" if real else "missing", len(real) or None)]
    region = next(g for g in chain(country) if REGIONS[g].level != "country")
    for area in dict.fromkeys((region, "global")):
        found = [c for c in members(area) if len(_window(history.macro_series(tables, c, "real_growth"), today)) >= MIN_YEARS]
        if found:
            pick = lambda f: _window(_median_by_year([f(c) for c in found]), today)  # noqa: E731
            return {"area": area, "level": REGIONS[area].level, "economies": len(found),
                    "real": pick(lambda c: history.macro_series(tables, c, "real_growth")),
                    "inflation": pick(lambda c: history.macro_series(tables, c, "inflation")),
                    "nominal": pick(lambda c: nominal_growth(tables, c)),
                    "rate_change": pick(lambda c: rate_changes(tables, c))}, skipped
        skipped.append(Skipped(area, "missing"))
    return None, skipped


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def fifth(values: Mapping[int, float], top: bool) -> list[int]:
    """The years in the top (or bottom) fifth of ``values``; ties go to the
    later year."""
    n = max(1, round(len(values) * SHARE))
    ordered = sorted(values, key=lambda y: (values[y], y if top else -y), reverse=top)
    return sorted(ordered[:n])


def _mean_in(series: Mapping[int, float], years) -> Optional[float]:
    found = [series[y] for y in years if y in series]
    return statistics.fmean(found) if found else None


def _scenario(sid: str, economy: dict, means: Mapping[str, float], industry_history: Mapping[str, Optional[tuple]],
              skipped: list) -> tuple[dict, list[Figure], dict]:
    rule = SCENARIOS[sid]["rule"]
    prefix = SCENARIOS[sid]["prefix"]
    years = (fifth(economy["inflation"], True) if rule == "highest_inflation"
             else fifth(economy["real"], rule == "strongest_growth"))
    meta = {"id": sid, "rule": rule, "area": economy["area"], "level": economy["level"],
            "economies": economy["economies"], "years": years, "periods": periods(years),
            "of_years": len(economy["real"]), "window": _period(economy["real"])}
    figures, missing = [], {}
    base = dict(source="imf", dataset="NGDP_RPCH+PCPIPCH", area=economy["area"], level=economy["level"],
                sample=economy["economies"], sample_kind="economies", as_of=None, url=history.IMF_PAGE,
                skipped=tuple(skipped))

    def add(key: str, value: Optional[float], why: str, **detail):
        if value is None or not math.isfinite(value):
            missing[key] = why
        else:
            figures.append(Figure(field=key, value=value, detail={"years": years, **detail}, **base))

    nominal = economy["nominal"]
    in_years = _mean_in(nominal, years)
    gap = None if in_years is None else in_years - statistics.fmean(nominal.values())
    if prefix == "bull":
        g = means.get("mc_growth_mean")
        add(f"{prefix}_growth_mult", round((g + gap) / g, 3) if gap is not None and g and g > 0 else None,
            "missing" if gap is None else "base_not_positive", change=None if gap is None else round(gap, 2), base=g)
    else:
        add(f"{prefix}_growth_adj", None if gap is None else round(gap, 2), "missing")
        worst = min((nominal[y] for y in years if y in nominal), default=None)
        add(f"{prefix}_growth_floor", None if worst is None else round(worst, 2), "missing")
    r = means.get("mc_rate_mean")
    change = _mean_in(economy["rate_change"], years)
    base.update(source="bis", dataset="WS_CBPOL", url=history.BIS_PAGE)
    add(f"{prefix}_rate_mult",
        round(max(r + change, 0.0) / r, 3) if change is not None and r and r > 0 else None,
        "missing" if change is None else "base_not_positive",
        change=None if change is None else round(change, 2), base=r)
    for key, (name, scale) in {"exit_mult": ("ev_ebitda", 1), "margin_mult": ("gross_margin", 100)}.items():
        picked = industry_history.get(name)
        if not picked:
            missing[f"{prefix}_{key}"] = "missing"
            continue
        area, series = picked
        inside = _mean_in(series, years)
        if inside is None:
            missing[f"{prefix}_{key}"] = "no_history_in_years"
            continue
        average = statistics.fmean(series.values())
        figures.append(Figure(f"{prefix}_{key}", round(inside / average, 3), SOURCE["id"], history.table_name(area),
                              area, REGIONS[area].level, None, "companies", None, history.ARCHIVE_PAGE, (),
                              {"years": [y for y in years if y in series],
                               "in_years": round(inside * scale, 2), "average": round(average * scale, 2),
                               "period": _period(series)}))
    return meta, figures, missing


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def history_group(tables: Mapping[str, Table], country: str) -> str:
    """The closest group with industry history: correlations are per group."""
    for area in chain(country):
        if history.table_name(area) in tables:
            return area
    return "global"


def _change(prev: Mapping, now: Mapping, metric: str, usable, relative: bool = False) -> Optional[float]:
    a, b = prev.get(metric), now.get(metric)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not (usable(a) and usable(b)):
        return None
    return b / a - 1 if relative else b - a


def panel(tables: Mapping[str, Table], group: str, today: date) -> tuple[list[list], dict[int, float], dict[int, float], set]:
    """One row per industry and year: [growth, change in EV/EBITDA, change in
    the policy rate, change in gross margin, relative change in EBITDA
    margin]; growth and the rate are the group's economies' median that
    year. Also the two economy-wide series by year, and the years the rows
    cover."""
    economies = members(group)
    growth = _window(_median_by_year([nominal_growth(tables, c) for c in economies]), today)
    rates = _window(_median_by_year([rate_changes(tables, a)
                                     for a in dict.fromkeys(history.policy_area(c) for c in economies)]), today)
    rows, covered = [], set()
    for industry in sorted(history.industries_in(tables, group) - {ALL_INDUSTRIES_ID}):
        years = history.industry_years(tables, group, industry)
        for y, now in years.items():
            prev = years.get(y - 1)
            if not prev or now.get("firms", 0) < MIN_FIRMS or prev.get("firms", 0) < MIN_FIRMS:
                continue
            covered.add(y)
            rows.append([growth.get(y), _change(prev, now, "ev_ebitda", _multiple_usable), rates.get(y),
                         _change(prev, now, "gross_margin", _margin_usable),
                         _change(prev, now, "ebitda_margin", _margin_usable, relative=True)])
    return rows, growth, rates, covered


def _rank_correlation(xs: list, ys: list) -> float:
    """Spearman's rho turned into the correlation of the normal draws that
    has it: 2 sin(pi rho / 6)."""
    rho = spearmanr(xs, ys).statistic
    return 0.0 if not math.isfinite(rho) else 2 * math.sin(math.pi * rho / 6)


def raw_correlations(rows: list[list], growth: Mapping[int, float], rates: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """The 5x5 correlations and how many observations each rests on. Growth
    and the rate are economy-wide, so their pair is measured on years, not
    pooled rows; a pair with too little evidence stays at 0."""
    m, counts = np.eye(len(VARIABLES)), np.zeros((len(VARIABLES), len(VARIABLES)), dtype=int)
    for (i, j) in CORR_KEYS:
        if (i, j) == (0, 2):
            years = [y for y in growth if y in rates]
            xs, ys, enough = [growth[y] for y in years], [rates[y] for y in years], len(years) >= MIN_YEARS
        else:
            pairs = [(r[i], r[j]) for r in rows if r[i] is not None and r[j] is not None]
            xs, ys, enough = [p[0] for p in pairs], [p[1] for p in pairs], len(pairs) >= MIN_PAIR_OBSERVATIONS
        counts[i, j] = counts[j, i] = len(xs)
        if enough and len(set(xs)) > 1 and len(set(ys)) > 1:
            m[i, j] = m[j, i] = _rank_correlation(xs, ys)
    return m, counts


def valid_matrix(m: np.ndarray) -> tuple[np.ndarray, float]:
    """``m`` rounded to ``CORR_DECIMALS``, shrunk toward the identity
    ((1 - s) m + s I, the smallest s in steps of ``SHRINK_STEP``) until its
    smallest eigenvalue is at least ``EIGEN_FLOOR`` and it passes the
    Cholesky check the simulation makes. Returns the matrix and s."""
    from core.config import is_valid_corr
    eye = np.eye(len(m))
    for k in range(int(round(1 / SHRINK_STEP)) + 1):
        s = k * SHRINK_STEP
        shrunk = (1 - s) * m + s * eye
        if np.linalg.eigvalsh(shrunk).min() >= EIGEN_FLOOR:
            rounded = np.round(shrunk, CORR_DECIMALS)
            np.fill_diagonal(rounded, 1.0)
            if is_valid_corr(rounded):
                return rounded, round(s, 2)
    return eye, 1.0


def correlations(tables: Mapping[str, Table], country: str, today: date) -> dict:
    group = history_group(tables, country)
    rows, growth, rates, years = panel(tables, group, today)
    raw, counts = raw_correlations(rows, growth, rates)
    matrix, shrink = valid_matrix(raw)
    return {"group": group, "level": REGIONS[group].level, "labels": list(VARIABLES),
            "matrix": matrix.tolist(), "raw": np.round(raw, 4).tolist(), "observations": counts.tolist(),
            "rows": len(rows), "economies": len(members(group)), "period": _period(years), "shrink": shrink}


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def risk_assumptions(country: str, industry: str, currency: str, tables: Mapping[str, Table],
                     economy: Mapping[str, Series], today: date) -> dict:
    """The sourced Monte Carlo Settings for a deal in ``country`` and
    ``industry`` priced in ``currency``."""
    figures: list[Figure] = []
    missing: dict[str, list] = {}

    def add(fig: Optional[Figure], skipped: list, *names: str) -> None:
        if fig is not None:
            figures.append(fig)
        else:
            for n in names:
                missing.setdefault(n, skipped)

    # Means: the deal's own sourced starting figures
    start = starting.starting_assumptions(country, industry, currency, tables, economy, today)
    by_field = {f["field"]: f for f in start["figures"]}
    starting_missing = {m["field"]: [Skipped(**s) for s in m["skipped"]] for m in start["missing"]}
    for field, key in MEANS.items():
        if field in by_field:
            f = dict(by_field[field])
            f["skipped"] = tuple(Skipped(**s) for s in f["skipped"])
            add(Figure(**{**f, "field": key}), [])
        else:
            add(None, starting_missing.get(field, []), key)
    means = {f.field: f.value for f in figures}

    # Spreads
    picked = {}
    for name, key, usable, scale, decimals in (("ev_ebitda", "mc_exit_std", _multiple_usable, 1, 2),
                                               ("gross_margin", "mc_gm_std", _margin_usable, 100, 2)):
        found, skipped = _pick(tables, country, industry, name, usable)
        picked[name] = found
        if found:
            area, series = found
            add(_industry_figure(key, round(statistics.stdev(series.values()) * scale, decimals), tables, area,
                                 industry, series, skipped, mean=round(statistics.fmean(series.values()) * scale, 2)),
                [])
        else:
            add(None, skipped, key)
    add(*_economy_spread("mc_growth_std", tables, country, nominal_growth, today, "imf", "NGDP_RPCH+PCPIPCH",
                         history.IMF_PAGE, measure="nominal_gdp_growth"), "mc_growth_std")
    add(*_economy_spread("mc_rate_std", tables, country, rate_changes, today, "bis", "WS_CBPOL",
                         history.BIS_PAGE, measure="policy_rate_change",
                         policy_area=history.policy_area(country)), "mc_rate_std")

    # Correlations
    corr = correlations(tables, country, today)
    if corr["rows"]:
        for (i, j), key in CORR_KEYS.items():
            figures.append(Figure(key, corr["matrix"][i][j], "damodaran+imf+bis", history.table_name(corr["group"]),
                                  corr["group"], corr["level"], int(corr["observations"][i][j]),
                                  "years" if (i, j) == (0, 2) else "industry_years", None, history.ARCHIVE_PAGE, (),
                                  {"raw": corr["raw"][i][j], "shrink": corr["shrink"], "period": corr["period"]}))
    else:
        for key in CORR_KEYS.values():
            missing[key] = [Skipped(corr["group"], "missing")]

    # Scenarios
    scenarios = {}
    economy_years, skipped = _economy_years(tables, country, today)
    if economy_years:
        for sid in SCENARIOS:
            meta, found, gaps = _scenario(sid, economy_years, means, picked, skipped)
            scenarios[sid] = meta
            figures += found
            for key, why in gaps.items():
                missing[key] = [Skipped(economy_years["area"], why)]
    else:
        for key in SCENARIO_SETTINGS:
            missing[key] = skipped

    settings = {f.field: f.value for f in figures}
    return {
        "country": country, "industry": industry, "currency": currency,
        "industry_name": start["industry_name"],
        "region": next(g for g in chain(country) if REGIONS[g].level != "country"),
        "settings": settings,
        "figures": [starting._plain(f) for f in figures],
        "missing": [{"field": k, "skipped": [asdict(s) for s in v]} for k, v in missing.items() if k not in settings],
        "scenarios": scenarios,
        "correlation": {k: v for k, v in corr.items() if k != "raw"},
        "window_start": history.WINDOW_START, "min_years": MIN_YEARS,
        "notes": list(NOTES),
        "source": dict(SOURCE),
    }
