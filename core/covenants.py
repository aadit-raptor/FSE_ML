"""Covenants: a lender's yearly tests of a deal (PLAN.md 7.5).

A covenant sits in one facility's agreement and tests the borrower once a
year, at the year end: **leverage** (debt over EBITDA, a maximum), **interest
cover** (EBITDA over interest, a minimum), **fixed-charge cover** (cash flow
over debt service, a minimum) and **minimum liquidity** (cash plus undrawn
revolving commitments, money). The level may step year by year
(``levels``; the last year repeats, as a reference-rate path does).

**What the words mean depends on the market**, and agreements in each market
write them differently. ``CONVENTIONS`` holds three ways of defining them,
chosen for the deal (``covenant_convention``) or from its country:

* ``us`` -- credit agreements in the US: net debt (less cash), EBITDA after
  rent (operating leases are not debt), interest paid in cash, and the
  fixed-charge cover ratio, (EBITDA - capex - cash taxes) / (cash interest +
  scheduled principal);
* ``lma`` -- the Loan Market Association's leveraged facilities (Europe and
  the UK, and the Asia Pacific association's documents, which follow them):
  net debt, the accounts frozen before IFRS 16 so leases are neither debt nor
  added back, and cashflow cover, (EBITDA - capex - tax - the change in
  working capital) / (cash interest + scheduled principal);
* ``india`` -- Indian bank lending: gross debt, the accounts as reported (Ind
  AS 116 leases count as the deal prices them), interest as the P&L books it,
  and the debt service coverage ratio, (profit after tax + D&A + interest) /
  (cash interest + scheduled principal).

These are how agreements in each market are commonly written, not any one
agreement: the screen says the facility agreement's own definitions decide.
Nothing here changes the deal: no covenant traps cash or accelerates a loan,
so the model and the simulation run exactly as without them, and a covenant
only reads their figures. A shareholder loan is the sponsor's money and is
left out of the debt every convention counts.

One function, :func:`evaluate`, tests one deal's figures (arrays of years) and
the simulation's paths (arrays of paths by years) alike, as the tax rules do
(``lbo_engine/tax.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from core.money import to_millions

METRICS = ("leverage", "interest_cover", "fixed_charge_cover", "liquidity")
# A maximum for leverage; a minimum for the rest
MAXIMUMS = frozenset({"leverage"})
SCOPES = ("total", "senior")

# How each facility ranks, by its kind (core.debt.TRANCHE_TYPES). "sponsor" is
# the sponsor's own money, never counted as debt by a lender.
SENIORITY: Mapping[str, str] = {
    "revolver": "senior_secured",
    "amortising_term_loan": "senior_secured",
    "institutional_term_loan": "senior_secured",
    "unitranche": "senior_secured",
    "second_lien": "second_lien",
    "senior_notes": "senior_unsecured",
    "pik_notes": "subordinated",
    "vendor_loan": "subordinated",
    "shareholder_loan": "sponsor",
}
# A deal sized by percentages has the engine's two tranches
# (lbo_engine/capital_structure.py): a senior term loan and a mezzanine
PERCENTAGE_FACILITIES = (("Senior Term Loan", "senior_secured"), ("Mezzanine", "subordinated"))


@dataclass(frozen=True)
class Convention:
    """How a market's agreements define the four tests."""
    net_debt: bool              # debt less the cash on the balance sheet
    leases_as_reported: bool    # leases count as the deal prices them (else frozen before IFRS 16)
    interest: str               # "cash" (paid, with commitment fees) or "expense" (as the P&L books it)
    fixed_charge: str           # "fccr" | "cashflow_cover" | "dscr"


CONVENTIONS: Mapping[str, Convention] = {
    "us": Convention(net_debt=True, leases_as_reported=False, interest="cash", fixed_charge="fccr"),
    "lma": Convention(net_debt=True, leases_as_reported=False, interest="cash", fixed_charge="cashflow_cover"),
    "india": Convention(net_debt=False, leases_as_reported=True, interest="expense", fixed_charge="dscr"),
}
# The US and its tax havens write US agreements (S&P's "us" region); India its own
US_COUNTRIES = frozenset({"US", "BM", "KY"})


def convention_for(chosen: str, country: str) -> str:
    """The convention a deal is tested under: its own choice, else its
    country's (the LMA's anywhere but the US and India)."""
    if chosen:
        return chosen
    code = (country or "").strip().upper()
    if code in US_COUNTRIES:
        return "us"
    return "india" if code == "IN" else "lma"


@dataclass(frozen=True)
class Covenant:
    """One test in one facility's agreement. ``levels`` is the maximum
    (leverage, x) or minimum (covers, x; liquidity, money) a year."""
    facility: str
    metric: str
    levels: tuple = ()
    scope: str = "total"        # leverage: every lender's debt, or the senior secured only

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, not {self.metric!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}, not {self.scope!r}")
        object.__setattr__(self, "levels", tuple(float(x) for x in self.levels))
        if not self.levels:
            raise ValueError("a covenant needs a level")

    def level_in_year(self, year: int) -> float:
        """The level for ``year`` (1-indexed); a short path repeats its last year."""
        return self.levels[min(year, len(self.levels)) - 1]

    def in_millions(self, unit: str) -> "Covenant":
        """A liquidity level is money; the ratios are not."""
        if self.metric != "liquidity" or unit == "millions":
            return self
        return replace(self, levels=tuple(to_millions(x, unit) for x in self.levels))


def coerce(covenants) -> tuple:
    """Covenants from the API's dicts (or already built)."""
    return tuple(c if isinstance(c, Covenant) else Covenant(**c) for c in covenants or ())


@dataclass(frozen=True)
class Facility:
    """A facility as a lender sees it. ``commitment`` is a revolver's limit
    (None for a loan drawn in full); ``maturity`` its final year."""
    name: str
    seniority: str
    maturity: int
    commitment: Optional[float] = None
    kind: Optional[str] = None
    # Its regular amortisation a year: what debt service counts. A bullet, or
    # what is left at maturity, is refinanced rather than paid from the year's
    # cash, and agreements leave it out of the fixed charges.
    amortisation: tuple = ()


def regular_amortisation(amort_type: str, amount: float, amort_pct: float, schedule, years: int) -> tuple:
    """A facility's regular repayments for years 1..``years``, from the
    engine's own amortisation fields (``lbo_engine.capital_structure.Tranche``)."""
    if amort_type == "amortizing":
        return tuple(amount * amort_pct for _ in range(years))
    if amort_type == "custom":
        return tuple(float(schedule[y]) if y < len(schedule) else 0.0 for y in range(years))
    return tuple(0.0 for _ in range(years))


@dataclass
class Figures:
    """The year-end figures the tests read, in millions. Every array's last
    axis is the year; ``opening``, ``closing`` and ``scheduled`` have the
    facility axis before it, in the order of the facilities passed with them.
    One deal: shape (years,) and (facilities, years); the simulation: a
    leading path axis."""
    ebitda: np.ndarray          # after lease costs (the model's operating EBITDA)
    net_income: np.ndarray
    da: np.ndarray
    interest_expense: np.ndarray
    cash_interest: np.ndarray   # paid in cash, commitment fees with it
    taxes: np.ndarray
    capex: np.ndarray
    delta_nwc: np.ndarray
    cash: np.ndarray            # the balance sheet's cash at the year end
    opening: np.ndarray         # each facility's balance at the start of the year
    closing: np.ndarray         # ... and at its end
    scheduled: np.ndarray       # each facility's scheduled repayment in the year
    lease_addback: float = 0.0  # what the deal adds back to value its EBITDA (post-IFRS 16 view)
    lease_liability: float = 0.0  # the lease liability the deal counts as debt


def _ratio(num, den):
    """num / den; a charge of nothing is covered without limit (+inf)."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    shape = np.broadcast(num, den).shape
    return np.divide(num, den, out=np.full(shape, np.inf), where=den > 0)


def _leverage(debt, ebitda):
    """Debt / EBITDA; no debt is no leverage, and debt against an EBITDA that
    is not positive is unlimited (+inf)."""
    debt, ebitda = np.asarray(debt, dtype=float), np.asarray(ebitda, dtype=float)
    shape = np.broadcast(debt, ebitda).shape
    out = np.divide(debt, ebitda, out=np.full(shape, np.inf), where=ebitda > 0)
    # Net debt can be negative (more cash than debt): a negative leverage
    # passes any maximum, whatever the EBITDA
    return np.where((debt <= 0) & ~(ebitda > 0), 0.0, out)


def metric_values(cov: Covenant, conv: Convention, facilities: Sequence[Facility], f: Figures) -> np.ndarray:
    """The covenant's figure each year (and path), as ``conv`` defines it."""
    ebitda = f.ebitda + (f.lease_addback if conv.leases_as_reported else 0.0)
    lending = np.array([fac.seniority != "sponsor" for fac in facilities], dtype=bool)
    if cov.metric == "leverage":
        counted = lending & np.array([cov.scope == "total" or fac.seniority == "senior_secured"
                                      for fac in facilities], dtype=bool)
        debt = (f.closing * counted[:, None]).sum(axis=-2) if facilities else np.zeros_like(f.ebitda)
        if conv.leases_as_reported:
            debt = debt + f.lease_liability
        if conv.net_debt:
            debt = debt - f.cash
        return _leverage(debt, ebitda)
    if cov.metric == "interest_cover":
        interest = f.cash_interest if conv.interest == "cash" else f.interest_expense
        return _ratio(ebitda, interest)
    if cov.metric == "fixed_charge_cover":
        years = f.ebitda.shape[-1]
        regular = np.array([[fac.amortisation[y] if y < len(fac.amortisation) else 0.0 for y in range(years)]
                            for fac in facilities], dtype=float).reshape(len(facilities), years)
        paid = np.minimum(f.scheduled, regular)
        scheduled = (paid * lending[:, None]).sum(axis=-2) if facilities else np.zeros_like(f.ebitda)
        service = f.cash_interest + scheduled
        if conv.fixed_charge == "fccr":
            flow = ebitda - f.capex - f.taxes
        elif conv.fixed_charge == "cashflow_cover":
            flow = ebitda - f.capex - f.taxes - f.delta_nwc
        else:   # dscr
            flow = f.net_income + f.da + f.interest_expense
        return _ratio(flow, service)
    # liquidity: cash plus what the committed lines still offer at the year end
    undrawn = np.zeros_like(f.cash, dtype=float)
    for i, fac in enumerate(facilities):
        if fac.commitment is not None:
            undrawn = undrawn + np.maximum(fac.commitment - f.closing[..., i, :], 0.0)
    return f.cash + undrawn


def tested(cov: Covenant, facilities: Sequence[Facility], f: Figures) -> np.ndarray:
    """Whether the covenant tests each year: while its facility has a balance
    at the start of the year, or is a committed line not yet at maturity. A
    covenant naming no facility of the deal never tests."""
    years = f.ebitda.shape[-1]
    for i, fac in enumerate(facilities):
        if fac.name == cov.facility:
            live = f.opening[..., i, :] > 1e-9
            if fac.commitment is not None:
                live = live | (np.arange(1, years + 1) <= fac.maturity)
            return np.broadcast_to(live, f.ebitda.shape)
    return np.zeros(f.ebitda.shape, dtype=bool)


@dataclass
class Tested:
    """One covenant's tests: each year's figure, level, whether it tested and
    whether it was breached (arrays shaped like the figures)."""
    covenant: Covenant
    value: np.ndarray
    level: np.ndarray
    tested: np.ndarray
    breach: np.ndarray


def evaluate(covenants: Sequence[Covenant], convention: str, facilities: Sequence[Facility],
             f: Figures) -> list:
    """Every covenant's yearly tests (:class:`Tested`)."""
    conv = CONVENTIONS[convention]
    years = f.ebitda.shape[-1]
    out = []
    for cov in covenants:
        value = metric_values(cov, conv, facilities, f)
        level = np.array([cov.level_in_year(y) for y in range(1, years + 1)])
        live = tested(cov, facilities, f)
        broken = value > level if cov.metric in MAXIMUMS else value < level
        out.append(Tested(cov, value, level, live, live & broken))
    return out


def headroom(metric: str, value: float, level: float) -> Optional[float]:
    """How far the figure is from the level, as a share of the level
    (negative when breached); None where either is unlimited or the level is 0."""
    if not (np.isfinite(value) and np.isfinite(level)) or level == 0:
        return None
    return float((level - value) / level if metric in MAXIMUMS else (value - level) / level)


# ---------------------------------------------------------------------------
# The deal model's figures
# ---------------------------------------------------------------------------
def _facility(t, seniority: str, years: int, kind: Optional[str]) -> Facility:
    """A facility from the engine's or the simulation's tranche."""
    return Facility(name=t.name, seniority=seniority, maturity=int(t.maturity_years), commitment=t.commitment,
                    kind=kind, amortisation=regular_amortisation(t.amort_type, t.amount, t.amort_pct,
                                                                 t.amort_schedule, years))


def deal_facilities(d, result) -> list:
    """The deal's facilities in the debt schedule's order: its listed
    tranches, or the engine's senior loan and mezzanine."""
    from core.debt import unique_names
    years = int(result.params.holding_period)
    kinds = dict(zip(unique_names(d.tranches), (s.kind for s in d.tranches)))
    ranks = dict(PERCENTAGE_FACILITIES)
    by_name = {t.name: t for t in result.capital_structure.tranches}
    out = []
    for name in result.debt_schedule.schedule:
        kind = kinds.get(name) if d.tranches else None
        out.append(_facility(by_name[name], SENIORITY[kind] if kind else ranks.get(name, "senior_secured"),
                             years, kind))
    return out


def deal_figures(d, result) -> Figures:
    """One deal run's figures for the tests, in millions."""
    from core.deal import leases_of
    om, cf, ds = result.operating_model, result.cash_flow, result.debt_schedule
    rows = list(ds.schedule.values())

    def per_facility(attr):
        return np.array([[getattr(r, attr) for r in records] for records in rows], dtype=float) \
            if rows else np.zeros((0, len(ds.years)))

    terms = leases_of(d)
    return Figures(
        ebitda=np.asarray(om.ebitda, dtype=float), net_income=np.asarray(om.net_income, dtype=float),
        da=np.asarray(om.da, dtype=float), interest_expense=np.asarray(om.interest_expense, dtype=float),
        cash_interest=np.asarray(ds.total_cash_interest, dtype=float), taxes=np.asarray(om.taxes, dtype=float),
        capex=np.asarray(cf.capex, dtype=float), delta_nwc=np.asarray(cf.delta_nwc, dtype=float),
        cash=np.asarray(ds.cash_balance, dtype=float), opening=per_facility("beginning_balance"),
        closing=per_facility("ending_balance"), scheduled=per_facility("mandatory_repayment"),
        lease_addback=terms.valuation_addback, lease_liability=terms.debt_like,
    )


def deal_view(d, result) -> dict:
    """The deal's yearly tests, in the shape the API answers (millions; the
    router hands money back in the deal's unit). A liquidity test's figures
    are ``liquidity``/``minimum_liquidity`` (money), a ratio's
    ``value``/``level``."""
    convention = convention_for(d.covenant_convention, d.country)
    facilities = deal_facilities(d, result)
    read = deal_figures(d, result)
    results = evaluate(d.covenants, convention, facilities, read)
    out = []
    for i, r in enumerate(results):
        money = r.covenant.metric == "liquidity"
        years = []
        for t in range(len(r.value)):
            value, level = float(r.value[t]), float(r.level[t])
            figures = ({"liquidity": value, "minimum_liquidity": level} if money else
                       {"value": value if np.isfinite(value) else None, "level": level})
            years.append({"year": t + 1, **figures, "tested": bool(r.tested[t]), "breach": bool(r.breach[t]),
                          "headroom": headroom(r.covenant.metric, value, level) if r.tested[t] else None})
        first = next((t + 1 for t in range(len(r.breach)) if r.breach[t]), None)
        out.append({"index": i, "facility": r.covenant.facility, "metric": r.covenant.metric,
                    "scope": r.covenant.scope, "years": years, "first_breach": first})
    firsts = [c["first_breach"] for c in out if c["first_breach"] is not None]
    return {"convention": convention, "chosen": bool(d.covenant_convention), "covenants": out,
            "first_breach": min(firsts) if firsts else None,
            "year_one": year_one(convention, facilities, read)}


def year_one(convention: str, facilities: Sequence[Facility], f: Figures) -> dict:
    """Each test's year-one figure under the convention, whatever covenants
    the deal has: where the screen starts a new covenant's level."""
    conv = CONVENTIONS[convention]
    probe = {"leverage": Covenant("", "leverage", (0.0,)),
             "senior_leverage": Covenant("", "leverage", (0.0,), "senior"),
             "interest_cover": Covenant("", "interest_cover", (0.0,)),
             "fixed_charge_cover": Covenant("", "fixed_charge_cover", (0.0,)),
             "liquidity": Covenant("", "liquidity", (0.0,))}
    out = {}
    for key, cov in probe.items():
        v = float(metric_values(cov, conv, facilities, f)[0]) if f.ebitda.shape[-1] else float("nan")
        out[key] = v if np.isfinite(v) else None
    return out


# ---------------------------------------------------------------------------
# The simulation's paths
# ---------------------------------------------------------------------------
def simulated_figures(paths: Mapping, params) -> Figures:
    """The simulation's tracked paths (``lender_paths``) as figures."""
    return Figures(
        ebitda=paths["ebitda"], net_income=paths["net_income"], da=paths["da"],
        interest_expense=paths["interest"], cash_interest=paths["cash_interest"], taxes=paths["taxes"],
        capex=paths["capex"], delta_nwc=paths["delta_nwc"], cash=paths["cash"],
        opening=paths["opening"], closing=paths["closing"], scheduled=paths["scheduled"],
        lease_addback=params.lease_ebitda_addback, lease_liability=params.lease_liability,
    )


def simulation_facilities(d, params) -> list:
    """The facilities in the simulation's order: the deal's tranches, or the
    two-bucket path's senior loan and mezzanine."""
    years = int(params.holding_period)
    if params.tranches:
        return [_facility(t, SENIORITY[spec.kind], years, spec.kind)
                for spec, t in zip(d.tranches, params.tranches)]
    # The two-bucket path: the senior loan amortises, the mezzanine is a bullet
    entry_ev = (params.entry_ebitda + params.lease_ebitda_addback) * params.entry_multiple
    senior = entry_ev * params.debt_pct * params.senior_pct
    (senior_name, senior_rank), (mezz_name, mezz_rank) = PERCENTAGE_FACILITIES
    return [Facility(name=senior_name, seniority=senior_rank, maturity=years,
                     amortisation=tuple(senior * params.senior_amort_pct for _ in range(years))),
            Facility(name=mezz_name, seniority=mezz_rank, maturity=years, amortisation=(0.0,) * years)]


def breach_odds(d, sim) -> Optional[dict]:
    """Each covenant's chance of breach a year over the simulated paths, and
    the chance it has breached by then; the same for any covenant. None for
    a deal without covenants or a run that tracked no paths. Drops the paths
    afterwards (they are the largest arrays a run holds)."""
    paths, sim.lender_paths = sim.lender_paths, None
    if not d.covenants or paths is None:
        return None
    convention = convention_for(d.covenant_convention, d.country)
    results = evaluate(d.covenants, convention, simulation_facilities(d, sim.params),
                       simulated_figures(paths, sim.params))
    n = paths["ebitda"].shape[0]

    def by_year(breach):
        by_now = np.logical_or.accumulate(breach, axis=-1)
        return [{"year": t + 1, "breach": float(breach[:, t].mean()), "breached_by": float(by_now[:, t].mean())}
                for t in range(breach.shape[-1])]

    covenants = [{"index": i, "facility": r.covenant.facility, "metric": r.covenant.metric,
                  "tested": [float(r.tested[:, t].mean()) for t in range(r.tested.shape[-1])],
                  "years": by_year(r.breach)} for i, r in enumerate(results)]
    anything = np.logical_or.reduce([r.breach for r in results])
    return {"convention": convention, "paths": int(n), "covenants": covenants, "any": by_year(anything)}
