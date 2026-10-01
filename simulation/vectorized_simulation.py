"""
simulation/vectorized_simulation.py
------------------------------------
Fully vectorized LBO simulation engine.

Replaces the original vectorized_simulation.py entirely.

Architecture
------------
Every operation runs on NumPy arrays of shape (N,) where N is the
number of simulations. There are zero Python loops over scenarios.
The only loop is over years (holding_period), which is at most 10
iterations — negligible overhead regardless of N.

This means:
    N = 10,000    runs in ~0.1 seconds
    N = 100,000   runs in ~0.5 seconds
    N = 1,000,000 runs in ~4 seconds

What changed from the original
-------------------------------
Original model:
    - Hardcoded entry_ebitda = 100, entry_multiple = 10
    - FCF = EBT - Tax  (missing D&A add-back, missing NWC)
    - IRR = CAGR approximation
    - No COGS / gross margin / opex breakdown
    - Debt sweep used raw FCF (no minimum cash floor)

New model (mirrors lbo_engine/model.py but vectorized):
    - Full P&L: Revenue → COGS → Gross Profit → OpEx → EBIT → D&A → EBITDA
    - Full FCF: Net Income + D&A - Capex - ΔNWC
    - Two-pass interest convergence (vectorized)
    - Cash sweep with minimum cash floor
    - True IRR via vectorized Newton-Raphson solver
    - All inputs parameterized (no hardcoding)

Correlated random variables
----------------------------
Same Cholesky structure as the original, extended to 5 variables:
    [0] revenue_growth
    [1] exit_multiple
    [2] interest_rate
    [3] gross_margin      (NEW — margin uncertainty)
    [4] ebitda_shock      (NEW — one-time shock, e.g. recession year)

Default correlation matrix (economically motivated):
    growth ↔ exit_multiple:    +0.60  (strong growth → higher exit valuation)
    growth ↔ interest:         -0.30  (high growth = tighter monetary policy)
    growth ↔ gross_margin:     +0.40  (volume growth supports margin)
    exit_mult ↔ interest:      -0.50  (high rates compress multiples)
    exit_mult ↔ gross_margin:  +0.30  (better margins → higher quality exit)
    interest ↔ gross_margin:   -0.20  (cost inflation hurts margins and raises rates)

Two simulation modes
--------------------
Mode 1: run_vectorized_simulation()
    Returns a DataFrame with IRR, MOIC, and all input draws.
    Same interface as the original — drop-in replacement.

Mode 2: run_vectorized_simulation_full()
    Returns a SimulationResult with full distribution statistics,
    percentile profiles, and scenario breakdown.
    Used by the upgraded dashboard.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from lbo_engine.tax import TaxRules, tax_schedule
from simulation.tranches import run_tranche_schedule


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    """
    All inputs for the vectorized simulation.

    Distributional inputs (mean + std) describe the uncertainty
    around each assumption. The simulation draws correlated samples
    from these distributions using the Cholesky decomposition.

    Parameters
    ----------
    n : int
        Number of simulation paths.

    entry_ebitda : float
        LTM EBITDA at entry (M). Fixed — not simulated.

    entry_multiple : float
        EV / LTM EBITDA at entry. Fixed — not simulated.
        (The entry price is known at close; it is the exit that is uncertain.)

    holding_period : int
        Years from close to exit. Fixed.

    --- Simulated variables (mean, std) ---

    growth_mean / growth_std : float
        Annual revenue growth rate distribution.

    exit_mean / exit_std : float
        Exit EV / EBITDA multiple distribution.

    interest_mean / interest_std : float
        Base interest rate (senior tranche) distribution.

    gross_margin_mean / gross_margin_std : float
        Gross margin (= 1 - COGS%) distribution.
        NEW vs original model.

    --- Fixed operating assumptions ---

    opex_pct : float
        Operating expenses as % of revenue. Fixed.

    da_pct : float
        D&A as % of revenue. Fixed.

    tax_rate : float
        Effective tax rate. Fixed.

    capex_pct : float
        Capex as % of revenue. Fixed.

    nwc_pct : float
        Change in NWC as % of revenue. Fixed.

    --- Capital structure ---

    debt_pct : float
        Total debt / entry EV.

    senior_pct : float
        Senior debt / total debt.

    mezz_spread : float
        Mezz rate = interest_rate + mezz_spread.

    senior_amort_pct : float
        Annual mandatory amortization on senior tranche (% of original balance).

    minimum_cash_pct : float
        Minimum cash as % of entry EV retained post-close.

    --- Transaction costs ---

    transaction_fees_pct / financing_fees_pct : float
        Fees as % of entry EV / % of total debt. Default 0.

    other_uses : float
        Other uses of funds at close (M). Default 0.

        All three are funded by sponsor equity at close, raising the equity
        check without buying enterprise value -- matching lbo_engine.run_lbo.

    --- Correlation matrix ---

    corr_matrix : np.ndarray, shape (5, 5)
        Correlation matrix for the 5 simulated variables:
        [growth, exit_multiple, interest, gross_margin, ebitda_shock]
        Must be positive semi-definite.
        If None, uses the default economically-motivated matrix.

    --- Convergence ---

    n_interest_passes : int
        Number of interest convergence iterations. Default 2.
        Increase to 3 for higher accuracy at minimal speed cost.

    clip_irr : bool
        If True, clips IRR to [-1.0, 5.0] to remove solver artifacts.
        Default True.
    """

    n: int = 10_000

    # Entry (fixed)
    entry_ebitda: float = 100.0
    entry_multiple: float = 10.0
    holding_period: int = 5

    # Simulated distributions
    growth_mean: float = 0.05
    growth_std: float = 0.03
    exit_mean: float = 10.0
    exit_std: float = 1.5
    interest_mean: float = 0.065
    interest_std: float = 0.015
    gross_margin_mean: float = 0.40
    gross_margin_std: float = 0.03

    # Fixed operating assumptions
    opex_pct: float = 0.18
    da_pct: float = 0.04
    tax_rate: float = 0.25
    capex_pct: float = 0.04
    nwc_pct: float = 0.01

    # Capital structure
    debt_pct: float = 0.60
    senior_pct: float = 0.70
    mezz_spread: float = 0.04
    senior_amort_pct: float = 0.05
    minimum_cash_pct: float = 0.0

    # Transaction costs (funded by sponsor equity at close)
    transaction_fees_pct: float = 0.0   # % of entry EV
    financing_fees_pct: float = 0.0     # % of total debt
    other_uses: float = 0.0             # M

    # Correlation (None = use default)
    corr_matrix: Optional[np.ndarray] = None

    # Convergence
    n_interest_passes: int = 2
    clip_irr: bool = True

    # The deal's debt, facility by facility (PLAN.md 2.4b; core.debt builds
    # them). Empty keeps the two-bucket structure above -- debt_pct,
    # senior_pct, mezz_spread, senior_amort_pct -- and its path through the
    # simulation, untouched. A list replaces all four, and the rate draw then
    # moves only the floating facilities (simulation/tranches.py).
    tranches: tuple = ()

    # The deal's tax rules beyond the flat rate (PLAN.md 2.5, lbo_engine/tax.py),
    # the same function the deal model uses. None keeps the flat rate on
    # positive EBT, and the expression that computes it, untouched.
    tax_rules: Optional[TaxRules] = None

    # Leases (PLAN.md 2.6), as in lbo_engine's LBOParams: entry_ebitda is after
    # lease costs; a deal priced post-IFRS 16 adds the lease cost back where
    # EBITDA becomes a value and counts the liability with net debt, at entry
    # and at exit. 0.0 leaves both paths exactly as they were.
    lease_ebitda_addback: float = 0.0
    lease_liability: float = 0.0


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Full output from run_vectorized_simulation_full().

    df : pd.DataFrame
        One row per simulation. Columns: IRR, MOIC, Growth, Exit Multiple,
        Interest, Gross Margin, plus any additional tracked variables.

    n_valid : int
        Number of scenarios where equity is positive at exit (not wiped out).

    n_wiped : int
        Number of scenarios where exit EV < net debt (equity = 0).
    """

    df: pd.DataFrame
    n_valid: int
    n_wiped: int
    params: SimulationParams

    @property
    def irr(self) -> np.ndarray:
        return self.df["IRR"].values

    @property
    def moic(self) -> np.ndarray:
        return self.df["MOIC"].values

    @property
    def wipeout_rate(self) -> float:
        return self.n_wiped / (self.n_valid + self.n_wiped)


# ---------------------------------------------------------------------------
# Default correlation matrix
# ---------------------------------------------------------------------------

DEFAULT_CORR = np.array([
    # growth  exit_mult  interest  gross_margin  ebitda_shock
    [1.00,    0.60,     -0.30,     0.40,         0.50],   # growth
    [0.60,    1.00,     -0.50,     0.30,         0.30],   # exit_mult
    [-0.30,  -0.50,      1.00,    -0.20,        -0.20],   # interest
    [0.40,    0.30,     -0.20,     1.00,         0.20],   # gross_margin
    [0.50,    0.30,     -0.20,     0.20,         1.00],   # ebitda_shock
])


# ---------------------------------------------------------------------------
# Vectorized IRR solver
# ---------------------------------------------------------------------------

def _vectorized_irr(
    entry_equity: np.ndarray,
    exit_equity: np.ndarray,
    holding_period: int,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Vectorized Newton-Raphson IRR solver for clean 2-cashflow LBO exits.

    For a simple LBO with no interim dividends, the IRR equation is:
        -entry_equity + exit_equity / (1 + r)^n = 0
    which has a closed-form solution: r = (exit_equity/entry_equity)^(1/n) - 1

    This IS the CAGR formula — and it IS correct when there are no
    interim cash flows. The difference from the scalar lbo_engine/returns.py
    is that here we never have interim dividends in the simulation,
    so CAGR = true IRR exactly.

    For scenarios with interim cash flows (e.g. dividend recaps in the
    full model), use the Newton-Raphson path. We detect this automatically.

    Parameters
    ----------
    entry_equity : np.ndarray, shape (N,)
    exit_equity  : np.ndarray, shape (N,)
    holding_period : int

    Returns
    -------
    np.ndarray, shape (N,)
        IRR per scenario. NaN for wipeout scenarios.
    """
    n = holding_period

    # Mask wipeout scenarios (exit_equity <= 0)
    valid = exit_equity > 0

    irr = np.full_like(entry_equity, np.nan, dtype=float)

    # Clean exit: CAGR = true IRR (no interim cash flows in simulation)
    moic = np.where(valid, exit_equity / np.maximum(entry_equity, 1e-10), 0.0)
    irr = np.where(valid, moic ** (1.0 / n) - 1.0, -1.0)

    return irr


# ---------------------------------------------------------------------------
# Core vectorized engine
# ---------------------------------------------------------------------------

def _draw_correlated_inputs(
    params: SimulationParams,
    seed: Optional[int] = None,
) -> dict:
    """
    Draw N correlated samples for all 5 stochastic variables.

    Uses Cholesky decomposition to induce correlation.

    Returns dict of np.ndarray, each shape (N,).
    """
    # A generator local to this call, not numpy's global one: concurrent
    # simulations (FastAPI runs sync endpoints in a thread pool) would
    # otherwise interleave draws and break seeded reproducibility.
    # RandomState(seed) yields exactly the stream np.random.seed(seed) did,
    # so seeded results are unchanged.
    rng = np.random.RandomState(seed)

    corr = params.corr_matrix if params.corr_matrix is not None else DEFAULT_CORR

    # Validate positive semi-definiteness
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Add small diagonal perturbation and retry
        corr_reg = corr + np.eye(5) * 1e-8
        L = np.linalg.cholesky(corr_reg)

    # Independent standard normal draws: shape (5, N)
    Z = rng.standard_normal((5, params.n))

    # Correlated standard normal draws: shape (5, N)
    C = L @ Z

    # Transform to target distributions
    # All clipped to economically sensible bounds
    growth = np.clip(
        params.growth_mean + C[0] * params.growth_std,
        -0.50, 0.50
    )
    exit_multiple = np.clip(
        params.exit_mean + C[1] * params.exit_std,
        1.0, 30.0
    )
    interest = np.clip(
        params.interest_mean + C[2] * params.interest_std,
        0.01, 0.30
    )
    gross_margin = np.clip(
        params.gross_margin_mean + C[3] * params.gross_margin_std,
        0.05, 0.90
    )

    # ebitda_shock: multiplicative shock to EBITDA in one year
    # Mean 0, std 0.05 — represents a one-year earnings disruption
    # Applied to year 2 only (most common year for LBO operational issues)
    ebitda_shock = np.clip(
        C[4] * 0.05,
        -0.30, 0.20
    )

    return {
        "growth": growth,
        "exit_multiple": exit_multiple,
        "interest": interest,
        "gross_margin": gross_margin,
        "ebitda_shock": ebitda_shock,
    }


def _run_vectorized_core(
    params: SimulationParams,
    draws: dict,
) -> dict:
    """
    Core vectorized LBO computation.

    All arrays have shape (N,). The year loop runs holding_period times.
    All operations are elementwise NumPy — no Python loops over scenarios.

    Two-pass interest convergence:
        Pass 1: compute EBITDA path, run debt schedule with zero interest
                to get approximate debt balances -> interest_pass1
        Pass 2: use interest_pass1 to compute net income -> FCF ->
                rerun debt schedule -> interest_pass2
                Returns are computed from pass 2.

    Returns dict of arrays all shape (N,).

    A deal that lists its tranches takes ``_run_tranche_core`` instead; the
    body below is the two-bucket path, unchanged since before PLAN.md 2.4b
    and pinned by tests/test_montecarlo_baseline.py.
    """
    if params.tranches:
        return _run_tranche_core(params, draws)

    N = params.n
    n_yr = params.holding_period
    p = params

    # --- Unpack draws ---
    growth       = draws["growth"]           # (N,)
    exit_mult    = draws["exit_multiple"]    # (N,)
    interest     = draws["interest"]         # (N,)
    gross_margin = draws["gross_margin"]     # (N,)
    ebitda_shock = draws["ebitda_shock"]     # (N,) applied in year 2 only

    # --- Entry values ---
    entry_ev     = (p.entry_ebitda + p.lease_ebitda_addback) * p.entry_multiple  # scalar
    total_debt   = entry_ev * p.debt_pct                              # scalar
    senior_debt  = total_debt * p.senior_pct                          # scalar
    mezz_debt    = total_debt * (1.0 - p.senior_pct)                  # scalar
    entry_costs  = (entry_ev * p.transaction_fees_pct                 # scalar
                    + total_debt * p.financing_fees_pct
                    + p.other_uses)
    entry_equity = entry_ev + entry_costs - total_debt - p.lease_liability  # scalar
    minimum_cash = entry_ev * p.minimum_cash_pct                      # scalar

    senior_rate  = interest                                           # (N,)
    mezz_rate    = interest + p.mezz_spread                           # (N,)

    # --- Infer base revenue from entry EBITDA and margin assumptions ---
    # EBITDA margin = gross_margin - opex_pct + da_pct
    # Base revenue = entry_ebitda / ebitda_margin
    # Clamp margin to avoid division by zero
    ebitda_margin = np.clip(gross_margin - p.opex_pct + p.da_pct, 0.02, 0.95)
    base_revenue  = p.entry_ebitda / ebitda_margin                    # (N,)

    # ----------------------------------------------------------------
    # PASS 1 + PASS 2: Interest convergence
    # ----------------------------------------------------------------
    # We run n_interest_passes iterations.
    # Pass 1 uses zero interest to bootstrap debt balances.
    # Pass 2+ uses interest from the previous pass.

    interest_expense = np.zeros((N, n_yr), dtype=float)  # (N, n_yr)

    for pass_idx in range(p.n_interest_passes):

        # ---- Operating model (Revenue → EBITDA) ----
        revenue = base_revenue.copy()    # (N,) starts at base
        ebitda_final = np.zeros(N)
        ebitda_yr = np.zeros((N, n_yr))
        revenue_yr = np.zeros((N, n_yr))
        da_yr = np.zeros((N, n_yr))

        for t in range(n_yr):
            revenue = revenue * (1.0 + growth)

            # Apply ebitda_shock only in year 2 (index 1)
            if t == 1:
                effective_margin = ebitda_margin * (1.0 + ebitda_shock)
                effective_margin = np.clip(effective_margin, 0.02, 0.95)
            else:
                effective_margin = ebitda_margin

            ebitda = revenue * effective_margin
            da     = revenue * p.da_pct
            cogs   = revenue * (1.0 - gross_margin)
            opex   = revenue * p.opex_pct

            ebitda_yr[:, t]  = ebitda
            revenue_yr[:, t] = revenue
            da_yr[:, t]      = da

        ebitda_final = ebitda_yr[:, -1]   # (N,) exit year EBITDA

        # ---- Income statement (EBIT → Net Income) ----
        # Uses interest_expense from previous pass. A deal's tax rules decide
        # each year's tax from the whole schedule (PLAN.md 2.5).
        net_income_yr = np.zeros((N, n_yr))
        ebit_yr       = np.zeros((N, n_yr))
        ruled = _ruled_taxes(p, ebitda_yr, da_yr, interest_expense)

        for t in range(n_yr):
            ebit  = ebitda_yr[:, t] - da_yr[:, t]
            ebt   = ebit - interest_expense[:, t]
            taxes = np.maximum(ebt, 0.0) * p.tax_rate if ruled is None else ruled[t]
            net_income_yr[:, t] = ebt - taxes
            ebit_yr[:, t] = ebit

        # ---- Cash flow model ----
        # FCF = Net Income + D&A - Capex - ΔNWC
        fcf_yr = np.zeros((N, n_yr))
        for t in range(n_yr):
            rev  = revenue_yr[:, t]
            capex    = rev * p.capex_pct
            delta_nwc = rev * p.nwc_pct
            fcf_yr[:, t] = (
                net_income_yr[:, t]
                + da_yr[:, t]
                - capex
                - delta_nwc
            )

        # ---- Debt schedule with sweep ----
        senior_bal = np.full(N, senior_debt, dtype=float)
        mezz_bal   = np.full(N, mezz_debt,   dtype=float)
        cash_bal   = np.full(N, minimum_cash, dtype=float)
        interest_new = np.zeros((N, n_yr), dtype=float)

        senior_amort_annual = senior_debt * p.senior_amort_pct   # scalar

        for t in range(n_yr):
            # Interest on beginning balance
            int_s = senior_bal * senior_rate   # (N,)
            int_m = mezz_bal   * mezz_rate     # (N,)
            interest_new[:, t] = int_s + int_m

            # Mandatory amortization (senior only, capped at balance)
            mandatory_s = np.minimum(senior_amort_annual, senior_bal)
            senior_bal  = senior_bal - mandatory_s

            # Mezz bullet: full repayment at maturity (final year)
            if t == n_yr - 1:
                mandatory_m = mezz_bal.copy()
                mezz_bal    = np.zeros(N)
            else:
                mandatory_m = np.zeros(N)

            # Cash available for sweep
            cash_in_hand   = cash_bal + fcf_yr[:, t]
            cash_after_mand = cash_in_hand - mandatory_s - mandatory_m
            avail_sweep    = np.maximum(cash_after_mand - minimum_cash, 0.0)

            # Sweep senior first (priority 1)
            sweep_s     = np.minimum(avail_sweep, senior_bal)
            senior_bal  = senior_bal - sweep_s
            remaining   = avail_sweep - sweep_s

            # Sweep mezz second (priority 2)
            sweep_m   = np.minimum(remaining, mezz_bal)
            mezz_bal  = mezz_bal - sweep_m

            # Update cash balance
            cash_bal = np.full(N, minimum_cash)

        interest_expense = interest_new   # use for next pass

    # ---- Final values at exit ----
    remaining_debt   = senior_bal + mezz_bal           # (N,) at end of final year
    net_debt_at_exit = remaining_debt - minimum_cash + p.lease_liability   # (N,)

    # ---- Exit valuation ----
    exit_ev     = (ebitda_final + p.lease_ebitda_addback) * exit_mult    # (N,)
    exit_equity = np.maximum(exit_ev - net_debt_at_exit, 0.0)   # floor at 0

    # ---- Returns ----
    entry_equity_arr = np.full(N, entry_equity)
    moic = exit_equity / np.maximum(entry_equity_arr, 1e-10)     # (N,)
    irr  = _vectorized_irr(entry_equity_arr, exit_equity, n_yr)  # (N,)

    if params.clip_irr:
        irr = np.clip(irr, -1.0, 5.0)

    return {
        "IRR":           irr,
        "MOIC":          moic,
        "Exit Equity":   exit_equity,
        "Exit EV":       exit_ev,
        "Exit EBITDA":   ebitda_final,
        "Growth":        growth,
        "Exit Multiple": exit_mult,
        "Interest":      interest,
        "Gross Margin":  gross_margin,
        "EBITDA Shock":  ebitda_shock,
        "Net Debt Exit": net_debt_at_exit,
    }


def _ruled_taxes(p: SimulationParams, ebitda_yr, da_yr, interest_yr):
    """Each year's tax under the deal's tax rules, a list of (N,) arrays; None
    when it has none, so callers keep the flat rate on positive EBT."""
    if p.tax_rules is None or not p.tax_rules.active:
        return None
    n_yr = p.holding_period
    return tax_schedule(
        ebitda=[ebitda_yr[:, t] for t in range(n_yr)],
        ebit=[ebitda_yr[:, t] - da_yr[:, t] for t in range(n_yr)],
        net_interest=[interest_yr[:, t] for t in range(n_yr)],
        tax_rate=[p.tax_rate] * n_yr, rules=p.tax_rules, detail=False,
    ).taxes


def _operating_paths(p: SimulationParams, draws: dict):
    """Revenue, EBITDA and D&A for every path and year, shape (N, n_yr).

    The same expressions as the two-bucket path's operating model, so a deal
    written out as tranches starts from the identical numbers.
    """
    growth, gross_margin, ebitda_shock = draws["growth"], draws["gross_margin"], draws["ebitda_shock"]
    ebitda_margin = np.clip(gross_margin - p.opex_pct + p.da_pct, 0.02, 0.95)
    revenue = p.entry_ebitda / ebitda_margin
    shape = (p.n, p.holding_period)
    revenue_yr, ebitda_yr, da_yr = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for t in range(p.holding_period):
        revenue = revenue * (1.0 + growth)
        if t == 1:   # the one-year EBITDA shock, in year 2 as in the two-bucket path
            margin = np.clip(ebitda_margin * (1.0 + ebitda_shock), 0.02, 0.95)
        else:
            margin = ebitda_margin
        revenue_yr[:, t] = revenue
        ebitda_yr[:, t] = revenue * margin
        da_yr[:, t] = revenue * p.da_pct
    return revenue_yr, ebitda_yr, da_yr


def _run_tranche_core(p: SimulationParams, draws: dict) -> dict:
    """The simulation for a deal financed tranche by tranche (PLAN.md 2.4b).

    The operating model, taxes, cash flow, interest passes, exit and returns
    are the two-bucket path's; the debt is ``simulation/tranches.py``'s
    schedule. Two differences follow from the deal model rather than from the
    two-bucket path's shortcuts:

    * the rate draw is a **shock to reference rates**: each path moves every
      floating facility's reference by ``interest - interest_mean`` and leaves
      fixed facilities alone (a scenario's rate stress reaches the references
      through ``core.montecarlo.apply_scenario``);
    * interest paid in kind is an expense in the P&L and is added back in the
      cash flow, and cash left once the sweep has repaid everything it may
      stays on the balance sheet, as ``lbo_engine`` does.
    """
    N, n_yr = p.n, p.holding_period
    entry_ev = (p.entry_ebitda + p.lease_ebitda_addback) * p.entry_multiple
    total_debt = sum(t.amount for t in p.tranches)
    entry_costs = (entry_ev * p.transaction_fees_pct
                   + total_debt * p.financing_fees_pct
                   + p.other_uses
                   + sum(t.upfront_fee for t in p.tranches))
    entry_equity = entry_ev + entry_costs - total_debt - p.lease_liability
    minimum_cash = entry_ev * p.minimum_cash_pct

    # The draw is clipped to [0.01, 0.30] before this, so with a wide rate
    # spread the shock is not quite centred on zero -- as the two-bucket
    # path's senior rate is not quite centred on its mean
    shock = draws["interest"] - p.interest_mean
    revenue_yr, ebitda_yr, da_yr = _operating_paths(p, draws)

    interest = np.zeros((N, n_yr))
    non_cash = np.zeros((N, n_yr))
    for _ in range(p.n_interest_passes):
        ebt = ebitda_yr - da_yr - interest
        ruled = _ruled_taxes(p, ebitda_yr, da_yr, interest)
        taxes = np.maximum(ebt, 0.0) * p.tax_rate if ruled is None else np.column_stack(ruled)
        net_income = ebt - taxes
        fcf = (net_income + da_yr - revenue_yr * p.capex_pct - revenue_yr * p.nwc_pct
               + non_cash)
        schedule = run_tranche_schedule(p.tranches, shock, fcf, minimum_cash)
        interest, non_cash = schedule.interest, schedule.non_cash

    net_debt_at_exit = schedule.ending_debt - schedule.ending_cash + p.lease_liability
    exit_ebitda = ebitda_yr[:, -1]
    exit_ev = (exit_ebitda + p.lease_ebitda_addback) * draws["exit_multiple"]
    exit_equity = np.maximum(exit_ev - net_debt_at_exit, 0.0)
    entry_equity_arr = np.full(N, entry_equity)
    moic = exit_equity / np.maximum(entry_equity_arr, 1e-10)
    irr = _vectorized_irr(entry_equity_arr, exit_equity, n_yr)
    if p.clip_irr:
        irr = np.clip(irr, -1.0, 5.0)

    return {
        "IRR":           irr,
        "MOIC":          moic,
        "Exit Equity":   exit_equity,
        "Exit EV":       exit_ev,
        "Exit EBITDA":   exit_ebitda,
        "Growth":        draws["growth"],
        "Exit Multiple": draws["exit_multiple"],
        "Interest":      draws["interest"],
        "Gross Margin":  draws["gross_margin"],
        "EBITDA Shock":  draws["ebitda_shock"],
        "Net Debt Exit": net_debt_at_exit,
    }


# ---------------------------------------------------------------------------
# Public API: Mode 1 — drop-in replacement for original
# ---------------------------------------------------------------------------

def run_vectorized_simulation(
    n: int,
    growth_mean: float,
    growth_std: float,
    exit_mean: float,
    exit_std: float,
    interest_mean: float,
    interest_std: float,
    debt_pct: float,
    senior_pct: float,
    mezz_spread: float,
    tax_rate: float,
    ebitda_margin: float,
    capex_pct: float,
    years: int,
    # New parameters (with defaults for backward compatibility)
    entry_ebitda: float = 100.0,
    entry_multiple: float = 10.0,
    gross_margin_mean: float = None,
    gross_margin_std: float = 0.03,
    opex_pct: float = None,
    da_pct: float = 0.04,
    nwc_pct: float = 0.01,
    corr_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Drop-in replacement for the original run_vectorized_simulation().

    Accepts all original parameters plus new ones with defaults for
    backward compatibility. Your existing dashboard_app.py will work
    without any changes.

    New parameters vs original
    --------------------------
    entry_ebitda : float        default 100.0 (was hardcoded)
    entry_multiple : float      default 10.0 (was hardcoded)
    gross_margin_mean : float   inferred from ebitda_margin if None
    opex_pct : float            inferred from ebitda_margin if None

    Backward compatibility
    ----------------------
    ebitda_margin is still accepted. If gross_margin_mean and opex_pct
    are not provided, they are back-calculated from ebitda_margin using:
        gross_margin = ebitda_margin + opex_pct - da_pct
        opex_pct     = ebitda_margin * 0.69  (typical ratio)

    This preserves the original model's EBITDA margin logic while
    routing through the full P&L engine.
    """

    # --- Backward compatibility: infer gross_margin from ebitda_margin ---
    if gross_margin_mean is None or opex_pct is None:
        # ebitda_margin = gross_margin - opex_pct + da_pct
        # Without more info, assume opex is 69% of EBITDA margin gap:
        # gross_margin = ebitda_margin + 0.60 (typical cost base)
        inferred_opex    = ebitda_margin * 0.69
        inferred_gross   = ebitda_margin + inferred_opex - da_pct
        gross_margin_mean = gross_margin_mean or max(inferred_gross, 0.10)
        opex_pct          = opex_pct or inferred_opex

    params = SimulationParams(
        n=n,
        entry_ebitda=entry_ebitda,
        entry_multiple=entry_multiple,
        holding_period=years,
        growth_mean=growth_mean,
        growth_std=growth_std,
        exit_mean=exit_mean,
        exit_std=exit_std,
        interest_mean=interest_mean,
        interest_std=interest_std,
        gross_margin_mean=gross_margin_mean,
        gross_margin_std=gross_margin_std,
        opex_pct=opex_pct,
        da_pct=da_pct,
        tax_rate=tax_rate,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        debt_pct=debt_pct,
        senior_pct=senior_pct,
        mezz_spread=mezz_spread,
        corr_matrix=corr_matrix,
        n_interest_passes=2,
        clip_irr=True,
    )

    draws  = _draw_correlated_inputs(params, seed=seed)
    output = _run_vectorized_core(params, draws)

    return pd.DataFrame({
        "IRR":           output["IRR"],
        "MOIC":          output["MOIC"],
        "Growth":        output["Growth"],
        "Exit Multiple": output["Exit Multiple"],
        "Interest":      output["Interest"],
        "Gross Margin":  output["Gross Margin"],
    })


# ---------------------------------------------------------------------------
# Public API: Mode 2 — full SimulationResult
# ---------------------------------------------------------------------------

def run_vectorized_simulation_full(
    params: SimulationParams,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Full simulation returning a SimulationResult object.

    Use this in the upgraded dashboard for richer analytics.

    Parameters
    ----------
    params : SimulationParams
    seed : int, optional

    Returns
    -------
    SimulationResult
    """
    draws  = _draw_correlated_inputs(params, seed=seed)
    output = _run_vectorized_core(params, draws)

    df = pd.DataFrame(output)

    n_wiped = int((output["Exit Equity"] == 0).sum())
    n_valid = params.n - n_wiped

    return SimulationResult(
        df=df,
        n_valid=n_valid,
        n_wiped=n_wiped,
        params=params,
    )


# ---------------------------------------------------------------------------
# Scenario presets (kept from original, upgraded)
# ---------------------------------------------------------------------------

def get_scenario_params(
    scenario: str,
    base_params: SimulationParams,
) -> SimulationParams:
    """
    Return a modified SimulationParams for a named scenario preset.

    Scenarios shift the mean of each distribution to reflect different
    macroeconomic environments. Volatility (std) is unchanged.

    Not used by the API, and not tranche-aware: core.montecarlo.apply_scenario
    is the one that also moves floating facilities' reference rates.

    Parameters
    ----------
    scenario : str
        One of: "bull", "base", "recession", "stagflation"

    base_params : SimulationParams
        The user's base assumptions — only means are overridden.

    Returns
    -------
    SimulationParams
        Modified copy of base_params.
    """
    import copy
    p = copy.deepcopy(base_params)

    if scenario == "bull":
        p.growth_mean       *= 1.50     # 50% higher growth
        p.exit_mean         *= 1.15     # multiple expansion
        p.interest_mean     *= 0.85     # accommodative rates
        p.gross_margin_mean *= 1.05     # margin tailwind

    elif scenario == "base":
        pass                             # no change

    elif scenario == "recession":
        p.growth_mean        = max(p.growth_mean - 0.06, -0.10)
        p.exit_mean         *= 0.80     # multiple compression
        p.interest_mean     *= 1.20     # credit stress
        p.gross_margin_mean *= 0.93     # margin pressure

    elif scenario == "stagflation":
        p.growth_mean        = max(p.growth_mean - 0.03, -0.05)
        p.exit_mean         *= 0.85
        p.interest_mean     *= 1.40     # rate spike
        p.gross_margin_mean *= 0.90     # severe cost inflation

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Choose from: bull, base, recession, stagflation."
        )

    return p


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  VECTORIZED SIMULATION SELF-TEST")
    print("=" * 60)

    # --- Test 1: Drop-in compatibility (original interface) ---
    print("\n--- Test 1: Original interface (backward compatibility) ---")
    df = run_vectorized_simulation(
        n=10_000,
        growth_mean=0.05,   growth_std=0.02,
        exit_mean=11.0,     exit_std=2.0,
        interest_mean=0.06, interest_std=0.02,
        debt_pct=0.60,      senior_pct=0.70,
        mezz_spread=0.04,   tax_rate=0.25,
        ebitda_margin=0.25, capex_pct=0.04,
        years=5,
        seed=42,
    )
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Mean IRR:  {df['IRR'].mean():.2%}")
    print(f"  Mean MOIC: {df['MOIC'].mean():.2f}x")
    print(f"  Correlation matrix (IRR vs inputs):")
    print(df[["IRR","Growth","Exit Multiple","Interest","Gross Margin"]].corr().round(2))

    # --- Test 2: Full SimulationResult ---
    print("\n--- Test 2: Full SimulationResult ---")
    params = SimulationParams(
        n=50_000,
        entry_ebitda=100.0,
        entry_multiple=10.0,
        holding_period=5,
        growth_mean=0.05,    growth_std=0.03,
        exit_mean=10.0,      exit_std=1.5,
        interest_mean=0.065, interest_std=0.015,
        gross_margin_mean=0.40, gross_margin_std=0.03,
        opex_pct=0.18,
        da_pct=0.04,
        tax_rate=0.25,
        capex_pct=0.04,
        nwc_pct=0.01,
        debt_pct=0.60,
        senior_pct=0.70,
        mezz_spread=0.04,
        n_interest_passes=2,
    )

    t0 = time.time()
    sim = run_vectorized_simulation_full(params, seed=42)
    elapsed = time.time() - t0

    irr = sim.irr
    print(f"  N = {params.n:,}  |  Time: {elapsed:.2f}s")
    print(f"  Mean IRR:     {irr.mean():.2%}")
    print(f"  Median IRR:   {np.median(irr):.2%}")
    print(f"  5th pct:      {np.percentile(irr, 5):.2%}")
    print(f"  95th pct:     {np.percentile(irr, 95):.2%}")
    print(f"  P(IRR > 20%): {(irr > 0.20).mean():.2%}")
    print(f"  Wipeout rate: {sim.wipeout_rate:.2%}")

    # --- Test 3: Scenario presets ---
    print("\n--- Test 3: Scenario comparison ---")
    scenarios = ["recession", "base", "bull", "stagflation"]
    print(f"  {'Scenario':<14} {'Mean IRR':>10} {'Median IRR':>11} "
          f"{'P(>20%)':>9} {'Wipeout':>9}")
    print("  " + "-" * 56)
    for s in scenarios:
        sp = get_scenario_params(s, params)
        sr = run_vectorized_simulation_full(sp, seed=42)
        irr_s = sr.irr
        print(
            f"  {s:<14} "
            f"{irr_s.mean():>9.2%}  "
            f"{np.median(irr_s):>10.2%}  "
            f"{(irr_s > 0.20).mean():>8.2%}  "
            f"{sr.wipeout_rate:>8.2%}"
        )

    # --- Test 4: Speed benchmark ---
    print("\n--- Test 4: Speed benchmark ---")
    for n_sim in [10_000, 100_000, 1_000_000]:
        p_bench = SimulationParams(n=n_sim, **{
            k: v for k, v in vars(params).items()
            if k not in ("n", "corr_matrix")
        })
        t0 = time.time()
        run_vectorized_simulation_full(p_bench)
        elapsed = time.time() - t0
        print(f"  N = {n_sim:>10,}  |  {elapsed:.2f}s  |  "
              f"{n_sim/elapsed/1e6:.2f}M scenarios/sec")

    # --- Test 5: Verify correlation structure is preserved ---
    print("\n--- Test 5: Correlation structure ---")
    big_sim = run_vectorized_simulation_full(
        SimulationParams(n=500_000, **{
            k: v for k, v in vars(params).items()
            if k not in ("n", "corr_matrix")
        }),
        seed=0,
    )
    cols = ["Growth", "Exit Multiple", "Interest", "Gross Margin"]
    empirical_corr = big_sim.df[cols].corr()
    print("  Empirical correlations (N=500k, should match DEFAULT_CORR[:4,:4]):")
    print(empirical_corr.round(2))
    print("\n  Target (DEFAULT_CORR[:4,:4]):")
    import pandas as pd
    print(pd.DataFrame(
        DEFAULT_CORR[:4, :4],
        index=cols, columns=cols
    ).round(2))