"""3-statement forecasting model.

Moved from pages/forecasting.py without changes to the calculations.
"""
from dataclasses import dataclass, field  # noqa: F401
from typing import List, Optional  # noqa: F401

import numpy as np


@dataclass
class HistoricalYear:
    year:          str
    revenue:       float
    cogs:          float          # negative
    rd:            float          # negative, 0 if N/A
    sga:           float          # negative
    other_income:  float          # flat, can be negative
    interest_exp:  float          # negative
    interest_inc:  float          # positive
    da:            float          # positive (add-back)
    sbc:           float          # positive (add-back)
    tax:           float          # negative
    capex:         float          # positive (cash out)
    dividends:     float          # positive (cash out)
    repurchases:   float          # positive (cash out)
    cash:          float          # balance sheet
    ar:            float
    inventory:     float
    other_current: float
    ppe_net:       float
    other_nca:     float
    ap:            float
    other_cl:      float
    deferred_rev:  float
    ltd:           float          # long-term debt
    common_stock:  float
    retained_earnings: float
    oci:           float
    # Deferred tax, lease, pension and other non-current liabilities. Held
    # flat in the forecast: no revenue driver, so no cash flow of its own.
    other_ncl:     float = 0.0
    # Goodwill, intangibles, long-term investments, lease assets. Held flat for
    # the same reason: they do not scale with revenue or consume cash as it grows.
    other_lta:     float = 0.0


@dataclass
class ForecastAssumptions:
    """Per-year forecast driver assumptions """
    revenue_growth:   float   # decimal e.g. 0.06
    gross_margin:     float   # decimal e.g. 0.38
    rd_pct:           float   # decimal e.g. 0.06
    sga_pct:          float   # decimal e.g. 0.07
    tax_rate:         float   # decimal e.g. 0.167
    da_pct:           float   # decimal e.g. 0.04  (of revenue)
    sbc_pct:          float   # decimal e.g. 0.02
    capex_pct:        float   # decimal e.g. 0.05
    ar_days:          float   # e.g. 45
    inv_days:         float   # e.g. 30
    ap_days:          float   # e.g. 60
    other_cl_pct:     float   # other current liabilities % of revenue
    deferred_rev_pct: float   # deferred revenue % of revenue
    other_nca_pct:    float   # other non-current assets % of revenue
    other_income:     float   # flat $M
    dividends:        float   # flat $M
    repurchases:      float   # flat $M
    ltd_change:       float   # net new borrowing (+ = draw, − = repay)
    interest_rate_cash: float # rate earned on cash balance
    interest_rate_debt: float # rate paid on debt balance
    min_cash:         float   # revolver plug floor


@dataclass
class ForecastYear:
    year: str
    # Income statement
    revenue:      float = 0
    cogs:         float = 0
    gross_profit: float = 0
    gross_margin: float = 0
    rd:           float = 0
    sga:          float = 0
    ebit:         float = 0
    ebit_margin:  float = 0
    interest_inc: float = 0
    interest_exp: float = 0
    other_income: float = 0
    pretax:       float = 0
    taxes:        float = 0
    net_income:   float = 0
    net_margin:   float = 0
    da:           float = 0
    ebitda:       float = 0
    ebitda_margin:float = 0
    sbc:          float = 0
    adj_ebitda:   float = 0
    adj_ebitda_margin: float = 0
    # Balance sheet
    cash:         float = 0
    ar:           float = 0
    inventory:    float = 0
    other_current:float = 0
    ppe_net:      float = 0
    other_nca:    float = 0
    total_assets: float = 0
    ap:           float = 0
    other_cl:     float = 0
    deferred_rev: float = 0
    other_ncl:    float = 0
    other_lta:    float = 0
    revolver:     float = 0
    ltd:          float = 0
    total_liab:   float = 0
    common_stock: float = 0
    retained_earn:float = 0
    oci:          float = 0
    total_equity: float = 0
    balance_check:float = 0
    # Cash flow
    cfo:          float = 0
    cfi:          float = 0
    cff:          float = 0
    net_cash_chg: float = 0
    # Schedules
    ppe_beg:      float = 0
    ppe_end:      float = 0
    re_beg:       float = 0
    re_end:       float = 0
    nwc:          float = 0
    delta_nwc:    float = 0
    revolver_draw:float = 0


def run_3_statement_model(
    last_hist: HistoricalYear,
    assumptions: List[ForecastAssumptions],
) -> List[ForecastYear]:
    """
    

    Key formulas:
      Revenue_t        = Revenue_{t-1} × (1 + growth_t)
      COGS_t           = Revenue_t × (1 - gross_margin_t)
      Gross Profit_t   = Revenue_t × gross_margin_t
      R&D_t            = Revenue_t × rd_pct_t
      SGA_t            = Revenue_t × sga_pct_t
      EBIT_t           = Gross Profit - R&D - SG&A
      Interest income  = Cash_{t-1} × interest_rate_cash_t   [avg balance approximation]
      Interest expense = (LTD_{t-1} + Revolver_{t-1}) × interest_rate_debt_t
      Pretax           = EBIT + interest_inc - interest_exp + other_income
      Taxes            = Pretax × tax_rate_t  (if pretax > 0, else 0)
      Net income       = Pretax - Taxes
      EBITDA           = EBIT + DA
      Adj EBITDA       = EBITDA + SBC

      PP&E_end         = PP&E_beg + Capex - DA_t
      RE_end           = RE_beg + NI - Dividends - Repurchases
      AR               = Revenue × ar_days / 365
      Inventory        = COGS × inv_days / 365
      AP               = COGS × ap_days / 365
      NWC              = AR + Inventory - AP

      CFO  = NI + DA + SBC - ΔNWC - Δother_current + Δother_cl
             + Δdeferred_rev - Δother_nca
      CFI  = -Capex
      CFF  = LTD_change - Dividends - Repurchases + Revolver_draw
      ΔCash = CFO + CFI + CFF

      Revolver (plug): if cash_end < min_cash → draw revolver to fill gap
                        if cash_end > 0 → pay down revolver first

      Balance check: Total assets - Total liabilities - Total equity.
      The forecast adds no gap of its own, so each year's check equals the
      opening (LTM) gap -- zero when the historical balance sheet balances.
    """
    results = []
    prev = last_hist

    # Track prior-year NWC for delta calculation
    prev_ar  = prev.ar
    prev_inv = prev.inventory
    prev_ap  = prev.ap
    prev_nwc = prev_ar + prev_inv - prev_ap

    prev_other_cur = prev.other_current
    prev_other_cl  = prev.other_cl
    prev_def_rev   = prev.deferred_rev
    prev_other_nca = prev.other_nca
    # Other current assets have no forecast driver, so hold the company's own
    # LTM ratio to revenue (previously a hard-coded 12%, which ignored the
    # historical input entirely).
    other_cur_pct  = (prev.other_current / prev.revenue) if prev.revenue else 0.0
    prev_cash      = prev.cash
    prev_revolver  = 0.0   # assume no revolver at start

    for i, a in enumerate(assumptions):
        yr = ForecastYear(year=f"F+{i+1}")

        # ── Income statement ──────────────────────────────────────────────
        yr.revenue      = prev.revenue * (1 + a.revenue_growth)
        yr.cogs         = -yr.revenue * (1 - a.gross_margin)
        yr.gross_profit = yr.revenue + yr.cogs           # revenue - |COGS|
        yr.gross_margin = yr.gross_profit / yr.revenue
        yr.rd           = -yr.revenue * a.rd_pct
        yr.sga          = -yr.revenue * a.sga_pct
        yr.ebit         = yr.gross_profit + yr.rd + yr.sga
        yr.ebit_margin  = yr.ebit / yr.revenue

        # Interest — based on PRIOR period balances (WSP convention)
        yr.interest_inc = prev_cash * a.interest_rate_cash
        yr.interest_exp = -(prev.ltd + prev_revolver) * a.interest_rate_debt
        yr.other_income = a.other_income
        yr.pretax       = yr.ebit + yr.interest_inc + yr.interest_exp + yr.other_income
        yr.taxes        = -(max(yr.pretax, 0) * a.tax_rate)
        yr.net_income   = yr.pretax + yr.taxes
        yr.net_margin   = yr.net_income / yr.revenue

        # EBITDA reconciliation
        yr.da           = yr.revenue * a.da_pct
        yr.ebitda       = yr.ebit + yr.da
        yr.ebitda_margin= yr.ebitda / yr.revenue
        yr.sbc          = yr.revenue * a.sbc_pct
        yr.adj_ebitda   = yr.ebitda + yr.sbc
        yr.adj_ebitda_margin = yr.adj_ebitda / yr.revenue

        # ── PP&E schedule ─────────────────────────────────────────────────
        yr.ppe_beg   = prev.ppe_net
        capex_abs    = yr.revenue * a.capex_pct
        yr.ppe_end   = yr.ppe_beg + capex_abs - yr.da
        yr.ppe_net   = yr.ppe_end

        # ── Working capital (AR/Inv/AP days method) ───────────────────────
        cogs_abs     = abs(yr.cogs)
        yr.ar        = yr.revenue * a.ar_days  / 365
        yr.inventory = cogs_abs   * a.inv_days / 365
        yr.ap        = -(cogs_abs * a.ap_days  / 365)   # liability
        yr.nwc       = yr.ar + yr.inventory + yr.ap     # net (AP is negative)
        yr.delta_nwc = yr.nwc - prev_nwc                # + = cash use

        # ── Other balance sheet items ─────────────────────────────────────
        yr.other_current = yr.revenue * other_cur_pct
        yr.other_nca     = yr.revenue * a.other_nca_pct
        yr.other_cl      = yr.revenue * a.other_cl_pct
        yr.deferred_rev  = yr.revenue * a.deferred_rev_pct

        # ── Retained earnings roll ────────────────────────────────────────
        yr.re_beg      = prev.retained_earnings
        yr.retained_earn = yr.re_beg + yr.net_income - a.dividends - a.repurchases
        yr.re_end      = yr.retained_earn

        # ── LTD and common stock ──────────────────────────────────────────
        yr.ltd         = prev.ltd + a.ltd_change
        yr.common_stock= prev.common_stock + yr.sbc  # SBC vesting adds to APIC (WSP)
        yr.oci         = prev.oci                   # assume static
        yr.other_ncl   = prev.other_ncl             # held flat, like OCI
        yr.other_lta   = prev.other_lta             # held flat

        # ── Cash flow statement ───────────────────────────────────────────
        delta_other_cur = yr.other_current - prev_other_cur
        delta_other_cl  = yr.other_cl  - prev_other_cl
        delta_def_rev   = yr.deferred_rev - prev_def_rev
        delta_other_nca = yr.other_nca - prev_other_nca

        yr.cfo = (yr.net_income
                  + yr.da
                  + yr.sbc
                  - yr.delta_nwc
                  - delta_other_cur
                  + delta_other_cl
                  + delta_def_rev
                  - delta_other_nca)

        yr.cfi = -capex_abs

        yr.cff = (a.ltd_change
                  - a.dividends
                  - a.repurchases)    # revolver added below after plug

        # ── Revolver plug (minimum cash) ──────────────────────────────────
        # Pre-revolver ending cash
        cash_pre  = prev_cash + yr.cfo + yr.cfi + yr.cff
        shortage  = a.min_cash - cash_pre          # > 0 means need to draw

        if shortage > 0:
            yr.revolver_draw = shortage            # draw revolver
        else:
            # Can we pay down existing revolver?
            yr.revolver_draw = max(-prev_revolver, cash_pre - a.min_cash) * 0
            # Pay down revolver if excess cash
            excess = cash_pre - a.min_cash
            paydown = min(excess, prev_revolver)
            yr.revolver_draw = -paydown

        yr.revolver = prev_revolver + yr.revolver_draw
        yr.cff     += yr.revolver_draw
        yr.net_cash_chg = yr.cfo + yr.cfi + yr.cff
        yr.cash     = prev_cash + yr.net_cash_chg

        # ── Balance sheet ─────────────────────────────────────────────────
        yr.total_assets = (yr.cash + yr.ar + yr.inventory + yr.other_current
                           + yr.ppe_net + yr.other_nca + yr.other_lta)

        yr.ap_abs    = abs(yr.ap)   # store as positive for display
        yr.total_liab= (yr.ap_abs + yr.other_cl + yr.deferred_rev
                        + yr.revolver + yr.ltd + yr.other_ncl)

        yr.total_equity = yr.common_stock + yr.retained_earn + yr.oci
        yr.balance_check= yr.total_assets - yr.total_liab - yr.total_equity
        if abs(yr.balance_check) < 1e-9:   # float noise, not a gap; avoids "$-0.0"
            yr.balance_check = 0.0

        results.append(yr)

        # ── Update prior-period references ────────────────────────────────
        prev           = yr
        prev.revenue   = yr.revenue
        prev.ltd       = yr.ltd
        prev.ppe_net   = yr.ppe_net
        prev.retained_earnings = yr.retained_earn
        prev.common_stock = yr.common_stock
        prev.oci       = yr.oci
        prev_nwc       = yr.nwc
        prev_other_cur = yr.other_current
        prev_other_cl  = yr.other_cl
        prev_def_rev   = yr.deferred_rev
        prev_other_nca = yr.other_nca
        prev_cash      = yr.cash
        prev_revolver  = yr.revolver

    return results


def opening_bs_gap(ltm):
    """Assets - liabilities - equity on the LTM (opening) balance sheet."""
    assets = (ltm.cash + ltm.ar + ltm.inventory + ltm.other_current
              + ltm.ppe_net + ltm.other_nca + ltm.other_lta)
    liab   = ltm.ap + ltm.other_cl + ltm.deferred_rev + ltm.ltd + ltm.other_ncl
    equity = ltm.common_stock + ltm.retained_earnings + ltm.oci
    gap = assets - liab - equity
    return 0.0 if abs(gap) < 1e-9 else gap   # float noise, not a gap


def run_forecast_simulation(ltm, assumptions, n=30000):
    """
    Run Monte Carlo simulation varying growth and margin assumptions.
    Returns dict with revenue paths and ebitda paths.
    """
    rng = np.random.default_rng(42)
    n_fwd = len(assumptions)

    # Derive distribution parameters from assumption spread
    g_means  = np.array([a.revenue_growth for a in assumptions])
    em_means = np.array([a.gross_margin - a.rd_pct - a.sga_pct + a.da_pct for a in assumptions])

    g_std  = max(np.std(g_means), 0.02)
    em_std = max(np.std(em_means), 0.01)

    g_mean  = np.mean(g_means)
    em_mean = np.mean(em_means)

    # Correlated draws (growth and margin positively correlated)
    corr = 0.40
    L = np.array([[1, 0], [corr, np.sqrt(1-corr**2)]])
    Z = rng.standard_normal((2, n))
    C = L @ Z
    growth_draws = g_mean  + C[0] * g_std
    margin_draws = em_mean + C[1] * em_std
    margin_draws = np.clip(margin_draws, 0.01, 0.80)

    # Simulate revenue and EBITDA paths
    rev_paths    = np.zeros((n, n_fwd))
    ebitda_paths = np.zeros((n, n_fwd))

    rev = np.full(n, ltm.revenue)
    for t, a in enumerate(assumptions):
        rev = rev * (1 + growth_draws)
        ebitda = rev * margin_draws
        rev_paths[:, t]    = rev
        ebitda_paths[:, t] = ebitda

    return {"revenue": rev_paths, "ebitda": ebitda_paths,
            "g_draws": growth_draws, "m_draws": margin_draws}


# ---------------------------------------------------------------------------
# Inputs: historical defaults, the LTM year, seeded assumptions
# (moved from _hist_input_block and _assumption_inputs in pages/forecasting.py)
# ---------------------------------------------------------------------------

# (key, latest-year default, input step). Costs are negative. Balance sheet
# items describe the latest year only.
HISTORICAL_FIELDS = [
    # ── Income statement ──
    ("h_rev",        265.0,  10.0),
    ("h_cogs",      -163.0,   5.0),
    ("h_rd",         -14.0,   1.0),
    ("h_sga",        -17.0,   1.0),
    ("h_int_inc",      5.7,   0.5),
    ("h_int_exp",     -3.2,   0.5),
    ("h_other",       -0.4,   0.1),
    ("h_tax",        -13.4,   1.0),
    ("h_da",          10.9,   0.5),
    ("h_sbc",          5.3,   0.5),
    # ── Balance sheet (latest year) ──
    ("h_cash",       237.0,  10.0),
    ("h_ar",          23.2,   1.0),
    ("h_inv",          4.0,   0.5),
    ("h_ocurr",       37.9,   2.0),
    ("h_ppe",         41.3,   2.0),
    ("h_nca",         22.3,   2.0),
    ("h_lta",          0.0,   5.0),
    ("h_ap",          55.9,   2.0),
    ("h_ocl",         32.7,   2.0),
    ("h_def",         10.3,   1.0),
    ("h_ltd",        102.5,   5.0),
    ("h_ncl",          0.0,   5.0),
    ("h_cs",          40.2,   2.0),
    ("h_re",         127.6,   5.0),
    ("h_oci",         -3.5,   0.5),
    # ── Additional data ──
    ("h_capex",       13.3,   1.0),
    ("h_divs",        13.7,   1.0),
    ("h_buybacks",    73.1,   5.0),
]


def default_history_value(default_latest, j, n_hist):
    """Default for year j of n_hist: earlier years are somewhat smaller."""
    scale = 0.85 ** (n_hist - 1 - j)
    return round(default_latest * scale, 1)


def default_history(n_hist=3):
    return {key: [default_history_value(d, j, n_hist) for j in range(n_hist)]
            for key, d, _ in HISTORICAL_FIELDS}


def ltm_from_history(data):
    """HistoricalYear from the LAST year (most recent / LTM) of the history."""
    def last(k):
        return data[k][-1] if k in data else 0.0

    return HistoricalYear(
        year="LTM",
        revenue=last("h_rev"),
        cogs=last("h_cogs"),
        rd=last("h_rd"),
        sga=last("h_sga"),
        other_income=last("h_other"),
        interest_exp=last("h_int_exp"),
        interest_inc=last("h_int_inc"),
        da=last("h_da"),
        sbc=last("h_sbc"),
        tax=last("h_tax"),
        capex=last("h_capex"),
        dividends=last("h_divs"),
        repurchases=last("h_buybacks"),
        cash=last("h_cash"),
        ar=last("h_ar"),
        inventory=last("h_inv"),
        other_current=last("h_ocurr"),
        ppe_net=last("h_ppe"),
        other_nca=last("h_nca"),
        ap=last("h_ap"),
        other_cl=last("h_ocl"),
        deferred_rev=last("h_def"),
        ltd=last("h_ltd"),
        common_stock=last("h_cs"),
        retained_earnings=last("h_re"),
        oci=last("h_oci"),
        other_ncl=last("h_ncl"),
        other_lta=last("h_lta"),
    )


def historical_metrics(data, n_hist):
    """Per-year ratios from the historical inputs. None where undefined."""
    out = []
    rev_vals = data.get("h_rev", [1.0]*n_hist)
    for j in range(n_hist):
        rev = rev_vals[j]
        cogs = data.get("h_cogs", [0]*n_hist)[j]
        gp = rev + cogs
        ebit_j = gp + data.get("h_rd", [0]*n_hist)[j] + data.get("h_sga", [0]*n_hist)[j]
        da_j = data.get("h_da", [0]*n_hist)[j]
        ebitda_j = ebit_j + da_j
        sbc_j = data.get("h_sbc", [0]*n_hist)[j]
        out.append({
            "revenue": rev,
            "revenue_growth": (rev/rev_vals[j-1]-1) if j > 0 and rev_vals[j-1] > 0 else None,
            "gross_margin": gp/rev if rev > 0 else None,
            "rd_pct": abs(data.get('h_rd', [0]*n_hist)[j])/rev if rev > 0 else None,
            "sga_pct": abs(data.get('h_sga', [0]*n_hist)[j])/rev if rev > 0 else None,
            "ebitda_margin": ebitda_j/rev if rev > 0 else None,
            "adj_ebitda_margin": (ebitda_j+sbc_j)/rev if rev > 0 else None,
        })
    return out


# Assumption grid keys, in display order
ASSUMPTION_KEYS = ["rev_g", "gm", "rd", "sga", "tax", "da", "sbc", "capex",
                   "ar_d", "inv_d", "ap_d", "ocl_pct", "def_pct", "nca_pct",
                   "other_inc", "divs", "buybacks", "ltd_chg", "r_cash", "r_debt",
                   "min_cash"]


def seed_assumptions(ltm):
    """Default value for every assumption key, derived from the LTM year."""
    rev     = ltm.revenue if ltm.revenue > 0 else 1
    cogs    = abs(ltm.cogs)
    gp_m    = (rev + ltm.cogs) / rev
    rd_pct  = abs(ltm.rd)  / rev
    sga_pct = abs(ltm.sga) / rev
    da_pct  = ltm.da / rev
    sbc_pct = ltm.sbc / rev
    capex_pct = ltm.capex / rev
    tax_rate  = abs(ltm.tax) / max(
        (rev + ltm.cogs + ltm.rd + ltm.sga +
         ltm.interest_inc + ltm.interest_exp + ltm.other_income), 0.01
    )
    ar_days  = ltm.ar  / rev * 365 if ltm.ar > 0 else 45
    inv_days = ltm.inventory / max(cogs, 1) * 365 if ltm.inventory > 0 else 30
    ap_days  = ltm.ap  / max(cogs, 1) * 365 if ltm.ap > 0 else 60
    other_cl_pct = ltm.other_cl / rev if rev > 0 else 0.12
    def_rev_pct  = ltm.deferred_rev / rev if rev > 0 else 0.04
    other_nca_pct= ltm.other_nca / rev if rev > 0 else 0.08

    return {
        "rev_g":     6.0,
        "gm":        round(gp_m*100,1),
        "rd":        round(rd_pct*100,1),
        "sga":       round(sga_pct*100,1),
        "tax":       round(min(max(tax_rate*100,5),40),1),
        "da":        round(da_pct*100,1),
        "sbc":       round(sbc_pct*100,1),
        "capex":     round(capex_pct*100,1),
        "ar_d":      round(ar_days,0),
        "inv_d":     round(inv_days,0),
        "ap_d":      round(ap_days,0),
        "ocl_pct":   round(other_cl_pct*100,1),
        "def_pct":   round(def_rev_pct*100,1),
        "nca_pct":   round(other_nca_pct*100,1),
        "other_inc": round(ltm.other_income,1),
        "divs":      round(ltm.dividends,1),
        "buybacks":  round(ltm.repurchases,1),
        "ltd_chg":   0.0,
        "r_cash":    2.2,
        "r_debt":    2.8,
        "min_cash":  round(ltm.cash * 0.20, 0),
    }


def assumptions_from_grid(collected, n_fwd):
    """ForecastAssumptions per year from the grid: key -> one value per year."""
    assumptions = []
    for j in range(n_fwd):
        def g(k): return collected[k][j]
        assumptions.append(ForecastAssumptions(
            revenue_growth   = g("rev_g") / 100,
            gross_margin     = g("gm")    / 100,
            rd_pct           = g("rd")    / 100,
            sga_pct          = g("sga")   / 100,
            tax_rate         = g("tax")   / 100,
            da_pct           = g("da")    / 100,
            sbc_pct          = g("sbc")   / 100,
            capex_pct        = g("capex") / 100,
            ar_days          = g("ar_d"),
            inv_days         = g("inv_d"),
            ap_days          = g("ap_d"),
            other_cl_pct     = g("ocl_pct") / 100,
            deferred_rev_pct = g("def_pct") / 100,
            other_nca_pct    = g("nca_pct") / 100,
            other_income     = g("other_inc"),
            dividends        = g("divs"),
            repurchases      = g("buybacks"),
            ltd_change       = g("ltd_chg"),
            interest_rate_cash = g("r_cash") / 100,
            interest_rate_debt = g("r_debt") / 100,
            min_cash         = g("min_cash"),
        ))
    return assumptions


def revenue_cagr(ltm, fwd):
    return (fwd[-1].revenue / ltm.revenue) ** (1/len(fwd)) - 1


def simulation_summary(fwd, sim_paths):
    """Statistics shown on the forecasting page's simulation overlay.

    Moved from render_forecasting and _plot_simulation_charts. A target above
    the deterministic EBITDA is an upside case ("Bull"), one below it a
    downside case ("Bear"). The Streamlit page had these swapped (finding 4).
    """
    rev_paths = sim_paths["revenue"]     # shape (n_scenarios, n_fwd)
    ebitda_paths = sim_paths["ebitda"]
    rev_final = rev_paths[:, -1]
    ebd_final = ebitda_paths[:, -1]

    def bands(paths):
        return {q: np.percentile(paths, int(q[1:]), axis=0)
                for q in ("p5", "p25", "p50", "p75", "p95")}

    def final_stats(arr, deterministic):
        return {"mean": np.mean(arr), "median": np.median(arr),
                "p5": np.percentile(arr, 5), "p25": np.percentile(arr, 25),
                "p75": np.percentile(arr, 75), "p95": np.percentile(arr, 95),
                "deterministic": deterministic}

    det_ebitda_final = fwd[-1].ebitda
    targets = [det_ebitda_final * 0.80,
               det_ebitda_final * 0.90,
               det_ebitda_final,
               det_ebitda_final * 1.10,
               det_ebitda_final * 1.20]
    target_probabilities = [{
        "target": t,
        "probability": (ebd_final >= t).mean(),
        "scenario": ("Bull" if t > det_ebitda_final else
                     "Bear" if t < det_ebitda_final else "Base"),
    } for t in targets]

    growth_final = (rev_paths[:, -1] / rev_paths[:, 0]) ** (1/len(fwd)) - 1

    return {
        "revenue_bands": bands(rev_paths),
        "ebitda_bands": bands(ebitda_paths),
        "revenue_final": final_stats(rev_final, fwd[-1].revenue),
        "ebitda_final": final_stats(ebd_final, fwd[-1].ebitda),
        "ebitda_final_median": np.percentile(ebitda_paths[:, -1], 50),
        "target_probabilities": target_probabilities,
        "growth_final_mean": np.mean(growth_final),
    }
