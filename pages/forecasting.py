"""
pages/forecasting.py
--------------------
Full 3-statement financial model with Monte Carlo simulation overlay.

Structure mirrors the Wall Street Prep Financial Statement Model:
  INCOME STATEMENT
    Revenue (growth % driver)
    Cost of sales (gross margin %)
    Gross Profit
    R&D (% of sales)
    SG&A (% of sales)
    EBIT
    Interest income  (cash balance × rate)
    Interest expense (debt balance × rate)
    Other income/expense (flat)
    Pretax profit
    Taxes (tax rate %)
    Net income
    D&A add-back → EBITDA
    SBC add-back → Adjusted EBITDA

  SUPPORTING SCHEDULES
    PP&E roll-forward      (beg + capex − depreciation = end)
    Other non-current      (beg − amort + additions = end)
    Retained earnings roll (beg + NI − dividends − buybacks = end)
    Working capital        (AR days, inventory days, AP days)
    Interest schedule      (on cash and debt balances)
    Revolver plug          (minimum cash check)

  BALANCE SHEET
    Cash (derived from CF statement — revolver as plug)
    AR, Inventory, Other current
    PP&E (from schedule)
    Other non-current (from schedule)
    AP, Other current liabilities
    Deferred revenue
    Long-term debt
    Common stock, Retained earnings, OCI
    Balance check = 0

  CASH FLOW STATEMENT
    Operating: NI + D&A + SBC + ΔWC + other
    Investing:  Capex
    Financing:  Debt changes + dividends + buybacks
    Net change in cash

  SIMULATION OVERLAY
    40,000 scenarios varying growth, margin, interest, exit multiple
    Confidence bands on revenue and EBITDA
    IRR distribution if acquired at current EV
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
from dataclasses import dataclass, field
from typing import List, Optional
from ml.edgar_extractor import fetch_financials, financials_to_session_state
try:
    from simulation.vectorized_simulation import (
        run_vectorized_simulation_full, SimulationParams
    )
    _SIM_AVAILABLE = True
except ImportError:
    _SIM_AVAILABLE = False

# ── Colours ──────────────────────────────────────────────────────────────────
BG  = "#05050c"; BG2 = "#0e0e1c"
C1  = "#6060c0"; C2  = "#40a0c0"; C3  = "#c06060"
C4  = "#40c080"; C5  = "#c0a040"; C6  = "#a060c0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2,
    "axes.edgecolor": "#2a2a42", "axes.labelcolor": "#8888a4",
    "text.color": "#c4c4d4", "xtick.color": "#5a5a72", "ytick.color": "#5a5a72",
    "grid.color": "#16162a", "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": "#c4c4d4",
    "legend.facecolor": BG2, "legend.edgecolor": "#2a2a42", "legend.fontsize": 8,
})

# ── UI helpers ────────────────────────────────────────────────────────────────

def _sz(n):
    return max(8, int(n * st.session_state.get("font_scale", 1.0)))

def _lbl(text):
    st.markdown(
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:{_sz(11)}px;'
        f'color:#5a5a72;display:block;margin-bottom:2px">{text}</span>',
        unsafe_allow_html=True,
    )

def _section(title, color="#85b7eb"):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{color};border-bottom:1px solid #16162a;'
        f'padding-bottom:5px;margin:18px 0 10px">◈ {title}</div>',
        unsafe_allow_html=True,
    )

def _note(text):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'color:#3a3a5a;font-style:italic;margin-bottom:6px">{text}</div>',
        unsafe_allow_html=True,
    )

def _blk(title, border, title_color):
    st.markdown(
        f'<div style="background:#080816;border-left:2px solid {border};'
        f'border-radius:6px;padding:8px 12px 4px;margin-bottom:4px">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{title_color};margin-bottom:8px">{title}</div></div>',
        unsafe_allow_html=True,
    )

def _chip(value, green=False):
    bg  = "#061a10" if green else "#0c2040"
    bdr = "#1d9e75" if green else "#185fa5"
    clr = "#5dcaa5" if green else "#85b7eb"
    st.markdown(
        f'<div style="background:{bg};border:0.5px solid {bdr};border-radius:4px;'
        f'padding:5px 10px;font-family:IBM Plex Mono,monospace;'
        f'font-size:{_sz(12)}px;color:{clr};margin-bottom:4px">{value}</div>',
        unsafe_allow_html=True,
    )

def _ni(label, key, value, **kw):
    _lbl(label)
    for k in ("min_value", "max_value", "step"):
        if k in kw:
            kw[k] = float(kw[k])
    return st.number_input(" ", value=float(value), key=key,
                           label_visibility="collapsed", **kw)

def _to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31])
    return buf.getvalue()

def _dl(label, data, fname, key):
    st.download_button(f"⬇ {label}", data=data, file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key, use_container_width=True)

# ── 3-Statement Model Engine ──────────────────────────────────────────────────

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

      CFO  = NI + DA + SBC - ΔNWC + Δother_cl + Δdeferred_rev + Δother_nca
      CFI  = -Capex
      CFF  = LTD_change - Dividends - Repurchases + Revolver_draw
      ΔCash = CFO + CFI + CFF

      Revolver (plug): if cash_end < min_cash → draw revolver to fill gap
                        if cash_end > 0 → pay down revolver first

      Balance check: Total assets - Total liabilities - Total equity = 0
    """
    results = []
    prev = last_hist

    # Track prior-year NWC for delta calculation
    prev_ar  = prev.ar
    prev_inv = prev.inventory
    prev_ap  = prev.ap
    prev_nwc = prev_ar + prev_inv - prev_ap

    prev_other_cl  = prev.other_cl
    prev_def_rev   = prev.deferred_rev
    prev_other_nca = prev.other_nca
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
        yr.other_current = yr.revenue * 0.12    # flat ratio — user can extend
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

        # ── Cash flow statement ───────────────────────────────────────────
        delta_other_cl  = yr.other_cl  - prev_other_cl
        delta_def_rev   = yr.deferred_rev - prev_def_rev
        delta_other_nca = yr.other_nca - prev_other_nca

        yr.cfo = (yr.net_income
                  + yr.da
                  + yr.sbc
                  - yr.delta_nwc
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
                           + yr.ppe_net + yr.other_nca)

        yr.ap_abs    = abs(yr.ap)   # store as positive for display
        yr.total_liab= (yr.ap_abs + yr.other_cl + yr.deferred_rev
                        + yr.revolver + yr.ltd)

        yr.total_equity = yr.common_stock + yr.retained_earn + yr.oci
        yr.balance_check= yr.total_assets - yr.total_liab - yr.total_equity

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
        prev_other_cl  = yr.other_cl
        prev_def_rev   = yr.deferred_rev
        prev_other_nca = yr.other_nca
        prev_cash      = yr.cash
        prev_revolver  = yr.revolver

    return results


# ── Historical input table ────────────────────────────────────────────────────

def _hist_input_block(n_hist, unit):
    """
    Render the historical data input table 
    Returns a HistoricalYear representing the most recent (LTM) year.
    """
    cols = st.columns([2] + [1]*n_hist, gap="small")
    yr_labels = [f"Year −{n_hist-i}" for i in range(n_hist)]

    with cols[0]:
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'color:#44445a;padding-top:24px">Line item ({unit})</div>',
            unsafe_allow_html=True,
        )

    for j, yr in enumerate(yr_labels):
        with cols[j+1]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:#85b7eb;text-align:center;margin-bottom:4px">{yr}</div>',
                unsafe_allow_html=True,
            )

    # Row definitions: (key, label, default_latest, step)
    rows = [
        # ── Income statement ──────────────────────────────────
        ("__hdr_is__",  "── INCOME STATEMENT ──", None, None),
        ("h_rev",       f"Revenue ({unit})",          265.0,   10.0),
        ("h_cogs",      f"Cost of sales ({unit}, negative)", -163.0, 5.0),
        ("h_rd",        f"R&D ({unit}, 0 if N/A, negative)",  -14.0,  1.0),
        ("h_sga",       f"SG&A ({unit}, negative)",           -17.0,  1.0),
        ("h_int_inc",   f"Interest income ({unit})",           5.7,   0.5),
        ("h_int_exp",   f"Interest expense ({unit}, negative)",-3.2,  0.5),
        ("h_other",     f"Other income/expense ({unit})",      -0.4,  0.1),
        ("h_tax",       f"Taxes ({unit}, negative)",          -13.4,  1.0),
        ("h_da",        f"D&A ({unit}, positive add-back)",    10.9,  0.5),
        ("h_sbc",       f"SBC ({unit}, positive add-back)",     5.3,  0.5),
        # ── Balance sheet ─────────────────────────────────────
        ("__hdr_bs__",  "── BALANCE SHEET (latest year) ──", None, None),
        ("h_cash",      f"Cash & equivalents ({unit})",       237.0, 10.0),
        ("h_ar",        f"Accounts receivable ({unit})",       23.2,  1.0),
        ("h_inv",       f"Inventories ({unit})",                4.0,  0.5),
        ("h_ocurr",     f"Other current assets ({unit})",      37.9,  2.0),
        ("h_ppe",       f"PP&E net ({unit})",                  41.3,  2.0),
        ("h_nca",       f"Other non-current assets ({unit})",  22.3,  2.0),
        ("h_ap",        f"Accounts payable ({unit})",          55.9,  2.0),
        ("h_ocl",       f"Other current liabilities ({unit})", 32.7,  2.0),
        ("h_def",       f"Deferred revenue ({unit})",          10.3,  1.0),
        ("h_ltd",       f"Long-term debt ({unit})",           102.5,  5.0),
        ("h_cs",        f"Common stock ({unit})",              40.2,  2.0),
        ("h_re",        f"Retained earnings ({unit})",         70.4,  5.0),
        ("h_oci",       f"Other comprehensive income ({unit})", -3.5, 0.5),
        # ── Additional data ───────────────────────────────────
        ("__hdr_ad__",  "── ADDITIONAL DATA ──", None, None),
        ("h_capex",     f"Capital expenditures ({unit})",      13.3,  1.0),
        ("h_divs",      f"Dividends ({unit})",                 13.7,  1.0),
        ("h_buybacks",  f"Buybacks / repurchases ({unit})",    73.1,  5.0),
    ]

    data = {}
    for key, label, default_latest, step in rows:
        if key.startswith("__hdr__") or label.startswith("──"):
            row_cols = st.columns([2] + [1]*n_hist, gap="small")
            with row_cols[0]:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:{_sz(9)}px;color:#44445a;margin-top:10px;'
                    f'text-transform:uppercase;letter-spacing:0.10em">{label}</div>',
                    unsafe_allow_html=True,
                )
            continue

        row_cols = st.columns([2] + [1]*n_hist, gap="small")
        with row_cols[0]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:#c4c4d4;padding-top:22px;border-bottom:0.5px solid #16162a;'
                f'padding-bottom:4px">{label}</div>',
                unsafe_allow_html=True,
            )

        year_vals = []
        for j in range(n_hist):
            # Scale default: earlier years are somewhat smaller
            scale = 0.85 ** (n_hist - 1 - j)
            default_j = round(default_latest * scale, 1)
            with row_cols[j+1]:
                v = st.number_input(
                    " ", value=default_j, step=float(step),
                    key=f"hist_{key}_{j}",
                    label_visibility="collapsed",
                )
            year_vals.append(v)
        data[key] = year_vals

    # Build HistoricalYear from the LAST year (most recent / LTM)
    def last(k):
        return data[k][-1] if k in data else 0.0

    # Also compute historical ratios for display
    hist_rows_display = []
    rev_vals = data.get("h_rev", [1.0]*n_hist)
    for j in range(n_hist):
        rev = rev_vals[j]
        cogs = data.get("h_cogs", [0]*n_hist)[j]
        gp = rev + cogs
        ebit_j = gp + data.get("h_rd", [0]*n_hist)[j] + data.get("h_sga", [0]*n_hist)[j]
        da_j = data.get("h_da", [0]*n_hist)[j]
        ebitda_j = ebit_j + da_j
        sbc_j = data.get("h_sbc", [0]*n_hist)[j]
        hist_rows_display.append({
            "Year": yr_labels[j],
            f"Revenue ({unit})": f"${rev:,.1f}",
            "Revenue growth": f"{(rev/rev_vals[j-1]-1):.1%}" if j > 0 and rev_vals[j-1] > 0 else "—",
            "Gross margin": f"{gp/rev:.1%}" if rev > 0 else "—",
            "R&D %": f"{abs(data.get('h_rd',[0]*n_hist)[j])/rev:.1%}" if rev > 0 else "—",
            "SG&A %": f"{abs(data.get('h_sga',[0]*n_hist)[j])/rev:.1%}" if rev > 0 else "—",
            "EBITDA margin": f"{ebitda_j/rev:.1%}" if rev > 0 else "—",
            "Adj EBITDA margin": f"{(ebitda_j+sbc_j)/rev:.1%}" if rev > 0 else "—",
        })

    ltm = HistoricalYear(
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
    )
    return ltm, pd.DataFrame(hist_rows_display).set_index("Year")


# ── Forecast assumptions input ────────────────────────────────────────────────

def _assumption_inputs(n_fwd, ltm, unit):
    """
    Render per-year forecast assumption inputs.
    Returns list of ForecastAssumptions.
    """
    _note(
        "Blue = input you set.  "
        "Enter one value per forecast year or use flat assumptions across all years."
    )

    # Compute historical averages for smart defaults
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

    yr_cols = [""] + [f"F+{i+1}" for i in range(n_fwd)]
    header_cols = st.columns([2] + [1]*n_fwd, gap="small")
    with header_cols[0]:
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'color:#44445a;text-transform:uppercase">Assumption</div>',
            unsafe_allow_html=True,
        )
    for j, hdr in enumerate(yr_cols[1:]):
        with header_cols[j+1]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:#6060c0;text-align:center">{hdr}</div>',
                unsafe_allow_html=True,
            )

    assumption_rows = [
        # (key, label, default, step, color, section)
        ("__is__",    "── INCOME STATEMENT DRIVERS ──", None, None, None, True),
        ("rev_g",     "Revenue growth (%)",        6.0,   0.5,  "#5dcaa5", False),
        ("gm",        "Gross profit margin (%)",  round(gp_m*100,1), 0.5, "#85b7eb", False),
        ("rd",        "R&D % of sales",           round(rd_pct*100,1), 0.25,"#afa9ec", False),
        ("sga",       "SG&A % of sales",          round(sga_pct*100,1),0.25,"#afa9ec", False),
        ("tax",       "Tax rate (%)",             round(min(max(tax_rate*100,5),40),1),0.5,"#ef9f27", False),
        ("__da__",    "── D&A & CAPEX ──",        None, None, None, True),
        ("da",        "D&A % of revenue",         round(da_pct*100,1), 0.25, "#5dcaa5", False),
        ("sbc",       "SBC % of revenue",         round(sbc_pct*100,1),0.25, "#5dcaa5", False),
        ("capex",     "Capex % of revenue",       round(capex_pct*100,1),0.25,"#ef9f27", False),
        ("__wc__",    "── WORKING CAPITAL (days) ──", None, None, None, True),
        ("ar_d",      "AR days (Revenue÷365)",    round(ar_days,0),  1.0, "#85b7eb", False),
        ("inv_d",     "Inventory days (COGS÷365)",round(inv_days,0), 1.0, "#85b7eb", False),
        ("ap_d",      "AP days (COGS÷365)",       round(ap_days,0),  1.0, "#85b7eb", False),
        ("__bs__",    "── OTHER B/S ASSUMPTIONS ──", None, None, None, True),
        ("ocl_pct",   "Other current liab % rev", round(other_cl_pct*100,1),0.25,"#afa9ec", False),
        ("def_pct",   "Deferred rev % of revenue",round(def_rev_pct*100,1), 0.25,"#afa9ec", False),
        ("nca_pct",   "Other NCA % of revenue",   round(other_nca_pct*100,1),0.25,"#afa9ec", False),
        ("__other__", "── FINANCING & OTHER ──", None, None, None, True),
        ("other_inc", f"Other income ({unit}, flat)", round(ltm.other_income,1),0.1,"#c4c4d4", False),
        ("divs",      f"Dividends ({unit}, flat)",    round(ltm.dividends,1),  1.0,"#f0997b", False),
        ("buybacks",  f"Buybacks ({unit}, flat)",     round(ltm.repurchases,1),5.0,"#f0997b", False),
        ("ltd_chg",   f"LTD net change ({unit})",     0.0,  5.0, "#f0997b", False),
        ("r_cash",    "Interest rate on cash (%)",  2.2,  0.1, "#40a0c0", False),
        ("r_debt",    "Interest rate on debt (%)",  2.8,  0.1, "#c06060", False),
        ("min_cash",  f"Minimum cash ({unit})",     round(ltm.cash * 0.20, 0), 5.0,"#44445a", False),
    ]

    collected = {k: [] for k, *_ in assumption_rows if not k.startswith("__")}

    for key, label, default, step, color, is_hdr in assumption_rows:
        if is_hdr:
            row_cols = st.columns([2] + [1]*n_fwd, gap="small")
            with row_cols[0]:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:{_sz(9)}px;color:#3a3a5a;margin-top:10px;'
                    f'text-transform:uppercase;letter-spacing:0.10em;'
                    f'border-top:0.5px solid #1a1a2e;padding-top:8px">{label}</div>',
                    unsafe_allow_html=True,
                )
            continue

        row_cols = st.columns([2] + [1]*n_fwd, gap="small")
        with row_cols[0]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:{color};padding-top:22px;border-bottom:0.5px solid #16162a;'
                f'padding-bottom:3px">{label}</div>',
                unsafe_allow_html=True,
            )

        for j in range(n_fwd):
            with row_cols[j+1]:
                val = st.number_input(
                    " ", value=float(default), step=float(step),
                    key=f"fwd_{key}_{j}",
                    label_visibility="collapsed",
                )
            collected[key].append(val)

    # Build ForecastAssumptions per year
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


# ── Output tables ─────────────────────────────────────────────────────────────

def _make_is_df(ltm, fwd, unit):
    cols = ["LTM"] + [y.year for y in fwd]
    def row(label, vals, fmt="dollar"):
        if fmt == "dollar":
            return [label] + [f"${v:,.1f}" for v in vals]
        elif fmt == "pct":
            return [label] + [f"{v:.1%}" for v in vals]
        else:
            return [label] + [str(v) for v in vals]

    ltm_gp = ltm.revenue + ltm.cogs
    ltm_ebit = ltm_gp + ltm.rd + ltm.sga
    ltm_ebitda = ltm_ebit + ltm.da
    ltm_adj = ltm_ebitda + ltm.sbc
    ltm_pretax = ltm_ebit + ltm.interest_inc + ltm.interest_exp + ltm.other_income
    ltm_ni = ltm_pretax + ltm.tax

    rows = [
        row("Revenue",          [ltm.revenue]  + [y.revenue      for y in fwd]),
        row("  Revenue growth", ["—"] + [f"{y.revenue/fwd[i-1].revenue - 1:.1%}" if i > 0 else "—"
                                          for i, y in enumerate(fwd)], fmt="str"),
        row("Cost of sales",    [ltm.cogs]     + [y.cogs         for y in fwd]),
        row("Gross profit",     [ltm_gp]       + [y.gross_profit for y in fwd]),
        row("  Gross margin",   [ltm_gp/ltm.revenue if ltm.revenue else 0] +
                                [y.gross_margin for y in fwd], fmt="pct"),
        row("R&D",              [ltm.rd]       + [y.rd           for y in fwd]),
        row("  R&D %",         [abs(ltm.rd)/ltm.revenue if ltm.revenue else 0] +
                                [abs(y.rd)/y.revenue for y in fwd], fmt="pct"),
        row("SG&A",             [ltm.sga]      + [y.sga          for y in fwd]),
        row("  SG&A %",        [abs(ltm.sga)/ltm.revenue if ltm.revenue else 0] +
                                [abs(y.sga)/y.revenue for y in fwd], fmt="pct"),
        row("EBIT",             [ltm_ebit]     + [y.ebit         for y in fwd]),
        row("  EBIT margin",    [ltm_ebit/ltm.revenue if ltm.revenue else 0] +
                                [y.ebit_margin for y in fwd], fmt="pct"),
        row("Interest income",  [ltm.interest_inc] + [y.interest_inc for y in fwd]),
        row("Interest expense", [ltm.interest_exp] + [y.interest_exp for y in fwd]),
        row("Other income/exp", [ltm.other_income] + [y.other_income for y in fwd]),
        row("Pretax profit",    [ltm_pretax]   + [y.pretax       for y in fwd]),
        row("Taxes",            [ltm.tax]      + [y.taxes        for y in fwd]),
        row("Net income",       [ltm_ni]       + [y.net_income   for y in fwd]),
        row("  Net margin",     [ltm_ni/ltm.revenue if ltm.revenue else 0] +
                                [y.net_margin for y in fwd], fmt="pct"),
        row("D&A (add-back)",   [ltm.da]       + [y.da           for y in fwd]),
        row("EBITDA",           [ltm_ebitda]   + [y.ebitda       for y in fwd]),
        row("  EBITDA margin",  [ltm_ebitda/ltm.revenue if ltm.revenue else 0] +
                                [y.ebitda_margin for y in fwd], fmt="pct"),
        row("SBC (add-back)",   [ltm.sbc]      + [y.sbc          for y in fwd]),
        row("Adj EBITDA",       [ltm_adj]      + [y.adj_ebitda   for y in fwd]),
        row("  Adj EBITDA marg",[ltm_adj/ltm.revenue if ltm.revenue else 0] +
                                [y.adj_ebitda_margin for y in fwd], fmt="pct"),
    ]

    df = pd.DataFrame(rows, columns=["Line item"] + cols)
    return df.set_index("Line item")


def _make_bs_df(ltm, fwd):
    cols = ["LTM"] + [y.year for y in fwd]
    def r(label, vals):
        return [label] + [f"${v:,.1f}" for v in vals]

    rows = [
        r("Cash",               [ltm.cash]     + [y.cash         for y in fwd]),
        r("Accounts receivable",[ltm.ar]       + [y.ar           for y in fwd]),
        r("Inventories",        [ltm.inventory]+ [y.inventory    for y in fwd]),
        r("Other current",      [ltm.other_current]+ [y.other_current for y in fwd]),
        r("PP&E net",           [ltm.ppe_net]  + [y.ppe_net      for y in fwd]),
        r("Other non-current",  [ltm.other_nca]+ [y.other_nca   for y in fwd]),
        r("TOTAL ASSETS",       [ltm.cash+ltm.ar+ltm.inventory+ltm.other_current+ltm.ppe_net+ltm.other_nca]
                                + [y.total_assets for y in fwd]),
        r("Accounts payable",   [ltm.ap]       + [y.ap_abs       for y in fwd]),
        r("Other curr liab",    [ltm.other_cl] + [y.other_cl     for y in fwd]),
        r("Deferred revenue",   [ltm.deferred_rev]+ [y.deferred_rev for y in fwd]),
        r("Revolver",           [0.0]          + [y.revolver     for y in fwd]),
        r("Long-term debt",     [ltm.ltd]      + [y.ltd          for y in fwd]),
        r("TOTAL LIABILITIES",  [ltm.ap+ltm.other_cl+ltm.deferred_rev+ltm.ltd]
                                + [y.total_liab for y in fwd]),
        r("Common stock",       [ltm.common_stock]+ [y.common_stock for y in fwd]),
        r("Retained earnings",  [ltm.retained_earnings]+ [y.retained_earn for y in fwd]),
        r("OCI",                [ltm.oci]      + [y.oci          for y in fwd]),
        r("TOTAL EQUITY",       [ltm.common_stock+ltm.retained_earnings+ltm.oci]
                                + [y.total_equity for y in fwd]),
        r("Balance check",      [0.0]          + [y.balance_check for y in fwd]),
    ]
    df = pd.DataFrame(rows, columns=["Line item"] + cols)
    return df.set_index("Line item")


def _make_cf_df(fwd):
    cols = [y.year for y in fwd]
    def r(label, vals):
        return [label] + [f"${v:,.1f}" for v in vals]

    rows = [
        r("Net income",         [y.net_income  for y in fwd]),
        r("D&A",                [y.da          for y in fwd]),
        r("SBC",                [y.sbc         for y in fwd]),
        r("Change in NWC",      [-y.delta_nwc  for y in fwd]),
        r("Cash from ops (CFO)",[y.cfo         for y in fwd]),
        r("Capital expenditures",[-(y.revenue * 0) for y in fwd]),  # shown in CFI
        r("Cash from investing (CFI)",[y.cfi   for y in fwd]),
        r("Debt changes",       [y.revolver_draw for y in fwd]),
        r("Dividends",          [-fwd[i].other_income*0 for i, y in enumerate(fwd)]),
        r("Cash from financing (CFF)",[y.cff   for y in fwd]),
        r("Net change in cash", [y.net_cash_chg for y in fwd]),
        r("Ending cash balance",[y.cash        for y in fwd]),
    ]
    df = pd.DataFrame(rows, columns=["Line item"] + cols)
    return df.set_index("Line item")


def _make_ppe_df(ltm, fwd):
    cols = ["LTM"] + [y.year for y in fwd]
    rows = [
        ["Beginning PP&E"] + [f"${ltm.ppe_net:,.1f}"] + [f"${y.ppe_beg:,.1f}" for y in fwd],
        ["+ Capex"]        + ["—"] + [f"${y.revenue*0:.1f}" for y in fwd],
        ["− Depreciation"] + ["—"] + [f"(${y.da:,.1f})" for y in fwd],
        ["Ending PP&E"]    + [f"${ltm.ppe_net:,.1f}"] + [f"${y.ppe_end:,.1f}" for y in fwd],
    ]
    df = pd.DataFrame(rows, columns=["Item"]+cols).set_index("Item")
    return df


def _make_re_df(ltm, fwd):
    cols = ["LTM"] + [y.year for y in fwd]
    rows = [
        ["Beginning RE"]  + [f"${ltm.retained_earnings:,.1f}"] + [f"${y.re_beg:,.1f}" for y in fwd],
        ["+ Net income"]  + ["—"] + [f"${y.net_income:,.1f}" for y in fwd],
        ["− Dividends"]   + [f"(${ltm.dividends:,.1f})"] + [f"(${y.other_income*0:.1f})" for y in fwd],
        ["− Repurchases"] + [f"(${ltm.repurchases:,.1f})"] + ["—" for y in fwd],
        ["Ending RE"]     + [f"${ltm.retained_earnings:,.1f}"] + [f"${y.re_end:,.1f}" for y in fwd],
    ]
    df = pd.DataFrame(rows, columns=["Item"]+cols).set_index("Item")
    return df


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_is_charts(ltm, fwd, company, unit):
    """Revenue, EBITDA, margins, net income — 2×2 chart grid."""
    yr_labels = ["LTM"] + [y.year for y in fwd]
    rev   = [ltm.revenue]   + [y.revenue   for y in fwd]
    ebitda= [ltm.revenue + ltm.cogs + ltm.rd + ltm.sga + ltm.da] + [y.ebitda for y in fwd]
    adj_e = [ebitda[0] + ltm.sbc] + [y.adj_ebitda for y in fwd]
    gm    = [(ltm.revenue+ltm.cogs)/ltm.revenue if ltm.revenue else 0] + [y.gross_margin for y in fwd]
    em    = [ebitda[0]/ltm.revenue if ltm.revenue else 0] + [y.ebitda_margin for y in fwd]
    ni    = [ltm.tax + (ltm.revenue+ltm.cogs+ltm.rd+ltm.sga+ltm.interest_inc+ltm.interest_exp+ltm.other_income)]
    ni   += [y.net_income for y in fwd]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fwd_marker = 1   # index where forecast starts

    # Revenue
    axes[0,0].bar(yr_labels[:fwd_marker], rev[:fwd_marker],
                  color=C2, alpha=0.6, width=0.5, label="Historical")
    axes[0,0].bar(yr_labels[fwd_marker:], rev[fwd_marker:],
                  color=C1, alpha=0.8, width=0.5, label="Forecast")
    axes[0,0].plot(yr_labels, rev, color=C1, lw=1.5, marker="o", ms=4)
    axes[0,0].set_title(f"{company} — Revenue ({unit})")
    axes[0,0].grid(axis="y"); axes[0,0].legend()
    axes[0,0].axvline(0.5, color="#2a2a42", lw=1, linestyle=":")

    # EBITDA vs Adj EBITDA
    x = np.arange(len(yr_labels))
    axes[0,1].bar(x-0.2, ebitda, width=0.35, color=C1, alpha=0.7, label="EBITDA")
    axes[0,1].bar(x+0.2, adj_e,  width=0.35, color=C4, alpha=0.7, label="Adj EBITDA")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(yr_labels)
    axes[0,1].set_title(f"{company} — EBITDA vs Adjusted EBITDA ({unit})")
    axes[0,1].grid(axis="y"); axes[0,1].legend()

    # Margin profile
    axes[1,0].plot(yr_labels, [v*100 for v in gm], color=C2, lw=2,
                   marker="o", ms=4, label="Gross margin")
    axes[1,0].plot(yr_labels, [v*100 for v in em], color=C1, lw=2,
                   marker="s", ms=4, label="EBITDA margin")
    axes[1,0].axvline(yr_labels[fwd_marker-1], color="#2a2a42",
                       lw=1, linestyle=":")
    axes[1,0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    axes[1,0].set_title(f"{company} — Margin profile")
    axes[1,0].grid(axis="y"); axes[1,0].legend()

    # Net income
    bar_c = [C4 if v >= 0 else C3 for v in ni]
    axes[1,1].bar(yr_labels, ni, color=bar_c, alpha=0.8, width=0.5)
    axes[1,1].axhline(0, color="#2a2a42", lw=1)
    axes[1,1].set_title(f"{company} — Net income ({unit})")
    axes[1,1].grid(axis="y")

    plt.tight_layout(pad=1.5)
    return fig


def _plot_simulation_charts(fwd, sim_paths, company, unit):
    """Revenue with MC confidence bands + EBITDA distribution."""
    if sim_paths is None:
        return None

    rev_paths = sim_paths["revenue"]     # shape (n_scenarios, n_fwd)
    ebitda_paths = sim_paths["ebitda"]

    yr_labels = [y.year for y in fwd]
    det_rev   = [y.revenue for y in fwd]
    det_ebitda= [y.ebitda  for y in fwd]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Revenue fan chart
    p5  = np.percentile(rev_paths, 5,  axis=0)
    p25 = np.percentile(rev_paths, 25, axis=0)
    p50 = np.percentile(rev_paths, 50, axis=0)
    p75 = np.percentile(rev_paths, 75, axis=0)
    p95 = np.percentile(rev_paths, 95, axis=0)

    axes[0].fill_between(yr_labels, p5,  p95, color=C1, alpha=0.12, label="5–95%")
    axes[0].fill_between(yr_labels, p25, p75, color=C1, alpha=0.22, label="25–75%")
    axes[0].plot(yr_labels, p50, color=C1, lw=2, label="Sim median")
    axes[0].plot(yr_labels, det_rev, color=C2, lw=2, linestyle="--",
                 marker="o", ms=4, label="Deterministic")
    axes[0].set_title(f"{company} — Revenue with simulation bands")
    axes[0].set_ylabel(unit); axes[0].legend(); axes[0].grid(axis="y")

    # EBITDA distribution (final year)
    axes[1].hist(ebitda_paths[:, -1], bins=60, color=C4,
                 alpha=0.75, edgecolor="none", density=True)
    axes[1].axvline(det_ebitda[-1], color=C2, lw=2, linestyle="--",
                    label=f"Det. ${det_ebitda[-1]:,.0f}")
    axes[1].axvline(np.percentile(ebitda_paths[:, -1], 50),
                    color=C1, lw=1.5, linestyle=":", label="Sim P50")
    axes[1].set_title(f"Year +{len(fwd)} EBITDA distribution")
    axes[1].set_xlabel(unit); axes[1].legend(); axes[1].grid(axis="y")

    # Revenue growth distribution
    if rev_paths.shape[1] >= 1:
        base_rev = det_rev[0] / (1 + fwd[0].revenue * 0)  # approx
        growth_final = (rev_paths[:, -1] / rev_paths[:, 0]) ** (1/len(fwd)) - 1
        axes[2].hist(growth_final * 100, bins=60, color=C5,
                     alpha=0.75, edgecolor="none", density=True)
        axes[2].axvline(np.mean(growth_final)*100, color=C2, lw=2,
                        linestyle="--", label=f"Mean {np.mean(growth_final):.1%}")
        axes[2].set_title("Implied CAGR distribution")
        axes[2].set_xlabel("Revenue CAGR (%)"); axes[2].legend()
        axes[2].grid(axis="y")

    plt.tight_layout(pad=1.5)
    return fig


# ── Monte Carlo runner for forecasting ───────────────────────────────────────

def _run_forecast_simulation(ltm, assumptions, n=30000):
    """
    Run Monte Carlo simulation varying growth and margin assumptions.
    Returns dict with revenue paths and ebitda paths.
    """
    if not _SIM_AVAILABLE:
        return None

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


# ── Main render ───────────────────────────────────────────────────────────────

def render_forecasting():
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(14)}px;'
        f'font-weight:500;color:#e0e0f0;letter-spacing:0.08em;'
        f'border-bottom:1px solid #16162a;padding-bottom:12px;margin-bottom:16px">'
        f'COMPANY FORECASTING —  3-Statement Model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
        f'color:#5a5a72;margin-bottom:16px">'
        f'Full Income Statement → Balance Sheet → Cash Flow model with '
        f'supporting schedules (PP&amp;E, retained earnings, working capital, '
        f'interest, revolver plug). Monte Carlo simulation overlaid on the '
        f'deterministic projections.</div>',
        unsafe_allow_html=True,
    )
   

# Inside render_forecasting(), add this block before _section("Company information"):
    st.markdown(
        f'<div style="background:#061a10;border:0.5px solid #1d9e75;border-radius:6px;'
        f'padding:10px 14px;margin-bottom:16px;font-family:IBM Plex Mono,monospace;">'
        f'<div style="font-size:{_sz(11)}px;color:#5dcaa5;font-weight:500;margin-bottom:6px">'
        f'⚡ Auto-fill from SEC EDGAR</div>'
        f'<div style="font-size:{_sz(9)}px;color:#3a6a50">For any US public company — '
        f'downloads last 5 years of audited financials automatically (free, no API key)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    ec1, ec2, ec3 = st.columns([2, 1, 2], gap="small")
    with ec1:
        edgar_ticker = st.text_input(
            " ", placeholder="Ticker (e.g. MCD, KO, MSFT, AAPL)",
            key="edgar_ticker", label_visibility="collapsed"
        )
    with ec2:
        edgar_fetch = st.button("⬇ Fetch from EDGAR", 
                                type="primary", key="edgar_fetch",
                                use_container_width=True)
    with ec3:
        if 'edgar_company_name' in st.session_state:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:#5dcaa5;padding-top:10px">✓ Loaded: {st.session_state.edgar_company_name}</div>',
                unsafe_allow_html=True,
            )

    if edgar_fetch and edgar_ticker:
        with st.spinner(f"Fetching financials for {edgar_ticker.upper()} from SEC EDGAR..."):
            try:
                extracted = fetch_financials(edgar_ticker.upper(), n_years=3)
                session_vals = financials_to_session_state(extracted)
                for key, val in session_vals.items():
                    st.session_state[key] = val
                st.session_state['edgar_company_name'] = extracted.company_name
                st.session_state['fc2_company'] = extracted.company_name
                if extracted.warnings:
                    st.warning("Data loaded with warnings:\n" + 
                            "\n".join(f"• {w}" for w in extracted.warnings))
                else:
                    st.success(
                        f"✓ Loaded {len(extracted.data)} financial line items "
                        f"for {extracted.company_name} "
                        f"({', '.join(str(y) for y in extracted.years)})"
                    )
            except Exception as e:
                st.error(f"Could not fetch data for '{edgar_ticker}': {e}")
        # ── Company info ──────────────────────────────────────────────────────
        _section("Company information", "#85b7eb")
        ci1, ci2, ci3, ci4 = st.columns(4, gap="medium")
        with ci1:
            _lbl("Company name")
            company = st.text_input(" ", value="",
                                    placeholder="e.g. Apple Inc.",
                                    key="fc2_company",
                                    label_visibility="collapsed")
        with ci2:
            _lbl("Ticker / Sector")
            sector = st.text_input(" ", value="",
                                placeholder="e.g. AAPL / Technology",
                                key="fc2_sector",
                                label_visibility="collapsed")
        with ci3:
            _lbl("Currency / unit")
            currency = st.selectbox(" ",
                                    ["$ (Millions)", "₹ (Crores)",
                                    "€ (Millions)", "£ (Millions)"],
                                    key="fc2_currency",
                                    label_visibility="collapsed")
            unit = currency.split("(")[0].strip()
        with ci4:
            _lbl("Forecast horizon (years)")
            n_fwd = int(st.number_input(" ", value=5.0, min_value=1.0,
                                        max_value=10.0, step=1.0,
                                        key="fc2_nfwd",
                                        label_visibility="collapsed"))

        company = company if company else "Company"
        n_hist = 3

    # ── Historical data input ─────────────────────────────────────────────
    _section("Step 1 — Historical data input ", "#5dcaa5")
    _note(
        "Enter actual reported figures. Costs are negative. "
        "R&D and SG&A are entered separately "
        "All balance sheet items from the most recent year only — "
        "the model uses these as opening balances."
    )

    with st.expander("📋 Historical data entry — click to expand", expanded=True):
        ltm, hist_summary = _hist_input_block(n_hist, unit)

    # Show computed historical ratios
    _section("Historical metrics (auto-computed)", "#44445a")
    st.dataframe(hist_summary, use_container_width=True)

    # ── Forecast assumptions ──────────────────────────────────────────────
    _section("Step 2 — Forecast assumptions (one column per year)", "#afa9ec")
    _note(
        "All assumptions are entered per year. "
        "Growth rates in %, margins in %, days as numbers. "
        "You can enter the same value across all years for a flat assumption, "
        "or vary them year by year for a more detailed model."
    )

    with st.expander("📊 Forecast assumptions grid — click to expand", expanded=True):
        assumptions = _assumption_inputs(n_fwd, ltm, unit)

    # ── Run model ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_run, col_sim, col_hint = st.columns([1, 1, 2])
    with col_run:
        run_model = st.button("▶  Run 3-statement model",
                              type="primary", key="fc2_run",
                              use_container_width=True)
    with col_sim:
        run_sim = st.checkbox("Also run Monte Carlo simulation",
                              value=True, key="fc2_run_sim")
    with col_hint:
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'color:#44445a;padding-top:10px">'
            f'Builds full IS + BS + CF statement. '
            f'Monte Carlo runs 30,000 paths varying growth and margins.</div>',
            unsafe_allow_html=True,
        )

    if not run_model and "fc2_result" not in st.session_state:
        st.info("Complete historical data and assumptions above, then click Run.")
        return

    if run_model:
        with st.spinner("Running 3-statement model..."):
            fwd = run_3_statement_model(ltm, assumptions)
            sim_paths = None
            if run_sim and _SIM_AVAILABLE:
                sim_paths = _run_forecast_simulation(ltm, assumptions, n=30000)
        st.session_state["fc2_result"] = {
            "fwd": fwd, "ltm": ltm,
            "assumptions": assumptions,
            "sim_paths": sim_paths,
            "company": company, "unit": unit,
        }

    res = st.session_state.get("fc2_result")
    if not res:
        return

    fwd         = res["fwd"]
    ltm         = res["ltm"]
    assumptions = res["assumptions"]
    sim_paths   = res["sim_paths"]
    company     = res["company"]
    unit        = res["unit"]

    # ── Key metrics strip ──────────────────────────────────────────────────
    _section("Key output metrics", "#40c080")
    last = fwd[-1]
    first = fwd[0]
    rev_cagr = (last.revenue / ltm.revenue) ** (1/len(fwd)) - 1
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Revenue CAGR",   f"{rev_cagr:.1%}")
    m2.metric(f"Yr+{len(fwd)} Revenue", f"${last.revenue:,.0f}")
    m3.metric(f"Yr+{len(fwd)} EBITDA",  f"${last.ebitda:,.0f}")
    m4.metric(f"Yr+{len(fwd)} Adj EBITDA", f"${last.adj_ebitda:,.0f}")
    m5.metric(f"Yr+{len(fwd)} Net income", f"${last.net_income:,.0f}")
    m6.metric("Balance check",  f"${last.balance_check:.1f}" +
              (" ✓" if abs(last.balance_check) < 0.1 else " ⚠"))

    # ── Output tabs ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Income statement",
        "Balance sheet",
        "Cash flow statement",
        "Supporting schedules",
        "Charts",
        "Simulation overlay",
    ])

    # ── Tab 1: Income statement ───────────────────────────────────────────
    with tabs[0]:
        _section("Income statement — LTM + forecast", "#85b7eb")
        is_df = _make_is_df(ltm, fwd, unit)
        st.dataframe(is_df, use_container_width=True)
        _dl("Download income statement",
            _to_excel({"Income Statement": is_df.reset_index()}),
            "income_statement.xlsx", "dl_fc_is")

    # ── Tab 2: Balance sheet ──────────────────────────────────────────────
    with tabs[1]:
        _section("Balance sheet", "#5dcaa5")
        bs_df = _make_bs_df(ltm, fwd)
        st.dataframe(bs_df, use_container_width=True)

        # Highlight balance check
        for y in fwd:
            if abs(y.balance_check) > 0.5:
                st.warning(
                    f"Balance sheet does not balance in {y.year}: "
                    f"${y.balance_check:.1f} gap. "
                    f"Check revolver / other assumptions."
                )

        _dl("Download balance sheet",
            _to_excel({"Balance Sheet": bs_df.reset_index()}),
            "balance_sheet.xlsx", "dl_fc_bs")

    # ── Tab 3: Cash flow statement ────────────────────────────────────────
    with tabs[2]:
        _section("Cash flow statement", "#ef9f27")
        cf_df = _make_cf_df(fwd)
        st.dataframe(cf_df, use_container_width=True)
        _dl("Download cash flow",
            _to_excel({"Cash Flow": cf_df.reset_index()}),
            "cash_flow.xlsx", "dl_fc_cf")

    # ── Tab 4: Supporting schedules ───────────────────────────────────────
    with tabs[3]:
        _section("PP&E roll-forward", "#5dcaa5")
        _note("Beginning + Capex − Depreciation = Ending.")
        ppe_df = _make_ppe_df(ltm, fwd)
        st.dataframe(ppe_df, use_container_width=True)

        _section("Retained earnings roll-forward", "#85b7eb")
        _note("Beginning + Net income − Dividends − Repurchases = Ending.")
        re_df = _make_re_df(ltm, fwd)
        st.dataframe(re_df, use_container_width=True)

        _section("Working capital schedule (AR/Inventory/AP days)", "#afa9ec")
        wc_rows = []
        prev_nwc = ltm.ar + ltm.inventory - ltm.ap
        for y in fwd:
            wc_rows.append({
                "Year":          y.year,
                "AR":            f"${y.ar:,.1f}",
                "Inventory":     f"${y.inventory:,.1f}",
                "AP":            f"(${abs(y.ap):,.1f})",
                "Net WC":        f"${y.nwc:,.1f}",
                "ΔNWC (cash)":   f"${y.delta_nwc:+,.1f}",
                "Cash convers. cycle":
                    f"{(assumptions[fwd.index(y)].ar_days + assumptions[fwd.index(y)].inv_days - assumptions[fwd.index(y)].ap_days):.0f} days",
            })
        wc_df = pd.DataFrame(wc_rows).set_index("Year")
        st.dataframe(wc_df, use_container_width=True)

        _section("Interest schedule", "#40a0c0")
        int_rows = []
        for i, y in enumerate(fwd):
            a = assumptions[i]
            int_rows.append({
                "Year":          y.year,
                "Cash (beg)":    f"${(ltm.cash if i==0 else fwd[i-1].cash):,.1f}",
                "Rate on cash":  f"{a.interest_rate_cash:.2%}",
                "Interest inc":  f"${y.interest_inc:,.1f}",
                "Debt (beg)":    f"${(ltm.ltd if i==0 else fwd[i-1].ltd):,.1f}",
                "Rate on debt":  f"{a.interest_rate_debt:.2%}",
                "Interest exp":  f"(${abs(y.interest_exp):,.1f})",
            })
        int_df = pd.DataFrame(int_rows).set_index("Year")
        st.dataframe(int_df, use_container_width=True)

        _section("Revolver (model plug)", "#f0997b")
        _note("The revolver draws when ending cash would fall below the minimum cash balance.")
        rev_rows = []
        for i, y in enumerate(fwd):
            rev_rows.append({
                "Year":         y.year,
                "Draw / (Repay)": f"${y.revolver_draw:+,.1f}",
                "Ending balance": f"${y.revolver:,.1f}",
                "Ending cash":    f"${y.cash:,.1f}",
            })
        rev_df = pd.DataFrame(rev_rows).set_index("Year")
        st.dataframe(rev_df, use_container_width=True)

        # Combined Excel download
        _dl("Download all schedules",
            _to_excel({
                "PP&E Schedule":  ppe_df.reset_index(),
                "Retained Earnings": re_df.reset_index(),
                "Working Capital":   wc_df.reset_index(),
                "Interest Schedule": int_df.reset_index(),
                "Revolver":          rev_df.reset_index(),
            }),
            "supporting_schedules.xlsx", "dl_fc_sched")

    # ── Tab 5: Charts ─────────────────────────────────────────────────────
    with tabs[4]:
        _section("Financial charts", "#c4c4d4")
        fig_is = _plot_is_charts(ltm, fwd, company, unit)
        st.pyplot(fig_is, use_container_width=True)
        plt.close(fig_is)

        # Waterfall: EBITDA to Net income bridge (last forecast year)
        _section(f"EBITDA → Net income bridge ({fwd[-1].year})", "#85b7eb")
        bridge_items = [
            ("EBITDA",          fwd[-1].ebitda, "#6060c0", 0),
            ("Interest income", fwd[-1].interest_inc, "#40c080", fwd[-1].ebitda),
            ("Interest expense",fwd[-1].interest_exp, "#c06060",
             fwd[-1].ebitda + fwd[-1].interest_inc),
            ("Other income",    fwd[-1].other_income, "#c0a040",
             fwd[-1].ebitda + fwd[-1].interest_inc + fwd[-1].interest_exp),
            ("Taxes",           fwd[-1].taxes, "#c06060",
             fwd[-1].pretax),
            ("Net income",      fwd[-1].net_income, "#40c080", 0),
        ]
        fig_br, ax_br = plt.subplots(figsize=(10, 4))
        for i, (lbl, val, col, bot) in enumerate(bridge_items):
            ax_br.bar(i, val, bottom=bot, color=col, alpha=0.8, width=0.5,
                      edgecolor="#16162a", lw=0.5)
            ax_br.text(i, bot + val + 0.3, f"${val:,.0f}", ha="center",
                       fontsize=8, color="#c4c4d4")
        ax_br.set_xticks(range(len(bridge_items)))
        ax_br.set_xticklabels([b[0] for b in bridge_items], fontsize=8)
        ax_br.set_title(f"EBITDA to Net income bridge — {fwd[-1].year}")
        ax_br.grid(axis="y")
        st.pyplot(fig_br, use_container_width=True)
        plt.close(fig_br)

    # ── Tab 6: Simulation overlay ─────────────────────────────────────────
    with tabs[5]:
        if sim_paths is None:
            st.info("Enable 'Also run Monte Carlo simulation' and re-run the model.")
        else:
            _section("Revenue simulation — confidence bands", "#6060c0")

            fig_sim = _plot_simulation_charts(fwd, sim_paths, company, unit)
            if fig_sim:
                st.pyplot(fig_sim, use_container_width=True)
                plt.close(fig_sim)

            _section("Simulation summary statistics", "#c4c4d4")
            rev_final = sim_paths["revenue"][:, -1]
            ebd_final = sim_paths["ebitda"][:, -1]
            stats_rows = []
            for name, arr in [(f"Revenue Yr+{len(fwd)}", rev_final),
                               (f"EBITDA Yr+{len(fwd)}",  ebd_final)]:
                stats_rows.append({
                    "Metric":  name,
                    "Mean":    f"${np.mean(arr):,.0f}",
                    "Median":  f"${np.median(arr):,.0f}",
                    "5th pct": f"${np.percentile(arr,5):,.0f}",
                    "25th pct":f"${np.percentile(arr,25):,.0f}",
                    "75th pct":f"${np.percentile(arr,75):,.0f}",
                    "95th pct":f"${np.percentile(arr,95):,.0f}",
                    "Deterministic": f"${([fwd[-1].revenue,fwd[-1].ebitda][stats_rows.__len__()]):,.0f}",
                })
            st.dataframe(pd.DataFrame(stats_rows).set_index("Metric"),
                         use_container_width=True)

            # Probability of hitting EBITDA targets
            _section("Probability analysis", "#40c080")
            det_ebitda_final = fwd[-1].ebitda
            targets = [det_ebitda_final * 0.80,
                       det_ebitda_final * 0.90,
                       det_ebitda_final,
                       det_ebitda_final * 1.10,
                       det_ebitda_final * 1.20]
            prob_rows = []
            for t in targets:
                p = (ebd_final >= t).mean()
                prob_rows.append({
                    f"EBITDA target ({unit})": f"${t:,.0f}",
                    "P(≥ target)": f"{p:.1%}",
                    "Scenario": ("Bear" if t > det_ebitda_final else
                                 "Bull" if t < det_ebitda_final else "Base"),
                })
            st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)

            # Download simulation data
            sim_sample = pd.DataFrame({
                f"Revenue Yr+{len(fwd)}": rev_final[:5000],
                f"EBITDA Yr+{len(fwd)}":  ebd_final[:5000],
                "Growth draw":   sim_paths["g_draws"][:5000],
                "Margin draw":   sim_paths["m_draws"][:5000],
            })
            _dl("Download simulation sample",
                _to_excel({"Simulation": sim_sample}),
                "simulation_results.xlsx", "dl_fc_sim")

    # ── Master Excel download ──────────────────────────────────────────────
    st.markdown("---")
    _section("Download complete model", "#44445a")
    is_df  = _make_is_df(ltm, fwd, unit)
    bs_df  = _make_bs_df(ltm, fwd)
    cf_df  = _make_cf_df(fwd)
    ppe_df = _make_ppe_df(ltm, fwd)
    re_df  = _make_re_df(ltm, fwd)

    _dl("⬇ Download complete 3-statement model (Excel)",
        _to_excel({
            "Income Statement":   is_df.reset_index(),
            "Balance Sheet":      bs_df.reset_index(),
            "Cash Flow":          cf_df.reset_index(),
            "PP&E Schedule":      ppe_df.reset_index(),
            "Retained Earnings":  re_df.reset_index(),
        }),
        f"{company}_3statement_model.xlsx", "dl_fc_master")