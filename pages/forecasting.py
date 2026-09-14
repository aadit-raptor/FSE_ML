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

from core.forecasting import (  # noqa: F401  (re-exported)
    ForecastAssumptions, ForecastYear, HistoricalYear, run_3_statement_model,
)
from core.forecasting import opening_bs_gap as _opening_bs_gap
from core.forecasting import run_forecast_simulation
from core.forecasting import (
    HISTORICAL_FIELDS, assumptions_from_grid, default_history_value,
    historical_metrics, ltm_from_history, revenue_cagr, seed_assumptions,
    simulation_summary,
)
try:
    from ml.edgar_extractor import fetch_financials, financials_to_session_state
    _EDGAR_AVAILABLE = True
except ImportError:
    _EDGAR_AVAILABLE = False
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
        key=key, width="stretch")

# ── 3-Statement Model Engine ──────────────────────────────────────────────────








# ── Historical input table ────────────────────────────────────────────────────

_HIST_DEFAULTS = {k: (d, st) for k, d, st in HISTORICAL_FIELDS}


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
        ("h_rev",       f"Revenue ({unit})", *_HIST_DEFAULTS["h_rev"]),
        ("h_cogs",      f"Cost of sales ({unit}, negative)", *_HIST_DEFAULTS["h_cogs"]),
        ("h_rd",        f"R&D ({unit}, 0 if N/A, negative)", *_HIST_DEFAULTS["h_rd"]),
        ("h_sga",       f"SG&A ({unit}, negative)", *_HIST_DEFAULTS["h_sga"]),
        ("h_int_inc",   f"Interest income ({unit})", *_HIST_DEFAULTS["h_int_inc"]),
        ("h_int_exp",   f"Interest expense ({unit}, negative)", *_HIST_DEFAULTS["h_int_exp"]),
        ("h_other",     f"Other income/expense ({unit})", *_HIST_DEFAULTS["h_other"]),
        ("h_tax",       f"Taxes ({unit}, negative)", *_HIST_DEFAULTS["h_tax"]),
        ("h_da",        f"D&A ({unit}, positive add-back)", *_HIST_DEFAULTS["h_da"]),
        ("h_sbc",       f"SBC ({unit}, positive add-back)", *_HIST_DEFAULTS["h_sbc"]),
        # ── Balance sheet ─────────────────────────────────────
        ("__hdr_bs__",  "── BALANCE SHEET (latest year) ──", None, None),
        ("h_cash",      f"Cash & equivalents ({unit})", *_HIST_DEFAULTS["h_cash"]),
        ("h_ar",        f"Accounts receivable ({unit})", *_HIST_DEFAULTS["h_ar"]),
        ("h_inv",       f"Inventories ({unit})", *_HIST_DEFAULTS["h_inv"]),
        ("h_ocurr",     f"Other current assets ({unit})", *_HIST_DEFAULTS["h_ocurr"]),
        ("h_ppe",       f"PP&E net ({unit})", *_HIST_DEFAULTS["h_ppe"]),
        ("h_nca",       f"Other non-current assets ({unit})", *_HIST_DEFAULTS["h_nca"]),
        ("h_lta",       f"Goodwill & other long-term assets ({unit})", *_HIST_DEFAULTS["h_lta"]),
        ("h_ap",        f"Accounts payable ({unit})", *_HIST_DEFAULTS["h_ap"]),
        ("h_ocl",       f"Other current liabilities ({unit})", *_HIST_DEFAULTS["h_ocl"]),
        ("h_def",       f"Deferred revenue ({unit})", *_HIST_DEFAULTS["h_def"]),
        ("h_ltd",       f"Long-term debt ({unit})", *_HIST_DEFAULTS["h_ltd"]),
        ("h_ncl",       f"Other non-current liabilities ({unit})", *_HIST_DEFAULTS["h_ncl"]),
        ("h_cs",        f"Common stock ({unit})", *_HIST_DEFAULTS["h_cs"]),
        ("h_re",        f"Retained earnings ({unit})", *_HIST_DEFAULTS["h_re"]),
        ("h_oci",       f"Other comprehensive income ({unit})", *_HIST_DEFAULTS["h_oci"]),
        # ── Additional data ───────────────────────────────────
        ("__hdr_ad__",  "── ADDITIONAL DATA ──", None, None),
        ("h_capex",     f"Capital expenditures ({unit})", *_HIST_DEFAULTS["h_capex"]),
        ("h_divs",      f"Dividends ({unit})", *_HIST_DEFAULTS["h_divs"]),
        ("h_buybacks",  f"Buybacks / repurchases ({unit})", *_HIST_DEFAULTS["h_buybacks"]),
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
            default_j = default_history_value(default_latest, j, n_hist)
            with row_cols[j+1]:
                v = st.number_input(
                    " ", value=default_j, step=float(step),
                    key=f"hist_{key}_{j}",
                    label_visibility="collapsed",
                )
            year_vals.append(v)
        data[key] = year_vals

    def _pct(v):
        return "—" if v is None else f"{v:.1%}"

    hist_rows_display = [{
        "Year": yr_labels[j],
        f"Revenue ({unit})": f"${m['revenue']:,.1f}",
        "Revenue growth": _pct(m["revenue_growth"]),
        "Gross margin": _pct(m["gross_margin"]),
        "R&D %": _pct(m["rd_pct"]),
        "SG&A %": _pct(m["sga_pct"]),
        "EBITDA margin": _pct(m["ebitda_margin"]),
        "Adj EBITDA margin": _pct(m["adj_ebitda_margin"]),
    } for j, m in enumerate(historical_metrics(data, n_hist))]

    ltm = ltm_from_history(data)
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

    # Set by the EDGAR fetch. Deleting the widget keys is not enough: a keyed
    # widget keeps its identity, so the browser sends its old value back on the
    # next interaction. Assigning the new defaults through session state is
    # what updates the browser too.
    reseed = st.session_state.pop("_fc2_reseed_grid", False)

    seeds = seed_assumptions(ltm)

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
        ("rev_g",     "Revenue growth (%)", seeds["rev_g"],   0.5,  "#5dcaa5", False),
        ("gm",        "Gross profit margin (%)", seeds["gm"], 0.5, "#85b7eb", False),
        ("rd",        "R&D % of sales", seeds["rd"], 0.25,"#afa9ec", False),
        ("sga",       "SG&A % of sales", seeds["sga"],0.25,"#afa9ec", False),
        ("tax",       "Tax rate (%)", seeds["tax"],0.5,"#ef9f27", False),
        ("__da__",    "── D&A & CAPEX ──",        None, None, None, True),
        ("da",        "D&A % of revenue", seeds["da"], 0.25, "#5dcaa5", False),
        ("sbc",       "SBC % of revenue", seeds["sbc"],0.25, "#5dcaa5", False),
        ("capex",     "Capex % of revenue", seeds["capex"],0.25,"#ef9f27", False),
        ("__wc__",    "── WORKING CAPITAL (days) ──", None, None, None, True),
        ("ar_d",      "AR days (Revenue÷365)", seeds["ar_d"],  1.0, "#85b7eb", False),
        ("inv_d",     "Inventory days (COGS÷365)", seeds["inv_d"], 1.0, "#85b7eb", False),
        ("ap_d",      "AP days (COGS÷365)", seeds["ap_d"],  1.0, "#85b7eb", False),
        ("__bs__",    "── OTHER B/S ASSUMPTIONS ──", None, None, None, True),
        ("ocl_pct",   "Other current liab % rev", seeds["ocl_pct"],0.25,"#afa9ec", False),
        ("def_pct",   "Deferred rev % of revenue", seeds["def_pct"], 0.25,"#afa9ec", False),
        ("nca_pct",   "Other NCA % of revenue", seeds["nca_pct"],0.25,"#afa9ec", False),
        ("__other__", "── FINANCING & OTHER ──", None, None, None, True),
        ("other_inc", f"Other income ({unit}, flat)", seeds["other_inc"],0.1,"#c4c4d4", False),
        ("divs",      f"Dividends ({unit}, flat)", seeds["divs"],  1.0,"#f0997b", False),
        ("buybacks",  f"Buybacks ({unit}, flat)", seeds["buybacks"],5.0,"#f0997b", False),
        ("ltd_chg",   f"LTD net change ({unit})", seeds["ltd_chg"],  5.0, "#f0997b", False),
        ("r_cash",    "Interest rate on cash (%)", seeds["r_cash"],  0.1, "#40a0c0", False),
        ("r_debt",    "Interest rate on debt (%)", seeds["r_debt"],  0.1, "#c06060", False),
        ("min_cash",  f"Minimum cash ({unit})", seeds["min_cash"], 5.0,"#44445a", False),
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
            wkey = f"fwd_{key}_{j}"
            if reseed or wkey not in st.session_state:
                st.session_state[wkey] = float(default)
            with row_cols[j+1]:
                val = st.number_input(
                    " ", step=float(step), key=wkey,
                    label_visibility="collapsed",
                )
            collected[key].append(val)

    assumptions = assumptions_from_grid(collected, n_fwd)
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
        r("Goodwill & other LT",[ltm.other_lta]+ [y.other_lta   for y in fwd]),
        r("TOTAL ASSETS",       [ltm.cash+ltm.ar+ltm.inventory+ltm.other_current+ltm.ppe_net+ltm.other_nca+ltm.other_lta]
                                + [y.total_assets for y in fwd]),
        r("Accounts payable",   [ltm.ap]       + [y.ap_abs       for y in fwd]),
        r("Other curr liab",    [ltm.other_cl] + [y.other_cl     for y in fwd]),
        r("Deferred revenue",   [ltm.deferred_rev]+ [y.deferred_rev for y in fwd]),
        r("Revolver",           [0.0]          + [y.revolver     for y in fwd]),
        r("Long-term debt",     [ltm.ltd]      + [y.ltd          for y in fwd]),
        r("Other non-curr liab",[ltm.other_ncl]+ [y.other_ncl    for y in fwd]),
        r("TOTAL LIABILITIES",  [ltm.ap+ltm.other_cl+ltm.deferred_rev+ltm.ltd+ltm.other_ncl]
                                + [y.total_liab for y in fwd]),
        r("Common stock",       [ltm.common_stock]+ [y.common_stock for y in fwd]),
        r("Retained earnings",  [ltm.retained_earnings]+ [y.retained_earn for y in fwd]),
        r("OCI",                [ltm.oci]      + [y.oci          for y in fwd]),
        r("TOTAL EQUITY",       [ltm.common_stock+ltm.retained_earnings+ltm.oci]
                                + [y.total_equity for y in fwd]),
        r("Balance check",      [_opening_bs_gap(ltm)] + [y.balance_check for y in fwd]),
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
                                width="stretch")
    with ec3:
        if 'edgar_company_name' in st.session_state:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
                f'color:#5dcaa5;padding-top:10px">✓ Loaded: {st.session_state.edgar_company_name}</div>',
                unsafe_allow_html=True,
            )

    if edgar_fetch and edgar_ticker and not _EDGAR_AVAILABLE:
        st.error("EDGAR autofill is unavailable — the optional `ml` "
                 "package is not installed. Enter figures manually below.")
    if edgar_fetch and edgar_ticker and _EDGAR_AVAILABLE:
        with st.spinner(f"Fetching financials for {edgar_ticker.upper()} from SEC EDGAR..."):
            try:
                extracted = fetch_financials(edgar_ticker.upper(), n_years=3)
                session_vals = financials_to_session_state(extracted)
                for key, val in session_vals.items():
                    st.session_state[key] = val
                st.session_state['edgar_company_name'] = extracted.company_name
                st.session_state['fc2_company'] = extracted.company_name
                # Re-seed the forecast grid from the fetched company; without
                # this it kept assumptions derived from the previous
                # historicals (e.g. the placeholder 38.5% gross margin for
                # MSFT). See _assumption_inputs() for why this is a flag.
                st.session_state["_fc2_reseed_grid"] = True
                st.session_state.pop("fc2_result", None)
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
    st.dataframe(hist_summary, width="stretch")

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
                              width="stretch")
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
                sim_paths = run_forecast_simulation(ltm, assumptions, n=30000)
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
    rev_cagr = revenue_cagr(ltm, fwd)
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
        st.dataframe(is_df, width="stretch")
        _dl("Download income statement",
            _to_excel({"Income Statement": is_df.reset_index()}),
            "income_statement.xlsx", "dl_fc_is")

    # ── Tab 2: Balance sheet ──────────────────────────────────────────────
    with tabs[1]:
        _section("Balance sheet", "#5dcaa5")
        bs_df = _make_bs_df(ltm, fwd)
        st.dataframe(bs_df, width="stretch")

        # Separate an unbalanced *input* from a gap the *forecast* introduces.
        # The model carries the opening gap forward unchanged, so blaming the
        # forecast assumptions for it would send the user to the wrong place.
        opening_gap = _opening_bs_gap(ltm)
        if abs(opening_gap) > 0.5:
            st.warning(
                f"The historical (LTM) balance sheet does not balance: "
                f"assets exceed liabilities + equity by ${opening_gap:,.1f}. "
                f"The forecast carries this gap forward unchanged. Check the "
                f"balance sheet inputs above -- with EDGAR data, lines the "
                f"extractor could not map (and so left at 0) are a common cause."
            )
        for y in fwd:
            model_gap = y.balance_check - opening_gap
            if abs(model_gap) > 0.5:
                st.warning(
                    f"The forecast introduced a ${model_gap:,.1f} balance sheet "
                    f"gap by {y.year}. Check revolver / other assumptions."
                )

        _dl("Download balance sheet",
            _to_excel({"Balance Sheet": bs_df.reset_index()}),
            "balance_sheet.xlsx", "dl_fc_bs")

    # ── Tab 3: Cash flow statement ────────────────────────────────────────
    with tabs[2]:
        _section("Cash flow statement", "#ef9f27")
        cf_df = _make_cf_df(fwd)
        st.dataframe(cf_df, width="stretch")
        _dl("Download cash flow",
            _to_excel({"Cash Flow": cf_df.reset_index()}),
            "cash_flow.xlsx", "dl_fc_cf")

    # ── Tab 4: Supporting schedules ───────────────────────────────────────
    with tabs[3]:
        _section("PP&E roll-forward", "#5dcaa5")
        _note("Beginning + Capex − Depreciation = Ending.")
        ppe_df = _make_ppe_df(ltm, fwd)
        st.dataframe(ppe_df, width="stretch")

        _section("Retained earnings roll-forward", "#85b7eb")
        _note("Beginning + Net income − Dividends − Repurchases = Ending.")
        re_df = _make_re_df(ltm, fwd)
        st.dataframe(re_df, width="stretch")

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
        st.dataframe(wc_df, width="stretch")

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
        st.dataframe(int_df, width="stretch")

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
        st.dataframe(rev_df, width="stretch")

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
        st.pyplot(fig_is, width="stretch")
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
        st.pyplot(fig_br, width="stretch")
        plt.close(fig_br)

    # ── Tab 6: Simulation overlay ─────────────────────────────────────────
    with tabs[5]:
        if sim_paths is None:
            st.info("Enable 'Also run Monte Carlo simulation' and re-run the model.")
        else:
            _section("Revenue simulation — confidence bands", "#6060c0")

            fig_sim = _plot_simulation_charts(fwd, sim_paths, company, unit)
            if fig_sim:
                st.pyplot(fig_sim, width="stretch")
                plt.close(fig_sim)

            _section("Simulation summary statistics", "#c4c4d4")
            summary = simulation_summary(fwd, sim_paths)
            rev_final = sim_paths["revenue"][:, -1]
            ebd_final = sim_paths["ebitda"][:, -1]
            stats_rows = []
            for name, stats in [(f"Revenue Yr+{len(fwd)}", summary["revenue_final"]),
                                (f"EBITDA Yr+{len(fwd)}",  summary["ebitda_final"])]:
                stats_rows.append({
                    "Metric":  name,
                    "Mean":    f"${stats['mean']:,.0f}",
                    "Median":  f"${stats['median']:,.0f}",
                    "5th pct": f"${stats['p5']:,.0f}",
                    "25th pct":f"${stats['p25']:,.0f}",
                    "75th pct":f"${stats['p75']:,.0f}",
                    "95th pct":f"${stats['p95']:,.0f}",
                    "Deterministic": f"${stats['deterministic']:,.0f}",
                })
            st.dataframe(pd.DataFrame(stats_rows).set_index("Metric"),
                         width="stretch")

            # Probability of hitting EBITDA targets
            _section("Probability analysis", "#40c080")
            prob_rows = []
            for row in summary["target_probabilities"]:
                prob_rows.append({
                    f"EBITDA target ({unit})": f"${row['target']:,.0f}",
                    "P(≥ target)": f"{row['probability']:.1%}",
                    "Scenario": row["scenario"],
                })
            st.dataframe(pd.DataFrame(prob_rows), width="stretch")

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