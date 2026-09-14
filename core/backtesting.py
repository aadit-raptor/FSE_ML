"""Backtesting: preloaded historical deals and the prediction model.

Moved from pages/backtesting.py; the calculations are unchanged, and settings
are passed in explicitly instead of read from session state.
"""
import numpy as np

from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full


PRELOADED_DEALS = {
    "Burger King (3G Capital, 2010)": {
        "description": "3G Capital acquired BK in 2010 for $4.0B enterprise value. "
                       "Classic operational turnaround — franchise refranchising, cost cuts, "
                       "and international expansion drove EBITDA nearly doubling over 5 years.",
        "entry": {
            "entry_ebitda": 445.0,
            "entry_multiple": 8.75,
            "exit_multiple": 8.8,
            "holding_period": 5,
            "debt_pct": 61.0,
            "senior_pct": 79.0,
            "base_rate": 6.82,
            "mezz_spread": 3.37,
            "revenue_growth": 2.9,
            "gross_margin": 35.5,
            "opex_pct": 22.2,
            "da_pct": 4.1,
            "tax_rate": 33.8,
            "capex_pct": 7.1,
            "nwc_pct": 1.1,
        },
        "actual_years": [2011, 2012, 2013, 2014, 2015],
        "actual": {
            "revenue":   [2574, 2102, 1884, 1910, 2024],
            "ebitda":    [447,  448,  476,  558,  672],
            "net_income":[89,   104,  131,  191,  273],
            "fcf":       [-16,  18,   54,   113,  191],
            "total_debt":[2644, 2626, 2572, 2459, 2268],
        },
        "actual_exit": {
            "exit_ev": 5914.0,
            "net_debt_at_exit": 2150.0,
            "sponsor_equity_entry": 1560.0,
            "moic": 2.39,
            "irr": 19.0,
        },
        "sector": "QSR / Consumer",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Hilton Hotels (Blackstone, 2007)": {
        "description": "Blackstone acquired Hilton for $26B in 2007 — right before the "
                       "financial crisis. Despite severe distress in 2008-09, Blackstone "
                       "restructured, expanded internationally, and exited via IPO in 2013 "
                       "for a record ~$14B profit.",
        "entry": {
            "entry_ebitda": 1400.0,
            "entry_multiple": 18.5,
            "exit_multiple": 16.0,
            "holding_period": 6,
            "debt_pct": 80.0,
            "senior_pct": 75.0,
            "base_rate": 7.5,
            "mezz_spread": 4.0,
            "revenue_growth": 3.0,
            "gross_margin": 38.0,
            "opex_pct": 24.0,
            "da_pct": 5.5,
            "tax_rate": 32.0,
            "capex_pct": 5.0,
            "nwc_pct": 1.5,
        },
        "actual_years": [2008, 2009, 2010, 2011, 2012, 2013],
        "actual": {
            "revenue":   [8162, 7424, 7683, 8783, 9276, 9738],
            "ebitda":    [1270, 1010, 1190, 1520, 1710, 2010],
            "net_income":[-475, -552, -138,  124,  352,  415],
            "fcf":       [-820, -480, 120,   450,  620,  890],
            "total_debt":[20500,20100,19800,19200,18500,17800],
        },
        "actual_exit": {
            "exit_ev": 32000.0,
            "net_debt_at_exit": 17500.0,
            "sponsor_equity_entry": 5200.0,
            "moic": 2.77,
            "irr": 17.2,
        },
        "sector": "Hospitality / Real Estate",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Dell (Silver Lake, 2013)": {
        "description": "Silver Lake and Michael Dell took Dell private for $24.9B "
                       "to restructure from a PC maker to an enterprise IT solutions provider, "
                       "away from public market short-termism. Returned public via VMware "
                       "tracking stock and then direct listing in 2018.",
        "entry": {
            "entry_ebitda": 3700.0,
            "entry_multiple": 6.7,
            "exit_multiple": 7.2,
            "holding_period": 5,
            "debt_pct": 70.0,
            "senior_pct": 78.0,
            "base_rate": 5.5,
            "mezz_spread": 4.5,
            "revenue_growth": -2.0,
            "gross_margin": 22.0,
            "opex_pct": 15.5,
            "da_pct": 3.5,
            "tax_rate": 28.0,
            "capex_pct": 2.5,
            "nwc_pct": 1.0,
        },
        "actual_years": [2014, 2015, 2016, 2017, 2018],
        "actual": {
            "revenue":   [57400, 54000, 50900, 61642, 78660],
            "ebitda":    [3500,  3200,  3050,  5800,  9200],
            "net_income":[-1800, -1100,  -750,  2200,  3000],
            "fcf":       [1200,  2100,   2800,  4200,  5800],
            "total_debt":[15000, 14500,  14200, 46000, 52000],
        },
        "actual_exit": {
            "exit_ev": 70000.0,
            "net_debt_at_exit": 52000.0,
            "sponsor_equity_entry": 7400.0,
            "moic": 2.43,
            "irr": 19.5,
        },
        "sector": "Technology / Enterprise IT",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Freescale Semiconductor (Consortium, 2006)": {
        "description": "Blackstone, Carlyle, TPG, and Permira acquired Freescale for $17.6B "
                       "in 2006 — one of the largest tech LBOs ever at the time. "
                       "The financial crisis decimated semiconductor demand. "
                       "Freescale filed for bankruptcy in 2009.",
        "entry": {
            "entry_ebitda": 1200.0,
            "entry_multiple": 14.7,
            "exit_multiple": 9.0,
            "holding_period": 5,
            "debt_pct": 85.0,
            "senior_pct": 70.0,
            "base_rate": 7.8,
            "mezz_spread": 5.0,
            "revenue_growth": 4.0,
            "gross_margin": 52.0,
            "opex_pct": 35.0,
            "da_pct": 6.5,
            "tax_rate": 30.0,
            "capex_pct": 6.0,
            "nwc_pct": 2.0,
        },
        "actual_years": [2007, 2008, 2009, 2010, 2011],
        "actual": {
            "revenue":   [5843, 5040, 3500, 4440, 4570],
            "ebitda":    [1050, 680,  120,  650,  720],
            "net_income":[-980, -1850,-2400,-280,  80],
            "fcf":       [-200, -600, -900, 100,  180],
            "total_debt":[16200,16800,18000,7200, 6800],
        },
        "actual_exit": {
            "exit_ev": 6500.0,
            "net_debt_at_exit": 6200.0,
            "sponsor_equity_entry": 2650.0,
            "moic": 0.11,
            "irr": -33.0,
        },
        "sector": "Semiconductors / Technology",
        "geography": "Global",
        "outcome": "DISTRESSED",
    },
    "Custom deal (enter manually)": {
        "description": "",
        "entry": {
            "entry_ebitda": 100.0, "entry_multiple": 10.0,
            "exit_multiple": 10.0, "holding_period": 5,
            "debt_pct": 60.0, "senior_pct": 70.0,
            "base_rate": 6.5, "mezz_spread": 4.0,
            "revenue_growth": 5.0, "gross_margin": 40.0,
            "opex_pct": 18.0, "da_pct": 4.0,
            "tax_rate": 25.0, "capex_pct": 4.0, "nwc_pct": 1.0,
        },
        "actual_years": [1, 2, 3, 4, 5],
        "actual": {
            "revenue":   [0, 0, 0, 0, 0],
            "ebitda":    [0, 0, 0, 0, 0],
            "net_income":[0, 0, 0, 0, 0],
            "fcf":       [0, 0, 0, 0, 0],
            "total_debt":[0, 0, 0, 0, 0],
        },
        "actual_exit": {
            "exit_ev": 0.0, "net_debt_at_exit": 0.0,
            "sponsor_equity_entry": 0.0, "moic": 0.0, "irr": 0.0,
        },
        "sector": "", "geography": "", "outcome": "UNKNOWN",
    },
}


def run_prediction_sim(entry, cfg, n=30000):
    """Run MC simulation using entry assumptions to get predicted IRR distribution."""
    params = SimulationParams(
        n=n,
        entry_ebitda=entry["entry_ebitda"],
        entry_multiple=entry["entry_multiple"],
        holding_period=entry["holding_period"],
        growth_mean=entry["revenue_growth"] / 100,
        growth_std=0.04,
        exit_mean=entry["exit_multiple"],
        exit_std=1.5,
        interest_mean=entry["base_rate"] / 100,
        interest_std=0.015,
        gross_margin_mean=entry["gross_margin"] / 100,
        gross_margin_std=0.03,
        opex_pct=entry["opex_pct"] / 100,
        da_pct=entry["da_pct"] / 100,
        tax_rate=entry["tax_rate"] / 100,
        capex_pct=entry["capex_pct"] / 100,
        nwc_pct=entry["nwc_pct"] / 100,
        debt_pct=entry["debt_pct"] / 100,
        senior_pct=entry["senior_pct"] / 100,
        mezz_spread=entry["mezz_spread"] / 100,
        # Same fee assumptions as the deal wizard and Monte Carlo pages, so
        # predictions are fee-inclusive like the actual results they are
        # compared against.
        transaction_fees_pct=cfg['tx_fee_pct'] / 100,
        financing_fees_pct=cfg['fin_fee_pct'] / 100,
        other_uses=cfg['other_uses'],
        n_interest_passes=2,
    )
    sim = run_vectorized_simulation_full(params, seed=42)
    return sim.df["IRR"].values


def predicted_ebitda(entry):
    """Compute deterministic predicted EBITDA path using entry assumptions."""
    ebitda_margin = entry["gross_margin"] / 100 - entry["opex_pct"] / 100 + entry["da_pct"] / 100
    ebitda_margin = max(ebitda_margin, 0.01)
    base_rev = entry["entry_ebitda"] / ebitda_margin
    result = []
    rev = base_rev
    for _ in range(entry["holding_period"]):
        rev *= (1 + entry["revenue_growth"] / 100)
        result.append(round(rev * ebitda_margin, 1))
    return result


def backtest_summary(entry, actual_results, actual_exit, cfg, n=30000):
    """Predicted vs actual comparison for a deal, as on the backtesting page.

    entry          -- entry assumptions (same keys as PRELOADED_DEALS "entry")
    actual_results -- per-year lists: revenue, ebitda, net_income, fcf, total_debt
    actual_exit    -- exit_ev, net_debt_at_exit, sponsor_equity_entry, moic, irr
    """
    hold = entry["holding_period"]
    entry_ebitda, entry_mult = entry["entry_ebitda"], entry["entry_multiple"]
    exit_mult, debt_pct = entry["exit_multiple"], entry["debt_pct"]
    gross_margin, opex_pct, da_pct = entry["gross_margin"], entry["opex_pct"], entry["da_pct"]
    act_exit_ev = actual_exit["exit_ev"]
    act_net_debt = actual_exit["net_debt_at_exit"]
    act_eq_entry = actual_exit["sponsor_equity_entry"]
    act_moic = actual_exit["moic"]
    act_irr = actual_exit["irr"]

    # ── Compute predicted EBITDA path ─────────────────────────────────────
    pred_ebitda = predicted_ebitda(entry)[:hold]
    pred_irr_dist = run_prediction_sim(entry, cfg, n=n)

    entry_ev = entry_ebitda * entry_mult
    pred_exit_ev = pred_ebitda[-1] * exit_mult if pred_ebitda else 0
    pred_net_debt = entry_ev * debt_pct / 100 * 0.75
    entry_debt = entry_ev * debt_pct / 100
    entry_costs = (entry_ev * cfg['tx_fee_pct'] / 100
                   + entry_debt * cfg['fin_fee_pct'] / 100
                   + cfg['other_uses'])
    pred_equity_entry = entry_ev + entry_costs - entry_debt   # fee-inclusive, as in the sim
    pred_exit_equity = max(pred_exit_ev - pred_net_debt, 0)
    pred_moic = pred_exit_equity / pred_equity_entry if pred_equity_entry > 0 else 0
    pred_irr_mean = float(np.mean(pred_irr_dist)) * 100

    # Where the actual IRR landed in the predicted distribution
    pct_rank = float(np.mean(pred_irr_dist * 100 < act_irr)) * 100

    # Error attribution (approximate drivers)
    actual_ebitda_margin = [
        actual_results["ebitda"][i] / actual_results["revenue"][i] * 100
        if actual_results["revenue"][i] > 0 else 0
        for i in range(hold)
    ]
    pred_ebitda_margin = (gross_margin - opex_pct + da_pct)
    ebitda_growth_miss = actual_results["ebitda"][-1] - pred_ebitda[-1]
    margin_diff = (np.mean(actual_ebitda_margin) - pred_ebitda_margin) * (sum(actual_results["revenue"]) / hold) / 100
    fcf_diff = sum(actual_results["fcf"]) - sum(pred_ebitda) * 0.3
    debt_diff = actual_results["total_debt"][-1] - entry_ev * debt_pct / 100 * 0.75
    attribution = {
        "ebitda_growth_miss": ebitda_growth_miss,
        "margin_difference": margin_diff,
        "fcf_conversion": fcf_diff * 0.1,
        "debt_paydown": -debt_diff * 0.05,
    }

    return {
        "predicted_ebitda": pred_ebitda,
        "predicted_irr_distribution": pred_irr_dist,
        "predicted_irr_mean": pred_irr_mean,
        "predicted_moic": pred_moic,
        "predicted_equity_entry": pred_equity_entry,
        "predicted_exit_equity": pred_exit_equity,
        "actual_irr": act_irr,
        "actual_moic": act_moic,
        "actual_equity_entry": act_eq_entry,
        "actual_exit_equity": act_exit_ev - act_net_debt,
        "actual_percentile": pct_rank,
        "actual_ebitda_margin": actual_ebitda_margin,
        "predicted_ebitda_margin": pred_ebitda_margin,
        "attribution": attribution,
    }
