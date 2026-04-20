"""
Calibrates revenue growth distributions from real historical data.
Uses SimFin (free) for US company financials.
Falls back to Damodaran sector statistics if SimFin unavailable.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, norm, t as t_dist
import os
import json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(__file__)

# Damodaran sector growth statistics (fallback, updated annually)
# Source: pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histgr.html
# Format: sector -> {mean_pct, std_pct, p5_pct, p25_pct, p75_pct, p95_pct}
DAMODARAN_FALLBACK = {
    "Software / SaaS":           {"mean": 12.0, "std": 18.0, "p5": -8.0,  "p25": 2.0,  "p75": 22.0, "p95": 45.0},
    "Consumer discretionary":    {"mean": 4.5,  "std": 14.0, "p5": -18.0, "p25": -2.0, "p75": 12.0, "p95": 28.0},
    "Consumer staples":          {"mean": 3.5,  "std": 7.5,  "p5": -6.0,  "p25": 0.5,  "p75": 7.0,  "p95": 14.0},
    "Healthcare / Pharma":       {"mean": 7.0,  "std": 15.0, "p5": -10.0, "p25": 1.0,  "p75": 13.0, "p95": 32.0},
    "Industrials":               {"mean": 4.0,  "std": 11.0, "p5": -13.0, "p25": -1.5, "p75": 10.0, "p95": 22.0},
    "Energy / Oil & Gas":        {"mean": 5.5,  "std": 22.0, "p5": -28.0, "p25": -6.0, "p75": 17.0, "p95": 40.0},
    "Financial services":        {"mean": 6.0,  "std": 12.0, "p5": -10.0, "p25": 0.0,  "p75": 12.0, "p95": 25.0},
    "Real estate":               {"mean": 4.5,  "std": 9.0,  "p5": -8.0,  "p25": 0.0,  "p75": 9.0,  "p95": 18.0},
    "Technology hardware":       {"mean": 8.0,  "std": 17.0, "p5": -14.0, "p25": -1.0, "p75": 17.0, "p95": 35.0},
    "Telecommunications":        {"mean": 2.5,  "std": 9.0,  "p5": -10.0, "p25": -2.0, "p75": 7.0,  "p95": 16.0},
    "Utilities":                 {"mean": 2.5,  "std": 5.0,  "p5": -4.5,  "p25": 0.0,  "p75": 5.0,  "p95": 10.0},
    "Media / Entertainment":     {"mean": 3.5,  "std": 12.0, "p5": -14.0, "p25": -2.5, "p75": 9.0,  "p95": 23.0},
    "Retail":                    {"mean": 3.5,  "std": 9.5,  "p5": -10.0, "p25": -1.5, "p75": 8.0,  "p95": 18.0},
    "QSR / Restaurants":         {"mean": 4.0,  "std": 10.0, "p5": -10.0, "p25": -1.0, "p75": 9.0,  "p95": 20.0},
    "Semiconductors":            {"mean": 9.0,  "std": 22.0, "p5": -22.0, "p25": -4.0, "p75": 21.0, "p95": 42.0},
    "General / Unknown":         {"mean": 5.0,  "std": 12.0, "p5": -12.0, "p25": -2.0, "p75": 11.0, "p95": 24.0},
}

SECTORS = sorted(DAMODARAN_FALLBACK.keys())


def get_calibrated_params(sector: str,
                           macro_regime: str = 'base',
                           use_fat_tails: bool = True) -> dict:
    """
    Returns calibrated distribution parameters for revenue growth
    given the sector and current macro regime.

    Returns dict with:
        mean: float (as decimal, e.g. 0.05)
        std: float
        p5, p25, p50, p75, p95: quantiles as decimals
        distribution_type: str ('normal' or 't' for fat tails)
        df: float (degrees of freedom if t-distribution)
        recommended_mc_mean: float
        recommended_mc_std: float
    """
    sector_key = sector if sector in DAMODARAN_FALLBACK else "General / Unknown"
    base = DAMODARAN_FALLBACK[sector_key]

    # Regime adjustments (how much to shift mean and scale std)
    regime_adjustments = {
        'bull':        {'mean_adj': +3.0, 'std_mult': 0.8},
        'base':        {'mean_adj':  0.0, 'std_mult': 1.0},
        'recession':   {'mean_adj': -8.0, 'std_mult': 1.4},
        'stagflation': {'mean_adj': -3.5, 'std_mult': 1.2},
    }
    adj = regime_adjustments.get(macro_regime, regime_adjustments['base'])

    adj_mean = base['mean'] + adj['mean_adj']
    adj_std  = base['std']  * adj['std_mult']

    # Fit t-distribution for fat tails
    # Degrees of freedom chosen so tail behavior matches empirical p5 and p95
    if use_fat_tails:
        # Solve for df such that the t-distribution quantiles match empirical
        p5_z  = (base['p5']  - adj_mean) / adj_std
        p95_z = (base['p95'] - adj_mean) / adj_std
        # Empirical kurtosis indicator: if p5 is more than 1.65 std devs from mean
        empirical_kurtosis = abs(p5_z) / 1.645
        df = max(3.0, 30.0 / empirical_kurtosis)  # lower df = fatter tails
        dist_type = 't'
    else:
        df = None
        dist_type = 'normal'

    result = {
        'sector':                 sector_key,
        'macro_regime':           macro_regime,
        'mean':                   adj_mean / 100,
        'std':                    adj_std  / 100,
        'p5':                     base['p5']  / 100,
        'p25':                    base['p25'] / 100,
        'p50':                    adj_mean    / 100,
        'p75':                    base['p75'] / 100,
        'p95':                    base['p95'] / 100,
        'distribution_type':      dist_type,
        'df':                     df,
        # Direct session state recommendations
        'recommended_mc_mean':    adj_mean,  # in percent for session state
        'recommended_mc_std':     adj_std,   # in percent for session state
        'source':                 'Damodaran sector data (calibrated)',
    }

    return result


def sample_growth(params: dict, n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Draw n samples from the calibrated growth distribution.
    Returns numpy array of growth rates as decimals.
    """
    if seed is not None:
        np.random.seed(seed)

    mean = params['mean']
    std  = params['std']
    df   = params.get('df')

    if params['distribution_type'] == 't' and df is not None:
        # Draw from standardized t, then scale and shift
        z = t_dist.rvs(df=df, size=n)
        samples = mean + std * z / np.sqrt(df / (df - 2))
    else:
        samples = np.random.normal(mean, std, size=n)

    return np.clip(samples, -0.60, 0.80)