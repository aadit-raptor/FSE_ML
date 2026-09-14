"""Model settings: defaults, resolution and the simulation correlation matrix.

Moved from pages/settings.py. Functions take settings explicitly (a mapping of
setting key -> value) instead of reading Streamlit session state, so the same
code serves the Streamlit app and the API.
"""
from typing import Mapping, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Defaults — exactly the original hardcoded values
# ---------------------------------------------------------------------------

DEFAULTS = {
    # --- Transaction fees (dashboard page 1) ---
    "tx_fee_pct":          2.3,    # % of EV
    "fin_fee_pct":         2.6,    # % of total debt
    "other_uses":          0.0,    # $M flat

    # --- Deal model defaults (pre-fill pages 1-2) ---
    "def_ebitda":          100.0,
    "def_entry_mult":      10.0,
    "def_exit_mult":       11.0,
    "def_hold":            5,
    "def_growth":          5.0,
    "def_gross_margin":    40.0,
    "def_opex":            18.0,
    "def_tax":             25.0,
    "def_da":              4.0,
    "def_debt_pct":        60.0,
    "def_senior_pct":      70.0,
    "def_base_rate":       6.5,
    "def_mezz_spread":     4.0,
    "def_capex":           4.0,
    "def_nwc":             1.0,
    "def_mincash":         0.0,
    "def_senior_amort":    5.0,    # % of original principal per year

    # --- Sensitivity table ranges ---
    "sens_em_min":         6.0,
    "sens_em_max":         12.0,
    "sens_em_steps":       7,
    "sens_hp_min":         3,
    "sens_hp_max":         7,

    # --- Simulation defaults (SimulationParams) ---
    "mc_n":                50000,
    "mc_growth_mean":      5.0,
    "mc_growth_std":       3.0,
    "mc_exit_mean":        10.0,
    "mc_exit_std":         1.5,
    "mc_rate_mean":        6.5,
    "mc_rate_std":         1.5,
    "mc_gm_mean":          40.0,
    "mc_gm_std":           3.0,
    "mc_hurdle":           20.0,
    "mc_n_passes":         2,
    "mc_clip_irr":         True,

    # --- Correlation matrix (10 unique off-diagonal values) ---
    # Variables: [growth, exit_mult, interest, gross_margin, ebitda_shock]
    "corr_g_em":    0.60,   # growth ↔ exit multiple
    "corr_g_ir":   -0.30,   # growth ↔ interest
    "corr_g_gm":    0.40,   # growth ↔ gross margin
    "corr_g_sh":    0.50,   # growth ↔ EBITDA shock
    "corr_em_ir":  -0.50,   # exit multiple ↔ interest
    "corr_em_gm":   0.30,   # exit multiple ↔ gross margin
    "corr_em_sh":   0.30,   # exit multiple ↔ EBITDA shock
    "corr_ir_gm":  -0.20,   # interest ↔ gross margin
    "corr_ir_sh":  -0.20,   # interest ↔ EBITDA shock
    "corr_gm_sh":   0.20,   # gross margin ↔ EBITDA shock

    # --- Scenario preset multipliers ---
    # Bull
    "bull_growth_mult":   1.50,
    "bull_exit_mult":     1.15,
    "bull_rate_mult":     0.85,
    "bull_margin_mult":   1.05,
    # Recession
    "rec_growth_adj":    -6.0,    # pp adjustment (not multiplier)
    "rec_growth_floor":  -10.0,   # pp floor
    "rec_exit_mult":      0.80,
    "rec_rate_mult":      1.20,
    "rec_margin_mult":    0.93,
    # Stagflation
    "stag_growth_adj":   -3.0,
    "stag_growth_floor": -5.0,
    "stag_exit_mult":     0.85,
    "stag_rate_mult":     1.40,
    "stag_margin_mult":   0.90,
}


def build_corr_matrix(cfg: Mapping) -> np.ndarray:
    """Reconstruct the 5×5 correlation matrix from settings values."""
    g_em  = cfg["corr_g_em"]
    g_ir  = cfg["corr_g_ir"]
    g_gm  = cfg["corr_g_gm"]
    g_sh  = cfg["corr_g_sh"]
    em_ir = cfg["corr_em_ir"]
    em_gm = cfg["corr_em_gm"]
    em_sh = cfg["corr_em_sh"]
    ir_gm = cfg["corr_ir_gm"]
    ir_sh = cfg["corr_ir_sh"]
    gm_sh = cfg["corr_gm_sh"]

    m = np.array([
        [1.00,  g_em,  g_ir,  g_gm,  g_sh],
        [g_em,  1.00, em_ir, em_gm, em_sh],
        [g_ir, em_ir,  1.00, ir_gm, ir_sh],
        [g_gm, em_gm, ir_gm,  1.00, gm_sh],
        [g_sh, em_sh, ir_sh, gm_sh,  1.00],
    ])
    return m


def is_valid_corr(m: np.ndarray) -> bool:
    """Check that the matrix is positive semi-definite (valid correlation matrix)."""
    try:
        np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


def resolve_config(overrides: Optional[Mapping] = None) -> dict:
    """Defaults with any overrides applied. Unknown keys are rejected."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown settings: {sorted(unknown)}")
    return {**DEFAULTS, **overrides}
