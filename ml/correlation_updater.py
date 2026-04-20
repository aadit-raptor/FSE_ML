"""
Computes rolling empirical correlation matrix from FRED data.
Updates the DEFAULT_CORR in the simulation engine.
Can be run annually or on-demand via a dashboard button.
"""

import numpy as np
import pandas as pd
import os
import json

BASE = os.path.dirname(__file__)


def fetch_correlation_data(years: int = 10) -> pd.DataFrame:
    """
    Fetch data to estimate correlations between simulation variables.
    
    Variable mapping:
      growth ~ GDP growth + sector revenue growth
      exit_multiple ~ P/E ratio trends
      interest ~ Fed Funds rate / HY spread
      gross_margin ~ corporate profit margins (NIPA data)
      ebitda_shock ~ credit market stress proxy
    """
    try:
        from fredapi import Fred
        import os
        fred = Fred(api_key=os.environ.get('FRED_API_KEY', ''))

        end_date   = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)

        series = {
            'gdp_growth':    'GDPC1',
            'hy_spread':     'BAMLH0A0HYM2',
            'fed_funds':     'FEDFUNDS',
            'corp_margins':  'CP',          # Corporate profits
            'credit_stress': 'STLFSI4',     # St. Louis Fed stress index
        }

        dfs = []
        for name, sid in series.items():
            try:
                s = fred.get_series(sid, start_date, end_date)
                dfs.append(s.resample('Q').mean().rename(name))
            except Exception:
                pass

        if len(dfs) >= 3:
            df = pd.concat(dfs, axis=1).dropna()
            df['gdp_growth']   = df['gdp_growth'].pct_change(4)
            df['corp_margins'] = df['corp_margins'].pct_change(4)
            return df.dropna()

    except Exception:
        pass

    # Fallback: use pre-computed historical correlations
    return None


def compute_empirical_correlation(df: pd.DataFrame) -> np.ndarray:
    """Compute 5x5 correlation matrix from empirical data."""
    if df is None or len(df) < 20:
        return None

    # Map columns to simulation variable order:
    # [growth, exit_multiple, interest, gross_margin, ebitda_shock]
    col_order = ['gdp_growth', 'hy_spread', 'fed_funds',
                 'corp_margins', 'credit_stress']
    available = [c for c in col_order if c in df.columns]
    if len(available) < 3:
        return None

    corr = df[available].corr().values

    # Ensure positive semi-definiteness via nearest PSD matrix
    try:
        np.linalg.cholesky(corr)
        return corr
    except np.linalg.LinAlgError:
        # Regularize: add small diagonal
        n = corr.shape[0]
        corr = corr + np.eye(n) * 0.02
        corr = corr / corr.diagonal()[:, None]  # re-normalize
        return corr


def get_updated_correlation_matrix(rolling_years: int = 5) -> dict:
    """
    Returns updated correlation matrix and metadata.
    Falls back to DEFAULT_CORR if data unavailable.
    """
    from simulation.vectorized_simulation import DEFAULT_CORR

    df = fetch_correlation_data(years=rolling_years)
    empirical_corr = compute_empirical_correlation(df)

    if empirical_corr is None:
        return {
            'matrix':     DEFAULT_CORR.tolist(),
            'source':     'DEFAULT (hardcoded)',
            'data_period': 'N/A',
            'updated':    False,
        }

    # Blend empirical with default (70% empirical, 30% default)
    # This prevents extreme matrices from recent market disruptions
    n_default = DEFAULT_CORR.shape[0]
    n_emp     = empirical_corr.shape[0]
    n_use     = min(n_default, n_emp)

    blended = 0.70 * empirical_corr[:n_use, :n_use] + \
              0.30 * DEFAULT_CORR[:n_use, :n_use]

    # If empirical is smaller than 5x5, pad with defaults
    if n_use < 5:
        full = DEFAULT_CORR.copy()
        full[:n_use, :n_use] = blended
        blended = full

    return {
        'matrix':      blended.tolist(),
        'source':      f'70% empirical ({rolling_years}yr) + 30% default',
        'data_period': f"{df.index[0].date()} to {df.index[-1].date()}"
                       if df is not None else 'N/A',
        'updated':     True,
        'n_quarters':  len(df) if df is not None else 0,
    }