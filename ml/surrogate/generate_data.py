"""
Generates 100,000 training samples by running the existing
vectorized simulation engine with Latin Hypercube Sampling
across the full parameter space.
Runtime: approximately 15-20 minutes on a standard laptop.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulation.vectorized_simulation import run_vectorized_simulation_full, SimulationParams

def generate(n_samples: int = 100_000, n_per_call: int = 2000, seed: int = 42):
    """
    Latin Hypercube Sampling gives better parameter space coverage
    than pure random sampling with the same number of points.
    Each call runs n_per_call scenarios internally for accuracy.
    """
    sampler = qmc.LatinHypercube(d=11, seed=seed)
    sample = sampler.random(n=n_samples)

    # Parameter bounds: [min, max] for each of 11 parameters
    # [growth_mean, growth_std, exit_mean, exit_std, interest_mean,
    #  gross_margin_mean, gross_margin_std, da_pct, capex_pct,
    #  nwc_pct, debt_pct]
    l_bounds = [0.00, 0.005, 5.0, 0.3, 0.015, 0.15, 0.01, 0.02, 0.01, 0.005, 0.25]
    u_bounds = [0.18, 0.10,  20.0, 4.0, 0.14,  0.80, 0.08, 0.09, 0.10, 0.05,  0.90]
    scaled = qmc.scale(sample, l_bounds, u_bounds)

    records = []
    print(f"Generating {n_samples} training samples...")

    for i, row in enumerate(scaled):
        if i % 1000 == 0:
            print(f"  {i}/{n_samples} ({i/n_samples*100:.1f}%)")

        params = SimulationParams(
            n=n_per_call,
            entry_ebitda=100.0,
            entry_multiple=10.0,
            holding_period=5,
            growth_mean=float(row[0]),
            growth_std=float(row[1]),
            exit_mean=float(row[2]),
            exit_std=float(row[3]),
            interest_mean=float(row[4]),
            gross_margin_mean=float(row[5]),
            gross_margin_std=float(row[6]),
            opex_pct=0.18,
            da_pct=float(row[7]),
            tax_rate=0.25,
            capex_pct=float(row[8]),
            nwc_pct=float(row[9]),
            debt_pct=float(row[10]),
            senior_pct=0.70,
            mezz_spread=0.04,
            n_interest_passes=2,
        )

        try:
            sim = run_vectorized_simulation_full(params, seed=i)
            irr = sim.irr

            records.append({
                # Inputs
                'growth_mean':       row[0],
                'growth_std':        row[1],
                'exit_mean':         row[2],
                'exit_std':          row[3],
                'interest_mean':     row[4],
                'gross_margin_mean': row[5],
                'gross_margin_std':  row[6],
                'da_pct':            row[7],
                'capex_pct':         row[8],
                'nwc_pct':           row[9],
                'debt_pct':          row[10],
                # Outputs — IRR quantiles
                'irr_p5':            float(np.percentile(irr, 5)),
                'irr_p10':           float(np.percentile(irr, 10)),
                'irr_p25':           float(np.percentile(irr, 25)),
                'irr_p50':           float(np.percentile(irr, 50)),
                'irr_p75':           float(np.percentile(irr, 75)),
                'irr_p90':           float(np.percentile(irr, 90)),
                'irr_p95':           float(np.percentile(irr, 95)),
                'irr_mean':          float(np.mean(irr)),
                'irr_std':           float(np.std(irr)),
                'p_above_20':        float((irr > 0.20).mean()),
                'p_wipeout':         float(sim.wipeout_rate),
            })
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")
            continue

    df = pd.DataFrame(records)
    out_path = os.path.join(os.path.dirname(__file__), 'training_data.parquet')
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")
    print(f"Feature summary:\n{df.describe()}")
    return df

if __name__ == '__main__':
    generate()