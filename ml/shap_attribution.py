"""
XGBoost model trained on simulation outputs.
SHAP values explain which assumptions drive predicted IRR.
Training data generated from your own simulation engine.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from dataclasses import dataclass
from typing import Dict, List, Optional

BASE = os.path.dirname(__file__)

FEATURE_COLS = [
    'growth_mean', 'growth_std', 'exit_mean', 'exit_std',
    'interest_mean', 'gross_margin_mean', 'da_pct',
    'capex_pct', 'debt_pct', 'senior_pct', 'mezz_spread',
]

FEATURE_LABELS = {
    'growth_mean':        'Revenue growth (mean)',
    'growth_std':         'Revenue growth (uncertainty)',
    'exit_mean':          'Exit multiple (mean)',
    'exit_std':           'Exit multiple (uncertainty)',
    'interest_mean':      'Interest rate',
    'gross_margin_mean':  'Gross margin',
    'da_pct':             'D&A rate',
    'capex_pct':          'Capex rate',
    'debt_pct':           'Leverage (Debt/EV)',
    'senior_pct':         'Senior % of debt',
    'mezz_spread':        'Mezz spread',
}


@dataclass
class SHAPResult:
    predicted_irr:       float
    shap_values:         Dict[str, float]   # feature -> shap_value
    ranked_drivers:      List[tuple]        # [(feature, shap_val, direction), ...]
    top_upside_driver:   str
    top_downside_driver: str
    explanation:         str                # natural language summary


def generate_training_data(n: int = 30000) -> pd.DataFrame:
    """Generate training data from simulation engine."""
    from simulation.vectorized_simulation import (
        run_vectorized_simulation_full, SimulationParams
    )

    rng = np.random.default_rng(42)
    records = []
    print(f"Generating {n} samples for SHAP model training...")

    for i in range(n):
        if i % 3000 == 0:
            print(f"  {i}/{n}")

        # Sample parameters from wide distributions
        gm = float(rng.uniform(0.0, 0.15))
        gs = float(rng.uniform(0.005, 0.08))
        em = float(rng.uniform(5.0, 20.0))
        es = float(rng.uniform(0.3, 4.0))
        ir = float(rng.uniform(0.02, 0.13))
        gm_m= float(rng.uniform(0.15, 0.80))
        da  = float(rng.uniform(0.02, 0.09))
        cp  = float(rng.uniform(0.01, 0.10))
        dp  = float(rng.uniform(0.25, 0.88))
        sp  = float(rng.uniform(0.50, 0.95))
        mz  = float(rng.uniform(0.01, 0.07))

        params = SimulationParams(
            n=2000, entry_ebitda=100.0, entry_multiple=10.0,
            holding_period=5,
            growth_mean=gm, growth_std=gs,
            exit_mean=em,   exit_std=es,
            interest_mean=ir,
            gross_margin_mean=gm_m, gross_margin_std=0.03,
            opex_pct=0.18, da_pct=da, tax_rate=0.25,
            capex_pct=cp, nwc_pct=0.01,
            debt_pct=dp, senior_pct=sp, mezz_spread=mz,
            n_interest_passes=2,
        )
        try:
            sim = run_vectorized_simulation_full(params, seed=i)
            records.append({
                'growth_mean': gm,   'growth_std': gs,
                'exit_mean':   em,   'exit_std':   es,
                'interest_mean': ir,
                'gross_margin_mean': gm_m, 'da_pct': da,
                'capex_pct': cp, 'debt_pct': dp,
                'senior_pct': sp, 'mezz_spread': mz,
                'median_irr':  float(np.median(sim.irr)),
                'mean_irr':    float(np.mean(sim.irr)),
                'p_above_20':  float((sim.irr > 0.20).mean()),
                'p5_irr':      float(np.percentile(sim.irr, 5)),
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    df.to_parquet(os.path.join(BASE, 'shap_training_data.parquet'), index=False)
    print(f"Saved {len(df)} samples")
    return df


def train_shap_model() -> xgb.XGBRegressor:
    """Train XGBoost model predicting median IRR from parameters."""
    data_path = os.path.join(BASE, 'shap_training_data.parquet')
    if not os.path.exists(data_path):
        generate_training_data()

    df = pd.read_parquet(data_path)
    print(f"Training on {len(df)} samples...")

    X = df[FEATURE_COLS].values
    y = df['median_irr'].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred) * 100
    print(f"Validation MAE: {mae:.2f}pp on median IRR")

    joblib.dump(model, os.path.join(BASE, 'shap_model.pkl'))

    # Pre-compute background dataset for SHAP (100 random training samples)
    background = X_tr[np.random.choice(len(X_tr), 100, replace=False)]
    joblib.dump(background, os.path.join(BASE, 'shap_background.pkl'))

    print("SHAP model saved.")
    return model


def compute_shap(params_dict: dict) -> SHAPResult:
    """
    Compute SHAP attribution for a specific set of deal parameters.
    params_dict keys must match FEATURE_COLS.
    """
    model      = joblib.load(os.path.join(BASE, 'shap_model.pkl'))
    background = joblib.load(os.path.join(BASE, 'shap_background.pkl'))

    x = np.array([[params_dict[f] for f in FEATURE_COLS]], dtype=np.float32)
    predicted_irr = float(model.predict(x)[0])

    # Use TreeExplainer (fast, exact for tree models)
    explainer   = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(x)[0]

    named_shap = {
        FEATURE_COLS[i]: float(shap_values[i])
        for i in range(len(FEATURE_COLS))
    }

    # Sort by absolute SHAP value
    ranked = sorted(
        [(f, named_shap[f], 'upside' if named_shap[f] > 0 else 'downside')
         for f in FEATURE_COLS],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_upside   = next((f for f, v, d in ranked if d == 'upside'), ranked[0][0])
    top_downside = next((f for f, v, d in ranked if d == 'downside'), ranked[-1][0])

    # Generate natural language explanation
    top3 = ranked[:3]
    explanation_parts = []
    for feat, val, direction in top3:
        label = FEATURE_LABELS[feat]
        pp_impact = abs(val) * 100
        if direction == 'upside':
            explanation_parts.append(
                f"{label} adds ~{pp_impact:.1f}pp to median IRR"
            )
        else:
            explanation_parts.append(
                f"{label} reduces median IRR by ~{pp_impact:.1f}pp"
            )
    explanation = (
        f"Predicted median IRR: {predicted_irr*100:.1f}%. "
        f"Key drivers: {'; '.join(explanation_parts)}."
    )

    return SHAPResult(
        predicted_irr=predicted_irr,
        shap_values=named_shap,
        ranked_drivers=ranked,
        top_upside_driver=FEATURE_LABELS[top_upside],
        top_downside_driver=FEATURE_LABELS[top_downside],
        explanation=explanation,
    )


def plot_shap_waterfall(result: SHAPResult,
                         base_value: float = 0.18) -> plt.Figure:
    """Create waterfall chart of SHAP values."""
    BG  = "#05050c"; BG2 = "#0e0e1c"
    A1  = "#6060c0"; A4  = "#40c080"; A3  = "#c06060"

    ranked = result.ranked_drivers[:8]  # top 8 drivers
    labels = [FEATURE_LABELS[f] for f, v, d in ranked]
    values = [v * 100 for f, v, d in ranked]  # convert to percentage points

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    running = base_value * 100
    for i, (label, val) in enumerate(zip(labels, values)):
        color = A4 if val > 0 else A3
        ax.barh(i, val, left=running, color=color, alpha=0.85, height=0.6,
                edgecolor=BG, linewidth=0.5)
        x_text = running + val + (0.5 if val > 0 else -0.5)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_text, i, f"{val:+.1f}pp", va='center', ha=ha,
                fontsize=9, color='#c4c4d4',
                fontfamily='monospace')
        running += val

    # Add final bar showing predicted IRR
    ax.axvline(result.predicted_irr * 100, color=A1, lw=2, linestyle='--',
               label=f'Predicted median IRR: {result.predicted_irr*100:.1f}%')
    ax.axvline(base_value * 100, color='#5a5a72', lw=1, linestyle=':',
               label=f'Base: {base_value*100:.0f}%')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color='#c4c4d4',
                        fontfamily='monospace')
    ax.set_xlabel("Impact on Median IRR (percentage points)", color='#8888a4')
    ax.set_title("SHAP Attribution — What Drives Your IRR Prediction?",
                 color='#c4c4d4', fontsize=10)
    ax.tick_params(colors='#5a5a72')
    for spine in ax.spines.values():
        spine.set_color('#2a2a42')
    ax.legend(fontsize=8, facecolor=BG2, edgecolor='#2a2a42',
              labelcolor='#c4c4d4')
    ax.grid(axis='x', color='#16162a', linewidth=0.5)

    plt.tight_layout()
    return fig


def shap_model_is_trained() -> bool:
    return all(
        os.path.exists(os.path.join(BASE, f))
        for f in ['shap_model.pkl', 'shap_background.pkl']
    )


if __name__ == '__main__':
    train_shap_model()
    result = compute_shap({
        'growth_mean': 0.05, 'growth_std': 0.03,
        'exit_mean': 10.0,   'exit_std': 1.5,
        'interest_mean': 0.065, 'gross_margin_mean': 0.40,
        'da_pct': 0.04, 'capex_pct': 0.04,
        'debt_pct': 0.60, 'senior_pct': 0.70, 'mezz_spread': 0.04,
    })
    print(result.explanation)
    print("\nRanked drivers:")
    for feat, val, direction in result.ranked_drivers:
        print(f"  {FEATURE_LABELS[feat]}: {val*100:+.2f}pp ({direction})")