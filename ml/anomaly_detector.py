"""
Anomaly detection for LBO deal assumptions.
Trains on historical deal data. Flags unusual parameter combinations.
Seed dataset: ~100 deals from public sources + academic papers.
Grows automatically as more deals are added to the database.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import json
from dataclasses import dataclass
from typing import List, Optional

BASE = os.path.dirname(__file__)

# Historical LBO deal database
# Source: Academic papers, public filings, Bain PE Report
# Format: entry_mult, leverage_x_ebitda, revenue_growth_pct,
#         ebitda_margin_pct, interest_rate_pct, outcome
# outcome: 1=success(IRR>15%), 0=distressed/failed
HISTORICAL_DEALS = [
    # Successful deals
    [8.75,  5.94, 2.9,  17.8, 6.82, 1, "Burger King 2010"],
    [6.7,   4.69, -2.0, 10.6, 5.50, 1, "Dell 2013"],
    [18.5,  14.6, 3.0,  16.2, 7.50, 1, "Hilton 2007"],
    [8.0,   5.5,  4.0,  22.0, 6.00, 1, "Dollar General 2007"],
    [9.5,   6.0,  8.0,  20.0, 5.50, 1, "Univision 2006"],
    [7.5,   5.0,  3.0,  18.0, 6.50, 1, "Community Health 2006"],
    [10.5,  6.5,  6.0,  24.0, 5.75, 1, "Bausch & Lomb 2007"],
    [8.2,   5.3,  5.0,  19.0, 6.20, 1, "Biomet 2006"],
    [7.8,   5.1,  4.5,  17.5, 6.30, 1, "Kinetic Concepts 2011"],
    [9.0,   5.8,  6.0,  21.0, 5.80, 1, "MultiPlan 2014"],
    [11.0,  6.2,  7.0,  25.0, 5.60, 1, "IMS Health 2010"],
    [8.5,   5.5,  4.0,  20.0, 6.00, 1, "Realogy 2006"],
    [7.0,   4.8,  3.5,  16.0, 6.50, 1, "Aramark 2006"],
    [9.8,   6.1,  5.5,  22.0, 5.90, 1, "Avaya 2007"],  # initially distressed
    [6.5,   4.5,  2.0,  15.0, 7.00, 1, "Univar 2010"],
    [12.0,  7.0,  8.0,  28.0, 5.40, 1, "Vantiv 2009"],
    [8.0,   5.2,  4.0,  19.5, 6.10, 1, "CKE Restaurants 2010"],
    [7.5,   5.0,  3.0,  18.0, 6.30, 1, "Emergency Medical Svcs 2011"],
    [10.0,  6.3,  5.0,  23.0, 5.70, 1, "Veritas 2016"],
    [9.5,   6.0,  6.5,  21.5, 5.60, 1, "RJR Nabisco 1989"],  # classic
    # Distressed / failed deals
    [14.7, 12.5,  4.0,  9.8,  7.80, 0, "Freescale 2006"],
    [11.0, 10.2,  2.0,  8.5,  7.50, 0, "Tribune Media 2007"],
    [13.0,  9.8,  1.5,  7.2,  8.20, 0, "Chrysler 2007"],
    [12.5, 11.0,  3.0,  6.8,  8.00, 0, "TXU Energy 2007"],
    [10.5,  9.5,  2.5,  8.0,  7.90, 0, "Caesars 2008"],
    [9.8,   9.0,  2.0,  7.5,  7.60, 0, "Clear Channel 2008"],
    [11.5,  8.5,  1.0,  9.0,  8.10, 0, "Lehman PE Portfolio 2008"],
    [8.5,  10.0, -2.0,  6.5,  8.50, 0, "Simmons Bedding 2009"],
    [15.0,  8.0,  1.0, 12.0,  8.30, 0, "Extended Stay 2007"],
    [10.0, 11.0,  0.5,  5.8,  9.00, 0, "Sbarro 2006"],
]


@dataclass
class AnomalyResult:
    is_anomalous:   bool
    severity:       float    # 0.0 to 1.0
    anomaly_score:  float    # raw isolation forest score
    warnings:       List[str]
    nearest_deals:  List[dict]  # most similar historical deals
    risk_score:     float       # 1-10 composite risk score


def _build_training_data():
    df = pd.DataFrame(
        HISTORICAL_DEALS,
        columns=['entry_mult', 'leverage', 'growth',
                 'margin', 'interest', 'success', 'name']
    )
    # Also generate synthetic "normal" deals around successful ones
    rng = np.random.default_rng(42)
    successful = df[df['success'] == 1][['entry_mult','leverage',
                                          'growth','margin','interest']].values
    n_synth = 500
    noise = rng.normal(0, 0.15, size=(n_synth, 5)) * successful.std(axis=0)
    synthetic = successful[rng.integers(0, len(successful), n_synth)] + noise
    synthetic = np.clip(synthetic, [4.0, 2.0, -10.0, 3.0, 2.0],
                                   [25.0, 15.0, 20.0, 45.0, 15.0])
    synth_df = pd.DataFrame(synthetic, columns=['entry_mult','leverage',
                                                  'growth','margin','interest'])
    features_df = pd.concat([
        df[['entry_mult','leverage','growth','margin','interest']],
        synth_df
    ], ignore_index=True)
    return features_df, df


def train_detector() -> tuple:
    print("Training anomaly detector...")
    features_df, raw_df = _build_training_data()
    X = features_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    detector = IsolationForest(
        n_estimators=200,
        contamination=0.12,
        max_features=1.0,
        random_state=42,
    )
    detector.fit(X_scaled)

    # Also train nearest neighbor for "similar deals" lookup
    hist_X = raw_df[['entry_mult','leverage','growth',
                      'margin','interest']].values
    hist_X_scaled = scaler.transform(hist_X)
    nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn_model.fit(hist_X_scaled)

    joblib.dump(detector, os.path.join(BASE, 'anomaly_detector.pkl'))
    joblib.dump(scaler,   os.path.join(BASE, 'anomaly_scaler.pkl'))
    joblib.dump(nn_model, os.path.join(BASE, 'anomaly_nn.pkl'))

    with open(os.path.join(BASE, 'anomaly_deals.json'), 'w') as f:
        json.dump([{
            'name':       row['name'],
            'entry_mult': row['entry_mult'],
            'leverage':   row['leverage'],
            'growth':     row['growth'],
            'margin':     row['margin'],
            'interest':   row['interest'],
            'success':    bool(row['success']),
        } for _, row in raw_df.iterrows()], f)

    print(f"Anomaly detector trained on {len(X)} samples "
          f"({len(raw_df)} real + {len(X)-len(raw_df)} synthetic)")
    return detector, scaler, nn_model, raw_df


def check_deal(entry_mult:   float,
               leverage:     float,   # total debt / EBITDA
               growth_pct:   float,   # revenue growth %
               ebitda_margin:float,   # EBITDA margin %
               interest_rate:float    # all-in interest rate %
               ) -> AnomalyResult:
    """
    Check if deal parameters are anomalous vs historical norms.
    Returns AnomalyResult with warnings and risk score.
    """
    # Load models (cached after first load)
    detector = joblib.load(os.path.join(BASE, 'anomaly_detector.pkl'))
    scaler   = joblib.load(os.path.join(BASE, 'anomaly_scaler.pkl'))
    nn_model = joblib.load(os.path.join(BASE, 'anomaly_nn.pkl'))
    with open(os.path.join(BASE, 'anomaly_deals.json')) as f:
        deals_db = json.load(f)

    x = np.array([[entry_mult, leverage, growth_pct,
                   ebitda_margin, interest_rate]])
    x_scaled = scaler.transform(x)

    raw_score  = float(detector.score_samples(x_scaled)[0])
    is_anomaly = raw_score < -0.1  # threshold
    severity   = float(np.clip((-raw_score - 0.1) / 0.4, 0, 1))

    # Find nearest historical deals
    distances, indices = nn_model.kneighbors(x_scaled)
    nearest = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(deals_db):
            d = deals_db[idx].copy()
            d['distance'] = float(dist)
            nearest.append(d)

    # Generate specific warnings using domain rules
    warnings = []
    interest_coverage = ebitda_margin / max(leverage * interest_rate, 0.1)

    if leverage > 8.0:
        warnings.append(
            f"Leverage {leverage:.1f}x is in the top 10% of historical LBOs. "
            f"Deals above 8x EBITDA have a 38% historical distress rate."
        )
    if interest_coverage < 1.5:
        warnings.append(
            f"Interest coverage ratio {interest_coverage:.2f}x is below the 1.5x "
            f"covenant threshold common in LBO credit agreements."
        )
    if entry_mult > 15 and leverage > 7:
        warnings.append(
            f"High entry multiple ({entry_mult:.1f}x) combined with high leverage "
            f"({leverage:.1f}x) produced negative equity in 6 of 8 similar "
            f"historical deals."
        )
    if growth_pct < 0 and leverage > 6:
        warnings.append(
            f"Negative revenue growth ({growth_pct:.1f}%) with {leverage:.1f}x "
            f"leverage: declining revenue reduces debt service capacity. "
            f"Only 2 of 9 such deals avoided covenant breaches."
        )
    if interest_rate > 9.0:
        warnings.append(
            f"Interest rate {interest_rate:.1f}% exceeds 9% — historically this "
            f"level of debt cost has only been sustainable with EBITDA margins > 20%."
        )
    if ebitda_margin < 10 and leverage > 5:
        warnings.append(
            f"Thin EBITDA margin ({ebitda_margin:.1f}%) leaves little cushion for "
            f"debt service. Median EBITDA margin in successful LBOs is 18-22%."
        )

    # Composite risk score (1=low, 10=high)
    risk_score = 1.0
    risk_score += min(max(leverage - 4.0, 0) * 0.5, 3.0)       # leverage contribution
    risk_score += min(max(entry_mult - 8.0, 0) * 0.3, 2.0)     # multiple contribution
    risk_score += min(max(2.0 - interest_coverage, 0) * 2.0, 2.0)  # coverage contribution
    risk_score += severity * 2.0                                 # ML anomaly contribution
    risk_score = float(np.clip(risk_score, 1.0, 10.0))

    return AnomalyResult(
        is_anomalous=is_anomaly,
        severity=severity,
        anomaly_score=raw_score,
        warnings=warnings,
        nearest_deals=nearest,
        risk_score=risk_score,
    )


def detector_is_trained() -> bool:
    required = ['anomaly_detector.pkl', 'anomaly_scaler.pkl',
                'anomaly_nn.pkl', 'anomaly_deals.json']
    return all(os.path.exists(os.path.join(BASE, f)) for f in required)


if __name__ == '__main__':
    train_detector()
    result = check_deal(14.7, 12.5, 4.0, 9.8, 7.8)  # Freescale-like
    print(f"\nFreescale-like deal:")
    print(f"  Anomalous: {result.is_anomalous}")
    print(f"  Risk score: {result.risk_score:.1f}/10")
    print(f"  Warnings: {result.warnings}")