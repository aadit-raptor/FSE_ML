"""
Macro regime detection using Hidden Markov Model on FRED data.
Trains on data from 1980-present. Classifies into 4 regimes.
Update the FRED_KEY with your free API key from fred.stlouisfed.org
"""

import numpy as np
import pandas as pd
from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(__file__)
FRED_KEY = os.environ.get('FRED_API_KEY', 'your_free_key_here')

# FRED series IDs for macro features
FRED_SERIES = {
    'gdp_growth':     'GDPC1',       # Real GDP (quarterly)
    'cpi_yoy':        'CPIAUCSL',    # CPI (monthly)
    'unemployment':   'UNRATE',      # Unemployment rate (monthly)
    'yield_curve':    'T10Y2Y',      # 10yr - 2yr spread (daily -> quarterly)
    'hy_spread':      'BAMLH0A0HYM2', # HY credit spread (daily -> quarterly)
    'fed_funds':      'FEDFUNDS',    # Fed funds rate (monthly)
    'pmi':            'NAPM',        # ISM Manufacturing PMI (monthly, ends 2022)
}


def fetch_fred_data(start: str = '1980-01-01') -> pd.DataFrame:
    """Download and preprocess macro data from FRED."""
    fred = Fred(api_key=FRED_KEY)
    dfs = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start)
            dfs[name] = s
            print(f"  Fetched {name}: {len(s)} observations")
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    # Combine and resample to quarterly
    df = pd.DataFrame(dfs)
    df = df.resample('Q').mean()

    # Compute rates of change where needed
    df['gdp_growth_qoq']  = df['gdp_growth'].pct_change(4) * 100  # YoY %
    df['cpi_yoy_chg']     = df['cpi_yoy'].pct_change(4) * 100     # YoY inflation
    df['unemployment_chg']= df['unemployment'].diff(4)              # YoY change in unemployment

    # Select final feature set (drop raw levels)
    features = [
        'gdp_growth_qoq',    # Economic growth
        'cpi_yoy_chg',       # Inflation
        'unemployment_chg',  # Labor market direction
        'yield_curve',       # Monetary policy / recession signal
        'hy_spread',         # Credit stress
        'fed_funds',         # Interest rate level
    ]

    result = df[features].dropna()
    print(f"  Final dataset: {len(result)} quarters from "
          f"{result.index[0].date()} to {result.index[-1].date()}")
    return result


def train_regime_model(df: pd.DataFrame, 
                        n_components: int = 4,
                        n_restarts: int = 20) -> tuple:
    """
    Train HMM with multiple random restarts to find global optimum.
    Returns (model, scaler, feature_df).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    best_model = None
    best_score = -np.inf

    print(f"Training HMM with {n_components} states ({n_restarts} restarts)...")

    for seed in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type='full',
                n_iter=500,
                tol=1e-6,
                random_state=seed,
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    print(f"Best log-likelihood: {best_score:.2f}")

    # Label states by economic interpretation
    # Use state means to assign economic labels:
    # - State with highest GDP growth + low HY spread = expansion
    # - State with lowest GDP growth + high HY spread = recession
    # - State with high inflation + low growth = stagflation
    # - Intermediate state = late cycle
    means = best_model.means_
    gdp_idx = 0      # index of gdp_growth_qoq
    hy_idx  = 4      # index of hy_spread
    cpi_idx = 1      # index of cpi_yoy_chg

    gdp_scores = means[:, gdp_idx]
    hy_scores  = means[:, hy_idx]
    cpi_scores = means[:, cpi_idx]

    # Score each state for each regime label
    # Expansion: high GDP, low spread
    # Recession: low GDP, high spread
    # Stagflation: low-mid GDP, high inflation
    # Late cycle: mid-high GDP, rising spread

    state_labels = {}
    ranked_gdp   = np.argsort(gdp_scores)
    ranked_spread= np.argsort(hy_scores)[::-1]

    # Recession = worst GDP + highest spread
    recession_state    = ranked_gdp[0]
    # Expansion = best GDP + lowest spread
    expansion_state    = ranked_gdp[-1]
    # Stagflation = high inflation + below-average GDP
    remaining = [s for s in range(n_components) 
                 if s not in [recession_state, expansion_state]]
    stagflation_state  = max(remaining, key=lambda s: cpi_scores[s])
    late_cycle_state   = [s for s in remaining if s != stagflation_state][0]

    state_labels = {
        expansion_state:   'bull',
        late_cycle_state:  'base',
        stagflation_state: 'stagflation',
        recession_state:   'recession',
    }

    print("\nState economic labels:")
    for state, label in state_labels.items():
        print(f"  State {state} -> {label.upper()}: "
              f"GDP={means[state,gdp_idx]:+.2f}%, "
              f"HY_spread={means[state,hy_idx]:.2f}pp, "
              f"CPI={means[state,cpi_idx]:+.2f}%")

    # Save model artifacts
    joblib.dump(best_model, os.path.join(BASE, 'hmm_model.pkl'))
    joblib.dump(scaler,     os.path.join(BASE, 'hmm_scaler.pkl'))
    joblib.dump(state_labels, os.path.join(BASE, 'hmm_labels.pkl'))
    df.to_parquet(os.path.join(BASE, 'macro_data.parquet'))

    print("Model saved.")
    return best_model, scaler, state_labels, df


def get_current_regime(model=None, scaler=None, 
                        state_labels=None, df=None) -> dict:
    """
    Classify the current macro regime.
    Loads saved model if not provided.
    """
    if model is None:
        model       = joblib.load(os.path.join(BASE, 'hmm_model.pkl'))
        scaler      = joblib.load(os.path.join(BASE, 'hmm_scaler.pkl'))
        state_labels= joblib.load(os.path.join(BASE, 'hmm_labels.pkl'))
        df          = pd.read_parquet(os.path.join(BASE, 'macro_data.parquet'))

    X = scaler.transform(df.values)
    states     = model.predict(X)
    posteriors = model.predict_proba(X)

    current_state    = int(states[-1])
    current_proba    = posteriors[-1]
    current_regime   = state_labels[current_state]
    confidence       = float(current_proba[current_state])

    # Compute label probabilities (sum over matching states)
    all_labels    = set(state_labels.values())
    label_probas  = {}
    for label in all_labels:
        states_for_label = [s for s, l in state_labels.items() if l == label]
        label_probas[label] = float(sum(current_proba[s] for s in states_for_label))

    # Historical frequency of each regime
    label_history = {}
    for label in all_labels:
        count = sum(1 for s in states if state_labels[s] == label)
        label_history[label] = count / len(states)

    # Most recent regime history (last 12 quarters)
    recent_states = [state_labels[s] for s in states[-12:]]

    # Compute regime-specific distribution adjustments
    # by comparing variable means in each regime to overall means
    overall_means = df.mean()
    regime_adjustments = {}
    for label in all_labels:
        mask = np.array([state_labels[s] == label for s in states])
        if mask.sum() > 0:
            regime_df = df.values[mask]
            regime_mean = regime_df.mean(axis=0)
            # GDP growth delta from average (for growth_mean adjustment)
            regime_adjustments[label] = {
                'gdp_delta':       float(regime_mean[0] - overall_means.iloc[0]),
                'hy_spread_mult':  float(regime_mean[4] / max(overall_means.iloc[4], 0.1)),
                'rate_mult':       float(regime_mean[5] / max(overall_means.iloc[5], 0.1)),
                'cpi_delta':       float(regime_mean[1] - overall_means.iloc[1]),
            }

    return {
        'regime':              current_regime,
        'confidence':          confidence,
        'label_probabilities': label_probas,
        'historical_frequency': label_history,
        'recent_history':      recent_states,
        'regime_adjustments':  regime_adjustments,
        'data_as_of':          str(df.index[-1].date()),
    }


def model_is_trained() -> bool:
    required = ['hmm_model.pkl', 'hmm_scaler.pkl', 
                'hmm_labels.pkl', 'macro_data.parquet']
    return all(os.path.exists(os.path.join(BASE, f)) for f in required)


if __name__ == '__main__':
    print("Fetching FRED data...")
    df = fetch_fred_data()

    print("\nTraining HMM regime model...")
    model, scaler, labels, df = train_regime_model(df)

    print("\nCurrent regime classification:")
    result = get_current_regime(model, scaler, labels, df)
    print(f"  Regime: {result['regime'].upper()}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Label probabilities: {result['label_probabilities']}")
    print(f"  Historical frequency: {result['historical_frequency']}")
    print(f"  Data as of: {result['data_as_of']}")