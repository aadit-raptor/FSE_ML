"""
Run this once to train all ML models.
Total estimated time: about 30 minutes on CPU.
After this, all models load instantly at dashboard startup.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("="*60)
    print("TRAINING ALL ML MODELS")
    print("="*60)

    # The deal risk score (PLAN.md 5.2) reads published averages: nothing to train

    # The multiple predictor (PLAN.md 5.4) reads published averages: nothing to train

    # The distress predictor (PLAN.md 5.3) reads published tables: nothing to train

    # Driver explanations (PLAN.md 5.6) rerun the simulation itself: nothing to train

    # Priority 1: Surrogate model (~30 minutes)
    print("\n[1/1] Generating surrogate training data and training (~30 minutes)...")
    from ml.surrogate.generate_data import generate
    generate(n_samples=50_000, n_per_call=1000)
    from ml.surrogate.train import train
    train()
    print("✓ Surrogate model ready")

    # Optional: Macro regime (requires FRED API key)
    fred_key = os.environ.get('FRED_API_KEY')
    if fred_key:
        print("\n[OPTIONAL] Training macro regime classifier...")
        from ml.macro_regime import fetch_fred_data, train_regime_model
        df = fetch_fred_data()
        train_regime_model(df)
        print("✓ Macro regime model ready")
    else:
        print("\n[OPTIONAL] Skipping macro regime (set FRED_API_KEY to enable)")

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("Restart the API to load the new models.")
    print("="*60)


if __name__ == '__main__':
    main()