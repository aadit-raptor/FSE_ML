"""
Run this once to train all ML models.
Total estimated time: 25-40 minutes on CPU.
After this, all models load instantly at dashboard startup.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("="*60)
    print("TRAINING ALL ML MODELS")
    print("="*60)

    # Priority 4: Anomaly detector (fastest, <5 seconds)
    print("\n[1/5] Training anomaly detector...")
    from ml.anomaly_detector import train_detector
    train_detector()
    print("✓ Anomaly detector ready")

    # Priority 8: Multiple predictor (<10 seconds)
    print("\n[2/5] Training multiple predictor...")
    from ml.multiple_predictor import train_multiple_predictor
    train_multiple_predictor()
    print("✓ Multiple predictor ready")

    # Priority 10: Distress model (<5 seconds)
    print("\n[3/5] Training distress model...")
    from ml.distress_model import train_distress_model
    train_distress_model()
    print("✓ Distress model ready")

    # Priority 5: SHAP model (~10 minutes)
    print("\n[4/5] Training SHAP model (this takes ~10 minutes)...")
    from ml.shap_attribution import train_shap_model
    train_shap_model()
    print("✓ SHAP model ready")

    # Priority 1: Surrogate model (~30 minutes)
    print("\n[5/5] Generating surrogate training data and training (~30 minutes)...")
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
    print("You can now restart the Streamlit dashboard.")
    print("="*60)


if __name__ == '__main__':
    main()