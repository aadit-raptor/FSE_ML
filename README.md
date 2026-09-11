---
title: Simulation Model
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---
# Simulation Model — LBO analysis platform

Deal wizard, Monte Carlo simulation, backtesting against historical deals,
3-statement company forecasting (with SEC EDGAR autofill) and settings.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Optional ML layer (`ml/`)

The app runs without it. Each feature is behind a guarded import and appears
only when its dependencies and trained model files are present:

| Feature | Page | Enable |
|---|---|---|
| Deal risk score (anomaly detector) | Deal inputs | `pip install -r requirements-ml.txt` (model files are committed) |
| Live IRR sliders (surrogate network) | Monte Carlo | install as above, then `python -m ml.surrogate.generate_data && python -m ml.surrogate.train` if the model files are missing |
| Macro regime detection | Monte Carlo | install as above, set `FRED_API_KEY`, then `python -m ml.macro_regime` |

The surrogate is trained on a fixed deal (10x entry, 5-year hold, default
fees and operating terms — see `TRAINING_FIXED` in
`ml/surrogate/generate_data.py`); the page warns when a deal differs, and
RUN SIMULATION is always exact. `requirements-ml.txt` pins scikit-learn to
1.8 because the committed anomaly-detector models were saved with 1.8.0.
