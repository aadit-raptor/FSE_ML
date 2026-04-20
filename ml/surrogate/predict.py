"""
Inference module. Loads the trained model once at import time.
predict() runs in < 1ms, enabling real-time Streamlit updates.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from dataclasses import dataclass
from typing import Optional

BASE = os.path.dirname(__file__)

X_COLS = [
    'growth_mean', 'growth_std', 'exit_mean', 'exit_std',
    'interest_mean', 'gross_margin_mean', 'gross_margin_std',
    'da_pct', 'capex_pct', 'nwc_pct', 'debt_pct'
]


class SurrogateNet(nn.Module):
    def __init__(self, n_in=11, n_out=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, n_out),
        )
    def forward(self, x): return self.net(x)


@dataclass
class SurrogatePrediction:
    irr_p5:      float
    irr_p10:     float
    irr_p25:     float
    irr_p50:     float
    irr_p75:     float
    irr_p90:     float
    irr_p95:     float
    irr_mean:    float
    irr_std:     float
    p_above_20:  float
    p_wipeout:   float


class SurrogatePredictor:
    _instance: Optional['SurrogatePredictor'] = None

    def __init__(self):
        model_path   = os.path.join(BASE, 'model.pt')
        scaler_X_path = os.path.join(BASE, 'scaler_X.pkl')
        scaler_y_path = os.path.join(BASE, 'scaler_y.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
            raise FileNotFoundError(
                "Surrogate model not trained yet. "
                "Run: python -m ml.surrogate.generate_data && "
                "python -m ml.surrogate.train"
            )

        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.model = SurrogateNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()
        self._available = True

    @classmethod
    def get_instance(cls) -> Optional['SurrogatePredictor']:
        if cls._instance is None:
            try:
                cls._instance = cls()
            except FileNotFoundError:
                return None
        return cls._instance

    def predict(self,
                growth_mean: float, growth_std: float,
                exit_mean: float,   exit_std: float,
                interest_mean: float, gross_margin_mean: float,
                gross_margin_std: float, da_pct: float,
                capex_pct: float,   nwc_pct: float,
                debt_pct: float) -> SurrogatePrediction:

        x = np.array([[
            growth_mean, growth_std, exit_mean, exit_std,
            interest_mean, gross_margin_mean, gross_margin_std,
            da_pct, capex_pct, nwc_pct, debt_pct
        ]], dtype=np.float32)

        x_scaled = self.scaler_X.transform(x)
        x_t = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            y_scaled = self.model(x_t).numpy()

        y = self.scaler_y.inverse_transform(y_scaled)[0]

        return SurrogatePrediction(
            irr_p5=float(y[0]),   irr_p10=float(y[1]),
            irr_p25=float(y[2]),  irr_p50=float(y[3]),
            irr_p75=float(y[4]),  irr_p90=float(y[5]),
            irr_p95=float(y[6]),  irr_mean=float(y[7]),
            irr_std=float(y[8]),  p_above_20=float(np.clip(y[9], 0, 1)),
            p_wipeout=float(np.clip(y[10], 0, 1)),
        )