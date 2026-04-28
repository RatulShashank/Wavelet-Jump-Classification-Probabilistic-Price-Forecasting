import pandas as pd
import numpy as np
from config.settings import CONFIG

class MicroRegimeDetector:
    """
    Rule-based detector for 1-minute market microstructure regimes.
    Section 8.2 — Micro Regime logic.
    """
    def __init__(self, efficiency_window: int = CONFIG.MICRO_EFF_WINDOW):
        self.window = efficiency_window

    def detect(self, df_slice: pd.DataFrame, cvd_slope: float, 
               cvd_divergence: float) -> int:
        """
        Input: 30-bar slice, and pre-computed CVD metrics.
        Output: micro_regime (0-3).
        """
        if len(df_slice) < self.window:
            return 0
            
        # 1. Efficiency Ratio
        net_move = abs(df_slice["close"].iloc[-1] - df_slice["close"].iloc[-self.window])
        total_path = df_slice["close"].diff().abs().iloc[-self.window+1:].sum()
        efficiency_ratio = net_move / (total_path + 1e-12)
        
        # 2. OI ROC (10-bar)
        if "oi" in df_slice.columns and len(df_slice) >= 10:
            oi_now = df_slice["oi"].iloc[-1]
            oi_prev_10 = df_slice["oi"].iloc[-10]
            oi_roc = (oi_now - oi_prev_10) / (oi_prev_10 + 1e-12) * 100
        else:
            oi_roc = 0.0

        # Assign Regime (First match wins)
        # 3 = Liquidation cascade
        if efficiency_ratio > 0.35 and oi_roc < -0.5:
            return 3
        # 1 = Trending with conviction
        elif efficiency_ratio > 0.35 and abs(cvd_slope) > 0.01: # threshold can be tuned
            return 1
        # 2 = Exhaustion (trending but CVD diverging)
        elif efficiency_ratio > 0.35 and cvd_divergence < 0.3:
            return 2
        # 0 = Balanced / ranging
        else:
            return 0

def get_composite_regime(macro: int, micro: int) -> int:
    """Calculates regime_int (0-15). Section 8.2."""
    return macro * 4 + micro
