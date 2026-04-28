import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict
from config.settings import CONFIG

class MicrostructureFeatureExtractor:
    """
    Extracts CVD, Delta, OI, Funding, and Session features.
    Section 6 — Market Microstructure Features logic.
    """
    def __init__(self, 
                 cvd_window: int = CONFIG.CVD_SLOPE_WINDOW,
                 oi_window: int = 30):
        self.cvd_window = cvd_window
        self.oi_window = oi_window

    def extract_features(self, df_slice: pd.DataFrame, funding_val: float, 
                         funding_8h_prev: float) -> Dict[str, float]:
        """
        Input: 
            df_slice: DataFrame slice up to and including jump bar t=0.
            funding_val: current funding rate.
            funding_8h_prev: funding rate 8 hours ago.
        """
        # Ensure we have enough data
        window_len = len(df_slice)
        jump_bar = df_slice.iloc[-1]
        
        features = {}

        # 6.1 — CVD and Delta
        cvd_data = df_slice["cvd"].iloc[-self.cvd_window:]
        price_data = df_slice["close"].iloc[-self.cvd_window:]
        
        # CVD Slope (15min)
        if len(cvd_data) >= 2:
            features["cvd_slope_15min"] = linregress(range(len(cvd_data)), cvd_data.values).slope
            # CVD/Price Correlation (Divergence)
            features["cvd_price_divergence"] = float(np.corrcoef(price_data.values, cvd_data.values)[0, 1])
        else:
            features["cvd_slope_15min"] = 0.0
            features["cvd_price_divergence"] = 0.0

        # Jump Bar Delta Ratio
        features["jump_bar_delta_ratio"] = float(jump_bar["delta"] / (jump_bar["volume"] + 1e-12))

        # CVD Acceleration (last 5 vs previous 5)
        if window_len >= 10:
            cvd_5a = linregress(range(5), df_slice["cvd"].iloc[-5:].values).slope
            cvd_5b = linregress(range(5), df_slice["cvd"].iloc[-10:-5].values).slope
            features["cvd_acceleration"] = float(cvd_5a - cvd_5b)
        else:
            features["cvd_acceleration"] = 0.0

        # 6.2 — Open Interest (Assumes 'oi' column exists in input df_slice)
        if "oi" in df_slice.columns:
            oi_now = df_slice["oi"].iloc[-1]
            oi_prev = df_slice["oi"].iloc[-2] if window_len > 1 else oi_now
            oi_30 = df_slice["oi"].iloc[-min(window_len, 31):]
            
            features["oi_change_on_jump_pct"] = float((oi_now - oi_prev) / (oi_prev + 1e-12) * 100)
            
            if len(oi_30) >= 2:
                features["oi_trend_30min"] = linregress(range(len(oi_30)), oi_30.values).slope
                
                # OI/Volume Divergence flag
                oi_slope_norm = features["oi_trend_30min"] / (oi_30.mean() + 1e-12)
                price_30 = df_slice["close"].iloc[-len(oi_30):]
                price_slope_norm = linregress(range(len(price_30)), price_30.values).slope / (price_30.mean() + 1e-12)
                features["oi_vol_divergence"] = 1.0 if abs(oi_slope_norm) < 0.1 and abs(price_slope_norm) > 0.3 else 0.0
            else:
                features["oi_trend_30min"] = 0.0
                features["oi_vol_divergence"] = 0.0
        else:
            # Fallback if OI not available (Phase 1 might only have klines)
            features["oi_change_on_jump_pct"] = 0.0
            features["oi_trend_30min"] = 0.0
            features["oi_vol_divergence"] = 0.0

        # 6.3 — Funding Rate
        features["funding_rate"] = float(funding_val)
        features["funding_8h_change"] = float(funding_val - funding_8h_prev)

        # 6.4 — Liquidations (Default to 0 logic per Section 6.4)
        features["liq_volume_1min"] = 0.0
        features["liq_count_1min"] = 0.0
        features["liq_side"] = 0.0
        features["liq_preceded_jump"] = 0.0

        # 6.5 — Session Context
        ts = df_slice.index[-1] # UTC
        features["hour_of_day"] = float(ts.hour)
        features["day_of_week"] = float(ts.dayofweek)
        # Session: 0=Asia (0-8), 1=London (8-16), 2=US (16-24)
        features["session"] = float(0 if ts.hour < 8 else (1 if ts.hour < 16 else 2))
        features["time_in_session_pct"] = float((ts.hour % 8 + ts.minute / 60) / 8)

        return features
