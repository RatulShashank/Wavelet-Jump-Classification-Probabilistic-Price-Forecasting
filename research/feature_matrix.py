import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

from config.settings import CONFIG
from features.jump_detector import JumpDetector
from features.wavelet import WaveletFeatureExtractor
from features.microstructure import MicrostructureFeatureExtractor
from features.vol_profile import VolumeProfileExtractor
from regime.macro_hmm import MacroRegimeDetector
from regime.micro_regime import MicroRegimeDetector, get_composite_regime

class FeatureMatrixBuilder:
    """
    Assembles the canonical feature matrix from raw data.
    Section 9 & 10 — Complete Feature Matrix logic.
    """
    def __init__(self):
        self.jump_detector = JumpDetector()
        self.wavelet_extractor = WaveletFeatureExtractor()
        self.micro_extractor = MicrostructureFeatureExtractor()
        self.vol_extractor = VolumeProfileExtractor()
        self.micro_regime_detector = MicroRegimeDetector()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k_path = "data/btcusdt_1m.parquet"
        f_path = "data/btcusdt_funding.parquet"
        if not os.path.exists(k_path) or not os.path.exists(f_path):
            raise FileNotFoundError("Raw parquet files missing. Run collector first.")
        
        return pd.read_parquet(k_path), pd.read_parquet(f_path)

    def build(self) -> pd.DataFrame:
        df, funding_df = self.load_data()
        
        # 1. Detect Jumps
        print("Detecting jumps...")
        jumps = self.jump_detector.detect_jumps(df)
        if jumps.empty:
            print("No jumps detected.")
            return pd.DataFrame()
            
        # 2. Pre-calculate Macro Regime (Simplified for now - using fit on full training period)
        # In a real system, this would be walk-forward.
        print("Computing macro regimes...")
        macro_detector = MacroRegimeDetector()
        # Resample to daily
        daily_df = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        # Add funding to daily
        daily_df['funding_rate'] = funding_df.resample('1D').mean()
        # Prepare and Fit
        hmm_feats = macro_detector.prepare_features(daily_df)
        macro_detector.fit(hmm_feats)
        daily_regimes = macro_detector.predict(hmm_feats)
        
        # 3. Pre-calculate Volume Profiles (Expensive)
        # We need Daily, Weekly, Monthly snapshots.
        # This is a placeholder for a more optimized caching implementation.
        print("Initializing volume profiles...")
        # For this version, we'll compute profiles on demand or assume 0 for speed
        # A Production version would pre-calculate these at period boundaries.
        
        # 4. Assemble Features per Jump
        feature_rows = []
        for ts, jump_row in tqdm(jumps.iterrows(), total=len(jumps), desc="Assembling Matrix"):
            # Check for look-ahead: all features use data up to ts (inclusive)
            idx = df.index.get_loc(ts)
            if idx < 30: continue
            
            # Windows
            # Pre-jump: (ts-30, ts] - 30 bars
            pre_window = df.iloc[idx-30 : idx]
            # Jump bar: ts
            jump_bar_slice = df.iloc[idx-30 : idx+1]
            
            # a. Wavelet Features
            # Section 4.3: x_bar is jump-aligned (flipped if jump is down)
            jump_dir = jump_row["direction"]
            aligned_returns = np.log(pre_window["close"] / pre_window["close"].shift(1)).fillna(0).values * jump_dir
            w_feats = self.wavelet_extractor.extract_features(aligned_returns)
            
            # b. Microstructure Features
            f_val = funding_df.asof(ts)["funding_rate"] if not funding_df.empty else 0.0
            f_8h = funding_df.asof(ts - pd.Timedelta(hours=8))["funding_rate"] if not funding_df.empty else 0.0
            m_feats = self.micro_extractor.extract_features(jump_bar_slice, f_val, f_8h)
            
            # c. Regime Features
            macro_val = daily_regimes.asof(ts)
            macro_r = int(macro_val) if pd.notna(macro_val) else 0
            micro_r = self.micro_regime_detector.detect(jump_bar_slice, m_feats["cvd_slope_15min"], m_feats["cvd_price_divergence"])
            r_feats = {
                "macro_regime": float(macro_r),
                "micro_regime": float(micro_r),
                "regime_int": float(get_composite_regime(macro_r, micro_r))
            }
            
            # d. Volume Profile (Placeholder defaults)
            v_feats = {c: 0.0 for c in [
                "dist_prev_day_poc", "dist_prev_day_vah", "dist_prev_day_val", "inside_prev_day_va",
                "dist_prev_week_poc", "dist_prev_week_vah", "dist_prev_week_val", "inside_prev_week_va",
                "dist_prev_month_poc", "dist_prev_month_vah", "dist_prev_month_val",
                "at_daily_extreme", "at_weekly_extreme", "jump_breaks_daily_va"
            ]}
            
            # e. Outcome / Target (Section 4.5)
            # 1-hour return after jump
            if idx + 60 < len(df):
                post_close = df["close"].iloc[idx+60]
                jump_close = df["close"].iloc[idx]
                target_ret = np.log(post_close / jump_close) * 100
            else:
                target_ret = np.nan
                
            # Combine all
            all_feats = {**w_feats, **m_feats, **r_feats, **v_feats}
            all_feats["next_hour_return"] = target_ret
            all_feats["timestamp"] = ts
            feature_rows.append(all_feats)
        
        fm = pd.DataFrame(feature_rows).set_index("timestamp")
        
        # 5. Anomaly Score (Isolation Forest)
        from sklearn.ensemble import IsolationForest
        print("Computing anomaly scores...")
        iso = IsolationForest(contamination=0.05, random_state=42)
        # Use microstructure features for anomaly detection
        m_cols = ["cvd_slope_15min", "cvd_price_divergence", "jump_bar_delta_ratio", "cvd_acceleration"]
        fm["isolation_score"] = iso.fit_predict(fm[m_cols].fillna(0))
        
        # 6. Audit (Section 10)
        print(f"Matrix Audit: {fm.shape[1]} columns, {len(fm)} rows.")
        missing_targets = fm["next_hour_return"].isna().sum()
        if missing_targets > 0:
            print(f"Dropping {missing_targets} jumps with insufficient post-jump data.")
            fm = fm.dropna(subset=["next_hour_return"])
            
        fm.to_parquet("data/feature_matrix.parquet")
        return fm

if __name__ == "__main__":
    builder = FeatureMatrixBuilder()
    builder.build()
