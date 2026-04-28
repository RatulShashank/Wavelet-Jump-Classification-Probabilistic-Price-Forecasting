import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Tuple
from config.settings import CONFIG

class JumpDetector:
    """
    Detects price jumps using a Gumbel-threshold statistical test.
    Section 4 — Jump Detection logic.
    """
    def __init__(self, sigma_window: int = CONFIG.JUMP_SIGMA_WINDOW,
                 threshold: float = CONFIG.JUMP_THRESHOLD,
                 gap: int = CONFIG.JUMP_CLUSTER_GAP):
        self.sigma_window = sigma_window
        self.threshold = threshold
        self.gap = gap

    def compute_seasonality(self, returns: pd.Series) -> pd.Series:
        """
        Computes intraday seasonality scalar f(t).
        f(h) = mean(|r(t)|) for all t where hour(t) == h, normalized to mean 1.0.
        """
        abs_rets = returns.abs()
        # Group by hour of day
        hour_means = abs_rets.groupby(returns.index.hour).mean()
        # Normalize so global mean of f equals 1.0
        f_scalar = hour_means / hour_means.mean()
        # Map back to the original series index
        return returns.index.map(lambda ts: f_scalar.get(ts.hour, 1.0))

    def detect_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implements Jump Score Formula from Section 4.
        x(t) = r(t) / (f(t) * sigma(t))
        """
        close = df["close"].astype(float)
        # r(t) = log return
        returns = np.log(close / close.shift(1))
        
        # sigma(t) = rolling std
        sigma = returns.rolling(window=self.sigma_window).std()
        
        # f(t) = seasonality
        f = self.compute_seasonality(returns)
        
        # Jump Score x(t)
        # Add small epsilon to denominator to avoid division by zero
        jump_score = returns / (f * sigma + 1e-12)
        
        # Detection
        is_jump = jump_score.abs() > self.threshold
        
        # Extract jump candidate rows
        jump_candidates = df[is_jump].copy()
        jump_candidates["jump_score"] = jump_score[is_jump]
        jump_candidates["direction"] = np.sign(jump_score[is_jump])
        
        # Cluster filter: discard any jump within 5 bars of previous jump
        filtered_jumps = []
        last_jump_ts = None
        
        # Convert index to list for iteration
        indices = jump_candidates.index
        for i in range(len(jump_candidates)):
            current_ts = indices[i]
            if last_jump_ts is None:
                filtered_jumps.append(jump_candidates.iloc[i])
                last_jump_ts = current_ts
            else:
                # Calculate time difference in minutes (assuming 1m bars)
                time_diff = (current_ts - last_jump_ts).total_seconds() / 60
                if time_diff > self.gap:
                    filtered_jumps.append(jump_candidates.iloc[i])
                    last_jump_ts = current_ts
        
        if not filtered_jumps:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(filtered_jumps)
        print(f"Detected {len(jump_candidates)} raw jumps. After cluster filtering: {len(result_df)}")
        return result_df

if __name__ == "__main__":
    # Test on existing parquet if available
    try:
        df = pd.read_parquet("data/btcusdt_1m.parquet")
        detector = JumpDetector()
        jumps = detector.detect_jumps(df)
        if not jumps.empty:
            print(f"First jump detected at: {jumps.index[0]}")
            print(f"Jump score: {jumps['jump_score'].iloc[0]:.2f}")
    except FileNotFoundError:
        print("Data file not found. Run collector/binance_vision.py first.")
