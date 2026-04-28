import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, Optional
from config.settings import CONFIG

class MacroRegimeDetector:
    """
    4-state Gaussian HMM for identifying macro market states.
    Section 8.1 — Macro Regime logic.
    """
    def __init__(self, n_states: int = CONFIG.MACRO_HMM_STATES):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=self.n_states, 
            covariance_type="diag", 
            n_iter=200, 
            random_state=42
        )
        self.is_fitted = False

    def prepare_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes features for HMM input on daily bars.
        """
        df = daily_df.copy()
        # 1. 20-day Realized Vol
        df["realized_vol_20d"] = np.log(df["close"] / df["close"].shift(1)).rolling(20).std()
        
        # 2. 7-day OI Trend (normalized)
        if "oi" in df.columns:
            df["oi_trend_7d"] = df["oi"].pct_change(7)
        else:
            df["oi_trend_7d"] = 0.0
            
        # 3. 7-day Avg Funding
        if "funding_rate" in df.columns:
            df["funding_avg_7d"] = df["funding_rate"].rolling(7).mean()
        else:
            df["funding_avg_7d"] = 0.0
            
        # 4. Price vs Month POC (Placeholder as this requires monthly profile calculation)
        df["price_vs_month_poc"] = 0.0 # Will be updated if monthly profiles available
        
        return df.dropna()

    def fit(self, features: pd.DataFrame):
        """Fits the HMM on historical daily data."""
        potential_cols = ["realized_vol_20d", "oi_trend_7d", "funding_avg_7d", "price_vs_month_poc"]
        # Only use features that are not constant
        cols = [c for c in potential_cols if c in features.columns and features[c].std() > 0]
        
        if not cols:
            print("No valid features for HMM. Using a dummy state.")
            self.is_fitted = True # Dummy fit
            self.model.means_ = np.zeros((self.n_states, 0))
            self.model.covars_ = np.zeros((self.n_states, 0, 0))
            return

        X = features[cols].values
        self.model.fit(X)
        self.is_fitted = True
        self.selected_cols = cols
        print(f"Macro HMM fitted successfully using features: {cols}")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predicts states for each day."""
        if not self.is_fitted:
            raise RuntimeError("HMM model not fitted.")
        
        if hasattr(self, 'selected_cols'):
            cols = self.selected_cols
        else:
            return pd.Series(0, index=features.index)
            
        X = features[cols].values
        states = self.model.predict(X)
        return pd.Series(states, index=features.index)

def label_states(model: GaussianHMM):
    """
    Heuristic to map integer states to Section 8.1 categories:
    0=Accumulation, 1=Trending, 2=Overextended, 3=Deleveraging.
    Based on state means (vol and funding).
    """
    means = model.means_
    # Sort states by realized_vol (index 0)
    # This is a simplification; a true labeling needs more logic or manual audit.
    vol_order = np.argsort(means[:, 0]) 
    return vol_order # Low vol -> High vol
