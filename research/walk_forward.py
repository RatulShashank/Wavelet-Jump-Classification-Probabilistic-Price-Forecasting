import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from models.lightgbm_classifier import JumpForecasterStack
from research.calibration import evaluate_calibration
from config.settings import CONFIG

class WalkForwardValidator:
    """
    Robust temporal cross-validation using TimeSeriesSplit.
    Section 12 & 17 Step 15 — Walk-Forward logic.
    """
    def __init__(self, n_splits: int = 5, gap: int = 500):
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)

    def run(self, fm: pd.DataFrame):
        """
        Runs walk-forward validation on the full dataset (Train + Val).
        Section 12 temporal split rules applied.
        """
        # Exclude Test set from walk-forward tuning
        study_fm = fm[fm.index <= CONFIG.VAL_END]
        
        results = []
        stack = JumpForecasterStack()
        
        print(f"Starting Walk-Forward Validation ({self.n_splits} splits)...")
        
        # TimeSeriesSplit works on integer indices
        indices = np.arange(len(study_fm))
        
        for i, (train_idx, val_idx) in enumerate(self.tscv.split(indices)):
            print(f"\n--- Split {i+1} ---")
            train_data = study_fm.iloc[train_idx]
            val_data = study_fm.iloc[val_idx]
            
            # Step 12: Ensure no random shuffle
            print(f"Train: {train_data.index.min()} to {train_data.index.max()}")
            print(f"Val:   {val_data.index.min()} to {val_data.index.max()}")
            
            # 1. Train Jump Type
            X_tr_j, y_tr_j = stack.prepare_data(train_data, "wavelet_label")
            X_vl_j, y_vl_j = stack.prepare_data(val_data, "wavelet_label")
            stack.train_jump_classifier(X_tr_j, y_tr_j, X_vl_j, y_vl_j)
            
            # 2. Train Outcome Forecaster
            X_tr_o, y_tr_o = stack.prepare_data(train_data, "next_hour_return")
            X_vl_o, y_vl_o = stack.prepare_data(val_data, "next_hour_return")
            X_vl_aug, y_vl_enc = stack.train_outcome_forecaster(X_tr_o, y_tr_o, X_vl_o, y_vl_o)
            
            # 3. Evaluate Calibration on this split
            y_probs = stack.clf_outcome.predict_proba(X_vl_aug)
            split_metrics = evaluate_calibration(y_probs, y_vl_enc)
            results.append(split_metrics)
            
        print("\n" + "="*50)
        print("WALK-FORWARD VALIDATION COMPLETE")
        print("="*50)
        
        return results

if __name__ == "__main__":
    try:
        fm = pd.read_parquet("data/feature_matrix.parquet")
        validator = WalkForwardValidator()
        validator.run(fm)
    except Exception as e:
        print(f"Walk-forward failed: {e}")
