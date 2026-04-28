import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict
from config.settings import CONFIG

class JumpForecasterStack:
    """
    Dual LightGBM stack: Jump Type Classifier -> Outcome Forecaster.
    Section 11.2 and 11.3 logic.
    """
    def __init__(self):
        self.le_jump = LabelEncoder()
        self.le_outcome = LabelEncoder()
        
        self.jump_params = {
            "objective": "multiclass",
            "num_class": 3,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "class_weight": "balanced",
            "n_estimators": 500,
            "random_state": 42,
            "verbose": -1
        }
        
        self.outcome_params = {
            "objective": "multiclass",
            "num_class": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "class_weight": "balanced",
            "n_estimators": 500,
            "random_state": 42,
            "verbose": -1
        }
        
        self.clf_jump = lgb.LGBMClassifier(**self.jump_params)
        self.clf_outcome = lgb.LGBMClassifier(**self.outcome_params)

    def prepare_data(self, fm: pd.DataFrame, target_col: str):
        # Identify feature columns (everything except targets)
        EXCLUDE = {"next_hour_return", "next_hour_outcome", "wavelet_label"}
        
        # Special case: if predicting wavelet_label, exclude the features used to create it
        if target_col == "wavelet_label":
            LABEL_FEATS = ["vol_buildup_ratio", "trend_alignment"] + [
                f"scat_imag_j{j1}_{j2}" for j1 in range(1, 7) for j2 in range(j1 + 1, 7)
            ]
            EXCLUDE = EXCLUDE.union(set(LABEL_FEATS))
            
        FEAT_COLS = [c for c in fm.columns if c not in EXCLUDE]
        
        X = fm[FEAT_COLS].fillna(0)
        y = fm[target_col]
        return X, y

    def train_jump_classifier(self, X_train, y_train, X_val, y_val):
        """Trains the 3-class jump type classifier."""
        y_train_enc = self.le_jump.fit_transform(y_train)
        y_val_enc = self.le_jump.transform(y_val)
        
        self.clf_jump.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            callbacks=[lgb.early_stopping(30)]
        )
        
        val_acc = accuracy_score(y_val_enc, self.clf_jump.predict(X_val))
        print(f"Jump Classifier Val Accuracy: {val_acc:.3f}")
        # Bug 1 Guard: Section 11.2
        if val_acc > 0.99:
            raise ValueError("Suspiciously perfect accuracy. Degenerate labels (Bug 1)?")

    def train_outcome_forecaster(self, X_train, y_train, X_val, y_val):
        """Trains the 5-class outcome forecaster with stacked probabilities."""
        # 1. Map outcomes to labels (Section 4.5)
        def get_bucket(r):
            if r >= CONFIG.BUCKET_STRONG_UP: return "strong_up"
            if r >= CONFIG.BUCKET_MILD_UP: return "mild_up"
            if r >= CONFIG.BUCKET_FLAT_LO: return "flat"
            if r >= CONFIG.BUCKET_MILD_DOWN: return "mild_down"
            return "strong_down"
        
        y_train_label = y_train.apply(get_bucket)
        y_val_label = y_val.apply(get_bucket)
        
        y_train_enc = self.le_outcome.fit_transform(y_train_label)
        y_val_enc = self.le_outcome.transform(y_val_label)
        
        # 2. Stack Jump Type Probabilities
        # IMPORTANT: Jump classifier expects the SAME feature set it was trained on
        # We must strip the leaking features from the outcome data for the jump prediction
        LABEL_FEATS = ["vol_buildup_ratio", "trend_alignment"] + [
            f"scat_imag_j{j1}_{j2}" for j1 in range(1, 7) for j2 in range(j1 + 1, 7)
        ]
        X_train_jump = X_train.drop(columns=[c for c in LABEL_FEATS if c in X_train.columns])
        X_val_jump = X_val.drop(columns=[c for c in LABEL_FEATS if c in X_val.columns])
        
        tp_train = self.clf_jump.predict_proba(X_train_jump)
        tp_val = self.clf_jump.predict_proba(X_val_jump)
        
        X_train_aug = np.hstack([X_train.values, tp_train])
        X_val_aug = np.hstack([X_val.values, tp_val])
        
        # 3. Fit
        self.clf_outcome.fit(
            X_train_aug, y_train_enc,
            eval_set=[(X_val_aug, y_val_enc)],
            callbacks=[lgb.early_stopping(30)]
        )
        
        return X_val_aug, y_val_enc

if __name__ == "__main__":
    # Complete workflow placeholder
    try:
        fm = pd.read_parquet("data/feature_matrix.parquet")
        # Split logic (Section 12)
        train_fm = fm[fm.index <= CONFIG.TRAIN_END]
        val_fm = fm[(fm.index >= CONFIG.VAL_START) & (fm.index <= CONFIG.VAL_END)]
        
        stack = JumpForecasterStack()
        
        # Train Jump Type
        X_tr, y_tr = stack.prepare_data(train_fm, "wavelet_label")
        X_vl, y_vl = stack.prepare_data(val_fm, "wavelet_label")
        stack.train_jump_classifier(X_tr, y_tr, X_vl, y_vl)
        
        # Train Outcome
        X_tr_o, y_tr_o = stack.prepare_data(train_fm, "next_hour_return")
        X_vl_o, y_vl_o = stack.prepare_data(val_fm, "next_hour_return")
        stack.train_outcome_forecaster(X_tr_o, y_tr_o, X_vl_o, y_vl_o)
        
        print("ML Stack training complete.")
    except Exception as e:
        print(f"Training failed: {e}")
