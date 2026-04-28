import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Dict, List
from config.settings import CONFIG

OUTCOME_BUCKETS = ["strong_down", "mild_down", "flat", "mild_up", "strong_up"]

def evaluate_calibration(
    y_pred_proba: np.ndarray,
    y_actual_enc: np.ndarray,
    class_labels: List[str] = OUTCOME_BUCKETS,
    n_bins: int = 8
) -> Dict:
    """
    Section 13.1 — Mean Brier Score and ECE calculation.
    Success: Mean Brier < 0.20, ECE < 0.05.
    """
    results = {}
    print("\nCalibration Results:")
    print(f"{'Class':<15} | {'Brier':<10} | {'ECE':<10} | {'Status'}")
    print("-" * 50)
    
    briers = []
    
    for i, label in enumerate(class_labels):
        # Binary target for this class
        y_bin = (y_actual_enc == i).astype(int)
        y_prob = y_pred_proba[:, i]
        
        brier = float(brier_score_loss(y_bin, y_prob))
        briers.append(brier)
        
        # Reliability curve for ECE
        try:
            prob_true, prob_pred = calibration_curve(y_bin, y_prob, n_bins=n_bins)
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
        except Exception:
            ece = float('nan')
            
        status = "PASS" if brier < CONFIG.TARGET_BRIER_SCORE else "FAIL"
        print(f"{label:<15} | {brier:.4f}     | {ece:.4f}     | {status}")
        results[label] = {"brier": brier, "ece": ece}
        
    mean_brier = np.mean(briers)
    print("-" * 50)
    print(f"Mean Brier: {mean_brier:.4f} ({'PASS' if mean_brier < 0.20 else 'FAIL'})")
    
    return results

def confidence_deviation_check(y_pred_proba: np.ndarray, y_actual_enc: np.ndarray):
    """
    Section 13.3 — Confidence vs Actual Accuracy.
    Identifies if model has 'Edge' in specific probability buckets.
    """
    confidence = y_pred_proba.max(axis=1)
    predicted_class = y_pred_proba.argmax(axis=1)
    correct = (predicted_class == y_actual_enc)
    
    print("\nConfidence Deviation Check:")
    print(f"{'Bin':<15} | {'Pred Conf':<10} | {'Actual Acc':<10} | {'Edge'}")
    print("-" * 50)
    
    bins = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() < 20: continue
        
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        dev = abs(acc - conf)
        edge = "EDGE" if dev < 0.05 else "NO EDGE"
        
        print(f"{lo:.0%}-{hi:.0%}{' ':<9} | {conf:.1%}{' ':<6} | {acc:.1%}{' ':<6} | {edge}")

if __name__ == "__main__":
    # Dummy verification
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 5, n)
    y_prob = np.random.dirichlet(np.ones(5), n)
    
    evaluate_calibration(y_prob, y_true)
    confidence_deviation_check(y_prob, y_true)
