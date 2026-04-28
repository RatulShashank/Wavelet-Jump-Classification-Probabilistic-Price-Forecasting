import pandas as pd
import numpy as np
from config.settings import CONFIG

SCAT_COLS = [
    "scat_imag_j1_2", "scat_imag_j1_3", "scat_imag_j1_4",
    "scat_imag_j1_5", "scat_imag_j1_6",
    "scat_imag_j2_3", "scat_imag_j2_4", "scat_imag_j2_5",
    "scat_imag_j2_6",
    "scat_imag_j3_4", "scat_imag_j3_5", "scat_imag_j3_6",
    "scat_imag_j4_5", "scat_imag_j4_6",
    "scat_imag_j5_6",
]

def validate_labeller_columns(fm: pd.DataFrame) -> None:
    """Section 11.1 — Bug 1 guard: ensures all required columns exist."""
    required = ["vol_buildup_ratio", "trend_alignment"] + SCAT_COLS
    missing = [c for c in required if c not in fm.columns]
    if missing:
        raise KeyError(
            f"Labeller requires columns not in feature matrix: {missing}\n"
            "This was Bug 1. Fix feature extraction/matrix assembly first."
        )

def wavelet_label(row: pd.Series) -> str:
    """
    Ground truth assigner using Section 16 thresholds.
    """
    vol_buildup = row["vol_buildup_ratio"]
    trend_align = row["trend_alignment"]
    
    # Bug 1 Fix: Mean of scatter imag columns for asymmetry
    scat_asymmetry = np.mean([row[c] for c in SCAT_COLS])
    
    # Section 11.1 Logic
    if vol_buildup > CONFIG.LABEL_ENDO_VOL_BUILDUP and scat_asymmetry < CONFIG.LABEL_ENDO_SCAT_ASYM:
        return "endogenous"
    
    if vol_buildup < CONFIG.LABEL_ANTI_VOL_BUILDUP and trend_align > CONFIG.LABEL_ANTI_TREND_ALIGN:
        return "anticipatory"
        
    return "exogenous"

def verify_label_distribution(fm: pd.DataFrame, label_col: str = "wavelet_label"):
    """Section 11.1 — Verification to prevent degenerate clusters."""
    dist = fm[label_col].value_counts()
    print(f"Label distribution:\n{dist}\n")
    if len(dist) < 2:
        raise ValueError(
            "DEGENERATE LABELS: All jumps have the same label.\n"
            "This was Bug 1. Check feature ranges and thresholds."
        )
    return dist

if __name__ == "__main__":
    # Test loading the matrix and labeling
    try:
        fm = pd.read_parquet("data/feature_matrix.parquet")
        validate_labeller_columns(fm)
        fm["wavelet_label"] = fm.apply(wavelet_label, axis=1)
        verify_label_distribution(fm)
        fm.to_parquet("data/feature_matrix.parquet")
        print("Feature matrix updated with ground-truth labels.")
    except Exception as e:
        print(f"Labeling failed: {e}")
