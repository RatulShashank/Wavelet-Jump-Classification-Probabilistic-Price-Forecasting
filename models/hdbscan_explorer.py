import hdbscan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from config.settings import CONFIG

class HDBSCANExplorer:
    """
    Research tool to explore data-driven clusters using HDBSCAN.
    Section 11.6 — HDBSCAN Exploration logic.
    """
    def __init__(self, min_cluster_size: int = 30, min_samples: int = 10):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        self.scaler = StandardScaler()

    def explore(self, fm: pd.DataFrame, label_col: str = "wavelet_label"):
        """
        Fits HDBSCAN on the training features and compares clusters to ground truth labels.
        """
        # Select input features only
        EXCLUDE = {"next_hour_return", "next_hour_outcome", "wavelet_label"}
        FEAT_COLS = [c for c in fm.columns if c not in EXCLUDE]
        
        X = fm[FEAT_COLS].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Running HDBSCAN on {X_scaled.shape[0]} samples and {X_scaled.shape[1]} features...")
        cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        fm["hdbscan_cluster"] = cluster_labels
        
        # Comparison logic
        print("\nCluster vs Wavelet Label Alignment:")
        comparison = pd.crosstab(fm["hdbscan_cluster"], fm[label_col])
        print(comparison)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_pct = (cluster_labels == -1).mean() * 100
        
        print(f"\nFound {n_clusters} clusters.")
        print(f"Noise (unclustered) samples: {noise_pct:.1f}%")
        
        return comparison

if __name__ == "__main__":
    try:
        fm = pd.read_parquet("data/feature_matrix.parquet")
        # Run on training set only (Section 11.6)
        train_fm = fm[fm.index <= CONFIG.TRAIN_END]
        
        explorer = HDBSCANExplorer()
        explorer.explore(train_fm)
    except Exception as e:
        print(f"HDBSCAN exploration failed: {e}")
