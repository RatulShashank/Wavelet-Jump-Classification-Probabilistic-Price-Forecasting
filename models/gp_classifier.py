import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import LabelEncoder
from config.settings import CONFIG

class UncertaintyGater:
    """
    Gaussian Process Classifier for uncertainty-calibrated outcome classification.
    Section 11.4 — Gaussian Process logic.
    """
    def __init__(self, n_subsamples: int = 2000):
        self.n_subsamples = n_subsamples
        # Section 11.4 Kernel
        self.kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gpc = GaussianProcessClassifier(
            kernel=self.kernel,
            n_restarts_optimizer=3,
            random_state=42
        )
        self.le = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        GP does not scale well to 8,000 samples. 
        Train on a random subsample (max 2000).
        """
        self.le.fit(y)
        y_enc = self.le.transform(y)
        
        n_available = len(X)
        if n_available > self.n_subsamples:
            idx = np.random.RandomState(42).choice(n_available, size=self.n_subsamples, replace=False)
            X_sub = X[idx]
            y_sub = y_enc[idx]
        else:
            X_sub = X
            y_sub = y_enc
            
        print(f"Fitting Gaussian Process on {len(X_sub)} samples...")
        self.gpc.fit(X_sub, y_sub)
        self.is_fitted = True
        print("GP fitting complete.")

    def predict_with_gate(self, X: np.ndarray, threshold: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
        """
        Signal gating rule: if max(gpc.predict_proba(x)) < 0.65 -> skip trade.
        Returns (predicted_class, is_gated)
        """
        if not self.is_fitted:
            raise RuntimeError("GP Classifier not fitted.")
            
        probs = self.gpc.predict_proba(X)
        max_probs = probs.max(axis=1)
        preds = self.le.inverse_transform(probs.argmax(axis=1))
        
        # gated = True if confidence is below threshold
        is_gated = max_probs < threshold
        return preds, is_gated

if __name__ == "__main__":
    # Test placeholder
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randint(0, 5, 100)
    
    gater = UncertaintyGater(n_subsamples=50)
    gater.fit(X_dummy, y_dummy)
    preds, gated = gater.predict_with_gate(X_dummy[:5])
    print(f"Predictions: {preds}")
    print(f"Gated (Skipped): {gated}")
