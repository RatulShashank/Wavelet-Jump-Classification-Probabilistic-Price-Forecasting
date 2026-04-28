import pandas as pd
import numpy as np
from typing import Dict, Optional
from config.settings import CONFIG

class VolumeProfileExtractor:
    """
    Computes POC, VAH, and VAL for multiple lookback periods.
    Section 7 — Volume Profile Features logic.
    """
    def __init__(self, n_bins: int = CONFIG.VOL_PROFILE_BINS,
                 va_pct: float = CONFIG.VOL_PROFILE_VA_PCT):
        self.n_bins = n_bins
        self.va_pct = va_pct

    def compute_profile(self, klines: pd.DataFrame) -> Dict[str, float]:
        """
        Returns poc, vah, val for the given klines slice.
        POC = price with highest volume.
        VAH/VAL = bounds of the Value Area (70% volume).
        """
        if klines.empty:
            return {"poc": 0.0, "vah": 0.0, "val": 0.0}

        # Use typical price for volume distribution
        prices = (klines["high"] + klines["low"] + klines["close"]) / 3
        bins = np.linspace(prices.min(), prices.max(), self.n_bins + 1)
        vol_by_bin = np.zeros(self.n_bins)

        # Distribute volume across bins hit by each bar
        # Vectorized version for speed
        for _, row in klines.iterrows():
            lo, hi = row["low"], row["high"]
            vol = row["volume"]
            mask = (bins[1:] >= lo) & (bins[:-1] <= hi)
            count = mask.sum()
            if count > 0:
                vol_by_bin[mask] += vol / count

        total_vol = vol_by_bin.sum()
        if total_vol == 0:
            return {"poc": 0.0, "vah": 0.0, "val": 0.0}

        # POC
        poc_bin = vol_by_bin.argmax()
        poc = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area
        target_vol = total_vol * self.va_pct
        lo_idx = hi_idx = poc_bin
        cum_vol = vol_by_bin[poc_bin]

        while cum_vol < target_vol and (lo_idx > 0 or hi_idx < self.n_bins - 1):
            add_lo = vol_by_bin[lo_idx - 1] if lo_idx > 0 else 0
            add_hi = vol_by_bin[hi_idx + 1] if hi_idx < self.n_bins - 1 else 0
            
            if add_lo >= add_hi and lo_idx > 0:
                lo_idx -= 1
                cum_vol += add_lo
            elif hi_idx < self.n_bins - 1:
                hi_idx += 1
                cum_vol += add_hi
            else:
                break

        vah = (bins[hi_idx] + bins[hi_idx + 1]) / 2
        val = (bins[lo_idx] + bins[lo_idx + 1]) / 2
        
        # Invariant check: vah >= poc >= val
        # Handle cases where bins are very narrow
        if not (vah >= poc >= val):
            vah = max(vah, poc)
            val = min(val, poc)

        return {"poc": float(poc), "vah": float(vah), "val": float(val)}

    def extract_features(self, current_price: float, open_price: float, 
                         prev_day: Dict, prev_week: Dict, prev_month: Dict) -> Dict[str, float]:
        """
        Computes distance features and structural flags for a jump.
        Section 7.3 — Features Per Jump.
        """
        f = {}
        p = current_price
        
        # Helper for % distance
        def dist(target): return (p - target) / (p + 1e-12) * 100 if target > 0 else 0.0

        # Daily
        f["dist_prev_day_poc"] = dist(prev_day["poc"])
        f["dist_prev_day_vah"] = dist(prev_day["vah"])
        f["dist_prev_day_val"] = dist(prev_day["val"])
        f["inside_prev_day_va"] = 1.0 if prev_day["val"] <= p <= prev_day["vah"] else 0.0

        # Weekly
        f["dist_prev_week_poc"] = dist(prev_week["poc"])
        f["dist_prev_week_vah"] = dist(prev_week["vah"])
        f["dist_prev_week_val"] = dist(prev_week["val"])
        f["inside_prev_week_va"] = 1.0 if prev_week["val"] <= p <= prev_week["vah"] else 0.0

        # Monthly
        f["dist_prev_month_poc"] = dist(prev_month["poc"])
        f["dist_prev_month_vah"] = dist(prev_month["vah"])
        f["dist_prev_month_val"] = dist(prev_month["val"])

        # Structural flags
        f["at_daily_extreme"] = 1.0 if (abs(f["dist_prev_day_vah"]) < 0.2 or abs(f["dist_prev_day_val"]) < 0.2) else 0.0
        f["at_weekly_extreme"] = 1.0 if (abs(f["dist_prev_week_vah"]) < 0.3 or abs(f["dist_prev_week_val"]) < 0.3) else 0.0
        
        # Jump breaks daily VA
        breaks = 0
        if prev_day["vah"] > 0:
            if (current_price > prev_day["vah"] and open_price <= prev_day["vah"]) or \
               (current_price < prev_day["val"] and open_price >= prev_day["val"]):
                breaks = 1
        f["jump_breaks_daily_va"] = float(breaks)

        return f
