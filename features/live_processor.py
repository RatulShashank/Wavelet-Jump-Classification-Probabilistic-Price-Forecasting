import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Optional

from collector.websocket_client import BinanceWebSocketClient
from storage.live_state import LiveStateManager
from features.jump_detector import JumpDetector
from research.feature_matrix import FeatureMatrixBuilder
from models.lightgbm_classifier import JumpForecasterStack
from models.gp_classifier import UncertaintyGater
from config.settings import CONFIG

class LiveSignalProcessor:
    """
    Orchestrates real-time jump detection, feature extraction, and prediction.
    Section 2 — Live System core logic.
    """
    def __init__(self, state_mgr: LiveStateManager):
        self.state_mgr = state_mgr
        self.jump_detector = JumpDetector()
        
        # We'll use a simplified version of the builder logic for live use
        from features.wavelet import WaveletFeatureExtractor
        from features.microstructure import MicrostructureFeatureExtractor
        from features.vol_profile import VolumeProfileExtractor
        
        self.wavelet_ex = WaveletFeatureExtractor()
        self.micro_ex = MicrostructureFeatureExtractor()
        self.vol_ex = VolumeProfileExtractor()
        
        # Strategy models - these would be loaded from disk in production
        self.model_stack = JumpForecasterStack()
        self.gater = UncertaintyGater()
        
    async def handle_ws_message(self, data: Dict):
        """Dispatcher for live websocket data streams."""
        event_type = data.get("e")
        
        if event_type == "kline":
            k = data["k"]
            if k["x"]: # Bar closed
                await self.process_bar_close(k)
        
        elif event_type == "markPrice":
            # Update mark price for sizing/liquidations if needed
            self.current_mark_price = float(data["p"])

    async def process_bar_close(self, kline_raw: Dict):
        """
        Main pipeline triggered every 1 minute.
        """
        # 1. Update state
        self.state_mgr.update_kline(kline_raw)
        
        # 2. Get rolling window (last 120 mins)
        df = self.state_mgr.get_recent_klines()
        if len(df) < CONFIG.JUMP_SIGMA_WINDOW:
            print("Warming up index...")
            return

        # 3. Detect Jump on the last bar
        # We run the detector on the full window, but only care if a jump is at t=0
        all_jumps = self.jump_detector.detect_jumps(df)
        
        if not all_jumps.empty and all_jumps.index[-1] == df.index[-1]:
            print(f"\n[LIVE] JUMP DETECTED at {df.index[-1]}! Magnitude: {all_jumps['jump_score'].iloc[-1]:.2f}")
            await self.generate_signal(df, all_jumps.iloc[-1])

    async def generate_signal(self, df: pd.DataFrame, jump_details: pd.Series):
        """Builds features and calls the model for a detected jump."""
        # a. Extract Features (Pre-jump window)
        ts = df.index[-1]
        pre_window = df.iloc[-31 : -1] # 30 bars before jump
        jump_dir = jump_details["direction"]
        
        aligned_rets = np.log(pre_window["close"] / pre_window["close"].shift(1)).fillna(0).values * jump_dir
        w_feats = self.wavelet_ex.extract_features(aligned_rets)
        
        # b. Microstructure
        # In live, we might need a REST call for Funding if not streamed
        m_feats = self.micro_ex.extract_features(df.iloc[-31:], 0.01, 0.01) # placeholder funding
        
        # c. Regime
        macro_r = self.state_mgr.get_macro_regime()
        # Simplified micro regime for now
        r_feats = {"macro_regime": macro_r, "micro_regime": 0, "regime_int": macro_r * 4}
        
        # Assemble feature vector
        all_feats = {**w_feats, **m_feats, **r_feats}
        # Placeholders for missing Volume Profile/Anomaly for brevity in this live sketch
        
        # d. Predict
        # Note: In production, model.predict_proba expects a specific feature order
        feat_vector = np.array([all_feats.get(c, 0.0) for c in self.model_stack.FEAT_COLS]).reshape(1, -1)
        
        probs = self.model_stack.clf_outcome.predict_proba(feat_vector)
        pred_label = self.model_stack.le_outcome.inverse_transform(probs.argmax(axis=1))[0]
        confidence = probs.max()
        
        print(f"[SIGNAL] Predicted: {pred_label} | Conf: {confidence:.2%}")
        
        # e. Uncertainty Gate
        preds_gp, gated = self.gater.predict_with_gate(feat_vector, threshold=0.65)
        if gated[0]:
            print("[GATE] Prediction discarded due to high model uncertainty.")
        else:
            print(f"[EXECUTE] Signal ready for {pred_label} entry.")
            # Trigger Order Manager here...

async def main_live():
    state = LiveStateManager()
    processor = LiveSignalProcessor(state)
    client = BinanceWebSocketClient(processor.handle_ws_message)
    
    print("Starting Live Signal Processor...")
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main_live())
