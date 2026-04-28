import pandas as pd
import numpy as np
import redis
import json
from typing import List, Dict, Optional
from config.settings import CONFIG

class LiveStateManager:
    """
    Manages real-time state using Redis.
    Stores rolling windows of klines, OI, and CVD.
    Section 2 & 14 — State Management logic.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.klines_key = "live:klines"
        self.macro_regime_key = "live:macro_regime"
        self.window_size = 120 # Enough for sigma(120) and pre_jump(30)

    def update_kline(self, kline_data: Dict):
        """Append a new 1m closed bar to the rolling window."""
        # Store as serialized JSON in a Redis list
        self.r.rpush(self.klines_key, json.dumps(kline_data))
        # Trim to window_size
        self.r.ltrim(self.klines_key, -self.window_size, -1)

    def get_recent_klines(self) -> pd.DataFrame:
        """Retrieves the rolling window as a Pandas DataFrame."""
        raw_list = self.r.lrange(self.klines_key, 0, -1)
        if not raw_list:
            return pd.DataFrame()
            
        data = [json.loads(s) for s in raw_list]
        df = pd.DataFrame(data)
        # Standardize columns for features
        # Assuming kline_data came from WS: { 't': open_time, 'o': open, ... }
        df.rename(columns={
            't': 'open_time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
            'v': 'volume', 'V': 'taker_buy_base', 'q': 'quote_vol'
        }, inplace=True)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        
        # Numeric conversions
        for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
        # Re-compute Delta/CVD in-memory for the window
        df["taker_sell_base"] = df["volume"] - df["taker_buy_base"]
        df["delta"] = df["taker_buy_base"] - df["taker_sell_base"]
        df["cvd"] = df["delta"].cumsum()
        
        return df

    def set_macro_regime(self, state: int):
        self.r.set(self.macro_regime_key, state)

    def get_macro_regime(self) -> int:
        val = self.r.get(self.macro_regime_key)
        return int(val) if val is not None else 0

if __name__ == "__main__":
    # Test (requires local redis)
    try:
        mgr = LiveStateManager()
        mgr.update_kline({"t": 1234567, "c": 65000})
        print(mgr.get_recent_klines())
    except Exception as e:
        print(f"Redis not connected: {e}")
