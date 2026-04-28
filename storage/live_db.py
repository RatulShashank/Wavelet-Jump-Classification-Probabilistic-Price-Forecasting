import pandas as pd
import os
from datetime import datetime
from config.settings import CONFIG

class LiveDatabase:
    """
    Handles persistence of live klines and signals.
    Phase 2 logic: Uses Parquet/CSV for initial build, 
    designed to be replaced by TimescaleDB.
    """
    def __init__(self, storage_path: str = "data/live"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def log_signal(self, timestamp: datetime, signal_data: dict):
        """Logs trade signals to a monthly CSV."""
        filename = f"{self.storage_path}/signals_{timestamp.strftime('%Y%m')}.csv"
        df = pd.DataFrame([signal_data])
        df["timestamp"] = timestamp
        
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)

    def save_live_klines(self, df: pd.DataFrame):
        """Persists closed klines to daily parquet files."""
        if df.empty: return
        
        date_str = df.index[-1].strftime('%Y-%m-%d')
        filename = f"{self.storage_path}/klines_{date_str}.parquet"
        
        # Append logic for Parquet (Read-Modify-Write)
        if os.path.exists(filename):
            existing = pd.read_parquet(filename)
            df = pd.concat([existing, df]).drop_duplicates()
            
        df.to_parquet(filename)

if __name__ == "__main__":
    db = LiveDatabase()
    # Test logging
    db.log_signal(datetime.now(), {"signal": "strong_up", "conf": 0.85})
    print("Live database test complete.")
