import pandas as pd
import numpy as np
import requests
import os
import zipfile
import io
from tqdm import tqdm
from datetime import datetime
from config.settings import CONFIG

class BinanceVisionDownloader:
    """
    Downloads historical klines and funding rates from Binance Vision.
    Implements Bug 1 and Bug 2 fixes from AGENT_CONTEXT.md.
    """
    def __init__(self, symbol: str = CONFIG.SYMBOL):
        self.symbol = symbol
        self.base_url = f"{CONFIG.BINANCE_VISION_BASE}/data/futures/um/monthly"

    def download_monthly_klines(self, year: int, month: int) -> pd.DataFrame:
        url = f"{self.base_url}/klines/{self.symbol}/1m/{self.symbol}-1m-{year}-{month:02d}.zip"
        print(f"Downloading klines: {url}")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"Unavailable: {url} (Status {response.status_code})")
                return None
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Binance Vision klines have a header
                    df = pd.read_csv(f, header=0, names=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_vol", "n_trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
            
            # Bug 2 Fix: Always use unit='ms' and utc=True
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            return df.set_index("open_time")
        except Exception as e:
            print(f"Error downloading klines for {year}-{month}: {e}")
            return None

    def download_monthly_funding(self, year: int, month: int) -> pd.DataFrame:
        url = f"{self.base_url}/fundingRate/{self.symbol}/{self.symbol}-fundingRate-{year}-{month:02d}.zip"
        print(f"Downloading funding: {url}")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"Unavailable: {url} (Status {response.status_code})")
                return None
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Funding historically has header: calc_time, funding_interval_hours, last_funding_rate
                    df = pd.read_csv(f)
            
            # Bug 2 Fix: Convert milliseconds to datetime
            df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
            df = df.rename(columns={"last_funding_rate": "funding_rate"})
            return df.set_index("calc_time")[["funding_rate"]]
        except Exception as e:
            print(f"Error downloading funding for {year}-{month}: {e}")
            return None

    def run_collection(self, start_date: str, end_date: str):
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        klines_list = []
        funding_list = []

        for d in tqdm(dates, desc="Collecting Months"):
            k = self.download_monthly_klines(d.year, d.month)
            f = self.download_monthly_funding(d.year, d.month)
            if k is not None: klines_list.append(k)
            if f is not None: funding_list.append(f)

        os.makedirs("data", exist_ok=True)

        if klines_list:
            full_k = pd.concat(klines_list).sort_index()
            # Bug 1 Fix: Real Delta computation
            full_k["taker_sell_base"] = full_k["volume"] - full_k["taker_buy_base"]
            full_k["delta"] = full_k["taker_buy_base"] - full_k["taker_sell_base"]
            full_k["cvd"] = full_k["delta"].cumsum()
            
            k_path = "data/btcusdt_1m.parquet"
            full_k.to_parquet(k_path)
            print(f"Klines saved: {len(full_k)} rows to {k_path}")
            # Verification: Section 3/17
            assert full_k.index[0].year >= 2021, f"Epoch bug detected: {full_k.index[0]}"
        
        if funding_list:
            full_f = pd.concat(funding_list).sort_index()
            f_path = "data/btcusdt_funding.parquet"
            full_f.to_parquet(f_path)
            print(f"Funding saved: {len(full_f)} rows to {f_path}")
            assert full_f.index[0].year >= 2021, f"Funding Epoch bug: {full_f.index[0]}"

if __name__ == "__main__":
    downloader = BinanceVisionDownloader()
    downloader.run_collection(CONFIG.TRAIN_START, CONFIG.TEST_END)
