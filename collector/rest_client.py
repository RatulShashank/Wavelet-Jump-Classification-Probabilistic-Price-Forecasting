import asyncio
import aiohttp
import time
import logging
from typing import Optional, List, Dict
from .rate_limiter import RateLimitState
from config.settings import CONFIG

class BinanceFuturesClient:
    """
    Asynchronous Binance USDM Futures REST Client with Rate Limiting.
    Section 3 — REST Endpoints architecture.
    """
    def __init__(self, base_url: str = CONFIG.BASE_URL):
        self.base_url = base_url
        self.limiter = RateLimitState()
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()
        
        for attempt in range(CONFIG.MAX_RETRIES):
            if self.limiter.check_safety():
                print("Safety ceiling reached (80%). Pausing for 5 seconds...")
                await asyncio.sleep(5)

            async with session.request(method, url, **kwargs) as response:
                self.limiter.update_from_headers(response.headers)
                
                if response.status == 200:
                    return await response.json()
                
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    print(f"Rate limited (429). Retry after {retry_after}s")
                    await asyncio.sleep(retry_after + 1)
                elif response.status == 418:
                    self.limiter.ban_events += 1
                    print("IP Banned (418). Sleeping 120s...")
                    await asyncio.sleep(120)
                elif response.status == 503:
                    backoff = (2 ** attempt) * CONFIG.BACKOFF_BASE_MS / 1000
                    print(f"Server overloaded (503). Backoff {backoff}s")
                    await asyncio.sleep(backoff)
                elif response.status == 451:
                    print("Geo-blocked (451). Wrong URL! Use Production URL.")
                    return None
                else:
                    print(f"Request failed ({response.status}): {await response.text()}")
                    return None
        return None

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[List]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self.request("GET", "/fapi/v1/klines", params=params)

    async def get_open_interest(self, symbol: str) -> Optional[Dict]:
        params = {"symbol": symbol}
        return await self.request("GET", "/fapi/v1/openInterest", params=params)

    async def close(self):
        if self.session:
            await self.session.close()

async def smoke_test():
    """Bug 3 Fix: Always smoke-test against production URL."""
    client = BinanceFuturesClient(base_url="https://fapi.binance.com")
    try:
        klines = await client.get_klines(CONFIG.SYMBOL, "1m", limit=5)
        if klines and len(klines) == 5:
            print("Smoke test: SUCCESS (Production URL reachable)")
            client.limiter.print_summary(0)
        else:
            print("Smoke test: FAILED")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(smoke_test())
