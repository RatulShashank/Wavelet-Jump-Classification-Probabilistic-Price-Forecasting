import asyncio
import pandas as pd
from collector.rest_client import BinanceFuturesClient
from config.settings import CONFIG

async def main():
    client = BinanceFuturesClient(base_url="https://fapi.binance.com")
    # Try to get funding for a specific window
    # startTime: 2024-04-01 (1711929600000)
    # endTime: 2024-04-02 (1712016000000)
    params = {"symbol": CONFIG.SYMBOL, "startTime": 1711929600000, "endTime": 1712016000000}
    res = await client.request("GET", "/fapi/v1/fundingRate", params=params)
    print(res)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
