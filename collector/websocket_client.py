import asyncio
import json
import websockets
import logging
from typing import Callable, Dict, Optional
from config.settings import CONFIG

class BinanceWebSocketClient:
    """
    Real-time data ingestion via Binance WebSockets.
    Section 3 & 14 — WebSocket Streams logic.
    """
    def __init__(self, callback: Callable[[Dict], None]):
        self.base_url = CONFIG.WS_BASE_URL
        self.callback = callback
        self.running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """
        Connects to multiple streams: Klines, Liquidations, and Mark Price.
        """
        # Stream names: symbol@kline_1m, !forceOrder@arr, symbol@markPrice
        streams = [
            f"{CONFIG.SYMBOL.lower()}@kline_1m",
            f"{CONFIG.SYMBOL.lower()}@markPrice",
            "!forceOrder@arr"
        ]
        stream_path = "/stream?streams=" + "/".join(streams)
        url = f"{self.base_url}{stream_path}"
        
        self.running = True
        print(f"Connecting to WebSocket: {url}")
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    print("WebSocket Connected.")
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        # Dispatch to callback
                        if "data" in data:
                            await self.callback(data["data"])
                        else:
                            await self.callback(data)
                            
            except (websockets.ConnectionClosed, Exception) as e:
                print(f"WebSocket Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False

async def example_callback(data: Dict):
    """Simple dispatcher for testing."""
    stream = data.get("e")
    if stream == "kline":
        k = data["k"]
        if k["x"]: # If bar closed
            print(f"1m Bar Closed: {k['c']} at {pd.to_datetime(data['E'], unit='ms')}")
    elif stream == "forceOrder":
        o = data["o"]
        print(f"LIQUIDATION: {o['S']} {o['q']} {CONFIG.SYMBOL} at {o['p']}")

if __name__ == "__main__":
    import pandas as pd # For timestamp formatting in logs
    client = BinanceWebSocketClient(example_callback)
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        client.stop()
