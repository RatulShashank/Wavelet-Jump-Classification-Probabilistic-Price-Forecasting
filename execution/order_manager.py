import asyncio
from typing import Dict, Optional
from collector.rest_client import BinanceFuturesClient
from config.settings import CONFIG

class LiveOrderManager:
    """
    Handles live order execution, risk control, and position management.
    Section 19 — Execution logic.
    """
    def __init__(self, rest_client: BinanceFuturesClient, dry_run: bool = True):
        self.rest_client = rest_client
        self.dry_run = dry_run # Default to True for safety
        self.symbol = CONFIG.SYMBOL
        self.active_position = False

    async def calculate_size(self, balance_pct: float = 0.05) -> float:
        """
        Estimates position size (in BTC) based on account balance and risk %
        """
        # In production, we'd fetch account balance here
        # For now, using a placeholder logic for size
        return 0.1 # 0.1 BTC placeholder

    async def execute_entry(self, direction: str, target_label: str):
        """
        Executes a Market order entry and sets a Hard Stop Loss.
        """
        if self.active_position:
            print("[EXECUTION] Active position exists. Skipping new entry.")
            return

        size = await self.calculate_size()
        side = "BUY" if target_label in ["strong_up", "mild_up"] else "SELL"
        
        if self.dry_run:
            print(f"[DRY RUN] Would execute {side} {size} {self.symbol} for signal {target_label}")
            self.active_position = True
            return

        # Real execution logic (Section 19)
        print(f"[LIVE] Executing {side} MARKET order for {size} {self.symbol}...")
        order_params = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": str(size)
        }
        
        response = await self.rest_client.request("POST", "/fapi/v1/order", params=order_params)
        
        if response and "orderId" in response:
            print(f"[LIVE] Entry success: {response['orderId']}")
            self.active_position = True
            await self.set_stop_loss(side, float(response["avgPrice"]), size)
        else:
            print("[LIVE] Entry FAILED.")

    async def set_stop_loss(self, entry_side: str, entry_price: float, size: float):
        """
        Sets a Hard Stop Loss at 1% distance.
        """
        stop_side = "SELL" if entry_side == "BUY" else "BUY"
        stop_price = entry_price * (0.99 if entry_side == "BUY" else 1.01)
        
        print(f"[LIVE] Setting Stop Loss at {stop_price:.2f}...")
        
        params = {
            "symbol": self.symbol,
            "side": stop_side,
            "type": "STOP_MARKET",
            "stopPrice": str(round(stop_price, 2)),
            "quantity": str(size),
            "reduceOnly": "true"
        }
        
        if self.dry_run:
            print(f"[DRY RUN] Would set STOP_MARKET at {stop_price}")
            return
            
        await self.rest_client.request("POST", "/fapi/v1/order", params=params)

    async def monitor_exit(self):
        """
        Logic to exit after 60 minutes (the validation window).
        """
        # To be implemented with a timer based on entry timestamp
        pass

async def test_execution():
    from collector.rest_client import BinanceFuturesClient
    client = BinanceFuturesClient()
    manager = LiveOrderManager(client, dry_run=True)
    await manager.execute_entry("up", "strong_up")
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_execution())
