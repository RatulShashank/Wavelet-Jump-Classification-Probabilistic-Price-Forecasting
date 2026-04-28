import asyncio
import signal
from collector.rest_client import BinanceFuturesClient
from collector.websocket_client import BinanceWebSocketClient
from storage.live_state import LiveStateManager
from features.live_processor import LiveSignalProcessor
from execution.order_manager import LiveOrderManager
from config.settings import CONFIG

class BTCJumpTradingSystem:
    """
    Unified Live System Entry Point.
    Integrates ingestion, processing, and execution for Phase 2.
    """
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        
        # 1. Initialize Components
        self.rest_client = BinanceFuturesClient()
        self.state_mgr = LiveStateManager()
        self.order_mgr = LiveOrderManager(self.rest_client, dry_run=self.dry_run)
        
        # 2. Processor depends on order manager for execution calls
        self.processor = LiveSignalProcessor(self.state_mgr)
        # Note: In a real system, we'd inject order_mgr into processor
        self.processor.order_mgr = self.order_mgr
        
        # 3. WS Client depends on processor callback
        self.ws_client = BinanceWebSocketClient(self.processor.handle_ws_message)

    async def run(self):
        """
        Starts the system and sets up termination handlers.
        """
        print(f"====================================================")
        print(f"  BTC JUMP LIVE SYSTEM — PHASE 2")
        print(f"  Status: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"  Target: {CONFIG.SYMBOL}")
        print(f"====================================================")

        # Set up graceful shutdown
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        
        def shutdown():
            print("\nShutting down system...")
            self.ws_client.stop()
            stop_event.set()

        # Handle signals
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
             # signal.add_signal_handler is not implemented on Windows
             pass

        # Start WS connection in background
        ws_task = asyncio.create_task(self.ws_client.connect())
        
        # Keep running until stop event or WS task failure
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Resource cleanup."""
        await self.rest_client.close()
        print("System stopped. Sessions closed.")

if __name__ == "__main__":
    # To run: python main_live.py
    # Change dry_run=False ONLY after passing 24h of dry run audit.
    system = BTCJumpTradingSystem(dry_run=True)
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        pass
