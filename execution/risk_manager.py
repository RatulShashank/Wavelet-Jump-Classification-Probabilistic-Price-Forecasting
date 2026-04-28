import numpy as np
from typing import Dict
from config.settings import CONFIG

class RiskManager:
    """
    Handles position sizing and drawdown circuit breakers.
    Section 19 — Risk Management logic.
    """
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_drawdown_limit: float = 0.15,
                 kelly_fraction: float = 0.5):
        self.capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0

    def calculate_kelly_size(self, win_rate: float, win_loss_ratio: float, 
                             current_balance: float) -> float:
        """
        Kelly Criterion: f* = p - (1-p)/b
        p = win rate, b = win/loss ratio.
        """
        if win_loss_ratio <= 0: return 0.0
        
        f_star = win_rate - (1 - win_rate) / win_loss_ratio
        # Apply fractional Kelly for safety
        size_pct = max(0, f_star * self.kelly_fraction)
        return current_balance * size_pct

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """
        Returns True if trading should be halted due to drawdown.
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if self.current_drawdown >= self.max_drawdown_limit:
            print(f"[CRITICAL] Max drawdown reached ({self.current_drawdown:.2%}). Circuit breaker active.")
            return True
        return False

    def get_position_size(self, signal_probs: np.ndarray, current_price: float) -> float:
        """
        Unified sizing logic for live system.
        """
        # Simplified placeholder for now. 
        # In production, this would use model confidence and historical win/loss metrics.
        confidence = signal_probs.max()
        if confidence < 0.65: return 0.0 # Signal too weak (GP Gater equivalent)
        
        return 0.1 # Default 0.1 BTC placeholder
