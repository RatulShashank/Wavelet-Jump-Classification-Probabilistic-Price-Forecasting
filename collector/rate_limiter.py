import time
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RateLimitState:
    """
    Tracks Binance API rate limits and weight usage.
    Section 3 — Rate Limit Manager logic.
    """
    weight_used_1m: int = 0
    weight_limit_1m: int = 2400
    safety_limit_1m: int = 1920  # 80% of 2400
    ban_events: int = 0
    requests_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def update_from_headers(self, headers: dict):
        self.requests_count += 1
        self.last_update = datetime.now()
        
        # Binance specific headers
        used = headers.get("X-MBX-USED-WEIGHT-1M")
        if used:
            self.weight_used_1m = int(used)

    def print_summary(self, elapsed: float):
        used_pct = (self.weight_used_1m / self.weight_limit_1m) * 100
        
        status = "OK"
        if used_pct > 85:
            status = "CRITICAL"
        elif used_pct > 70:
            status = "WARNING"
            
        summary = f"""
==================================================
  PIPELINE COMPLETE — API CREDIT SUMMARY
==================================================
  Weight used (1m):    {self.weight_used_1m} / {self.weight_limit_1m}
  Weight remaining:    {self.weight_limit_1m - self.weight_used_1m} ({100 - used_pct:.1f}% free)
  Requests this run:   {self.requests_count}
  Elapsed:             {elapsed:.1f}s
  Ban events:          {self.ban_events}
  Status: {status}
==================================================
"""
        print(summary)

    def check_safety(self):
        """Returns True if weight usage exceeds safety limit."""
        return self.weight_used_1m >= self.safety_limit_1m
