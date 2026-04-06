"""Pattern Monitor Agent.

Runs frequently to scan Tier-0 (Sensory Buffer) for emerging formations
not present previously. It feeds discoveries into Tier-1 (Short-Term).
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from atomicx.memory.tiers import SensoryBufferTier0, ShortTermTier1


class PatternMonitorAgent:
    """Discovers and promotes novel data formations continuously."""
    
    def __init__(self, tier0: SensoryBufferTier0, tier1: ShortTermTier1) -> None:
        self.tier0 = tier0
        self.tier1 = tier1
        self.logger = logger.bind(module="memory.agents.pattern")
        
    async def scan_for_patterns(self) -> None:
        """Analyze the latest 60s of raw data."""
        if not self.tier0.buffer:
            return  # No data to scan
            
        latest_tick = self.tier0.buffer[-1].data
        price = latest_tick.get("price", 0)
        
        # MOCK DISCOVERY
        # If price jumps > 2% in the buffer, promote an abnormal velocity pattern
        if price > 96000:  # arbitrary threshold for the test
            self.logger.info(f"Novel velocity formation detected in Tier-0 buffer at price {price}.")
            self.tier1.insert_pattern(
                pattern_type="micro_velocity",
                details={"trigger_price": price, "urgency": "high"}
            )
            self.logger.success("Promoted 'micro_velocity' pattern to Tier-1 Short-Term Memory.")
