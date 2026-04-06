"""Accumulator Agent.

Executes ICEBERG or TWAP orders to build positions quietly without moving
the market. Manages slippage constraints in real-time.
"""

from __future__ import annotations

import asyncio
from loguru import logger
from atomicx.execution.agents.base import ChildAgent, AgentTask


class AccumulatorAgent(ChildAgent):
    """Gradually accumulates a position over time."""
    
    def __init__(self, task: AgentTask) -> None:
        super().__init__(task)
        self.accumulated_usd = 0.0
        self.max_slippage_bps = task.constraints.get("max_slippage_bps", 10)
        self.logger = logger.bind(module="action.agents.accumulator")
        self.status = "accumulating"

    async def execute_tick(self, current_price: float, liquidity_profile: dict) -> dict:
        """Evaluate market conditions and purchase a slice if safe."""
        if self.accumulated_usd >= self.task.target_size_usd:
            self.progress_pct = 1.0
            return {}

        # 1. Slippage Protection (Simulated)
        # Check if current liquidity can support our slice size
        slice_size = self.task.target_size_usd * 0.1  # Attempt 10% slices
        remaining = self.task.target_size_usd - self.accumulated_usd
        chunk = min(slice_size, remaining)

        book_depth = liquidity_profile.get("bid_depth" if self.task.direction == "bearish" else "ask_depth", 100000)
        
        if chunk > book_depth * 0.05:
            # Slippage risk too high, abort chunk
            self.logger.warning(f"[{self.agent_id}] Accumulation paused. Chunk {chunk} exceeds 5% of book depth {book_depth}.")
            return {"action": "wait", "reason": "slippage_protection"}
            
        # 2. Execute Chunk (Simulated)
        self.logger.info(f"[{self.agent_id}] Executing chunk: ${chunk:.2f} {self.task.direction.upper()} at estimated price ${current_price:.2f}")
        self.accumulated_usd += chunk
        self.progress_pct = self.accumulated_usd / self.task.target_size_usd

        return {
            "action": "market_order",
            "size": chunk,
            "direction": self.task.direction,
            "fill_price": current_price
        }
