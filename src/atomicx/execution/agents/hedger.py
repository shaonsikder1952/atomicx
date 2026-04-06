"""Hedger Agent.

Handles derivatives hedges or delta-neutral strategies to protect
active portfolios based on high-volatility intents.
"""

from __future__ import annotations

import asyncio
from loguru import logger
from atomicx.execution.agents.base import ChildAgent, AgentTask


class HedgerAgent(ChildAgent):
    """Executes a defensive hedge safely without triggering liquidations."""
    
    def __init__(self, task: AgentTask) -> None:
        super().__init__(task)
        self.accumulated_usd = 0.0
        self.logger = logger.bind(module="action.agents.hedger")
        self.status = "hedging"

    async def execute_tick(self, current_price: float, liquidity_profile: dict) -> dict:
        """Gradually build the hedge position."""
        if self.accumulated_usd >= self.task.target_size_usd:
            self.progress_pct = 1.0
            return {}

        # Faster execution chunks for hedging (high urgency)
        chunk = self.task.target_size_usd * 0.25
        remaining = self.task.target_size_usd - self.accumulated_usd
        chunk = min(chunk, remaining)

        self.logger.warning(f"[{self.agent_id}] Placing HEDGE order: ${chunk:.2f} {self.task.direction.upper()} at estimated price ${current_price:.2f}")
        self.accumulated_usd += chunk
        self.progress_pct = self.accumulated_usd / self.task.target_size_usd

        return {
            "action": "hedge_order",
            "size": chunk,
            "direction": "short" if self.task.direction == "bearish" else "long",
            "fill_price": current_price
        }
