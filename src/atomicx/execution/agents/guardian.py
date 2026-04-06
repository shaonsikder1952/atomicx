"""Stop-Loss Guardian Agent.

An independent watchtower agent assigned to a specific execution plan.
Instantly flattens positions if invalidation triggers are hit.
"""

from __future__ import annotations

from loguru import logger
from atomicx.execution.agents.base import ChildAgent, AgentTask


class StopLossGuardian(ChildAgent):
    """Watches market risk and instantly closes positions on trigger."""
    
    def __init__(self, task: AgentTask) -> None:
        super().__init__(task)
        self.stop_pct = task.constraints.get("stop_pct", 0.05)
        self.entry_price = 0.0  # Would be synced from monitor
        self.logger = logger.bind(module="action.agents.guardian")
        self.status = "watching"

    async def execute_tick(self, current_price: float, liquidity_profile: dict) -> dict:
        """Watch the price against the stop validation point."""
        if self.entry_price == 0.0:
            self.entry_price = current_price  # Initial reference point mock

        drift = 0.0
        if self.task.direction == "bullish":
            drift = (self.entry_price - current_price) / self.entry_price
        elif self.task.direction == "bearish":
            drift = (current_price - self.entry_price) / self.entry_price
            
        if drift >= self.stop_pct:
            self.logger.error(f"[{self.agent_id}] STOP LOSS TRIGGERED! Drift: {drift:.2%} >= Limit: {self.stop_pct:.2%}")
            self.progress_pct = 1.0
            return {
                "action": "flatten_position",
                "reason": "stop_loss_hit",
                "trigger_price": current_price
            }
            
        return {}
