"""Base Child Agent Module.

Defines the interface for all temporary, specialized execution agents that
operate within the Action Engine fleet.
"""

from __future__ import annotations

import abc
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    """A specific execution task assigned to a child agent."""
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    symbol: str
    direction: str
    target_size_usd: float
    urgency: str = "normal"  # low, normal, high, immediate
    constraints: dict = Field(default_factory=dict)


class ChildAgent(abc.ABC):
    """Base interface for all Action Engine child agents."""
    
    def __init__(self, task: AgentTask) -> None:
        self.agent_id = f"{self.__class__.__name__}-{uuid.uuid4().hex[:6]}"
        self.task = task
        self.status = "initialized"
        self.created_at = datetime.now(tz=timezone.utc)
        self.progress_pct = 0.0

    @abc.abstractmethod
    async def execute_tick(self, current_price: float, liquidity_profile: dict) -> dict:
        """Perform one execution tick.
        
        Returns a dict describing any actions taken (e.g., place_order, abort).
        """
        pass
        
    def abort(self, reason: str) -> None:
        """Emergency stop this agent."""
        self.status = "aborted"
        self.progress_pct = -1.0
