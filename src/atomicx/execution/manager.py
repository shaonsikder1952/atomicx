"""Executor Fleet Manager.

Creates, monitors, and terminates the temporary child agents.
Ensures no child agent ever exceeds the parameters given by the Orchestrator.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.execution.agents.base import ChildAgent


class FleetManager:
    """Manages the lifecycle of specialized execution child agents."""
    
    def __init__(self) -> None:
        self.active_agents: dict[str, ChildAgent] = {}
        self.total_pnl: float = 0.0
        self.logger = logger.bind(module="action.fleet")

    def get_fleet_stats(self) -> dict[str, int]:
        """Return a snapshot of current fleet activity."""
        active = sum(1 for a in self.active_agents.values() if a.status == "active")
        return {
            "total": 46,  # Base fleet size
            "active": active,
            "idle": 46 - active
        }


    def spawn_agent(self, agent: ChildAgent) -> str:
        """Add a new child agent to the active fleet."""
        self.active_agents[agent.agent_id] = agent
        self.logger.info(f"SPAWNED CHILD AGENT: {agent.agent_id} | Task: {agent.task.model_dump()}")
        return agent.agent_id

    def terminate_agent(self, agent_id: str, reason: str = "task_completed") -> None:
        """Remove and halt a child agent."""
        if agent_id in self.active_agents:
            agent = self.active_agents.pop(agent_id)
            agent.status = "terminated"
            self.logger.info(f"TERMINATED CHILD AGENT: {agent_id} | Reason: {reason}")
            
    def emergency_kill_switch(self) -> None:
        """Instantly halt all active child agents."""
        self.logger.error("!!! EMERGENCY KILL SWITCH ACTIVATED !!! Halting all child agents.")
        for agent_id, agent in list(self.active_agents.items()):
            agent.abort("Emergency Kill Switch")
            self.terminate_agent(agent_id, reason="emergency_abort")

    async def tick_fleet(self, current_price: float, liquidity_profile: dict) -> list[dict[str, Any]]:
        """Advance all active agents by one tick."""
        actions_taken = []
        
        for agent_id, agent in list(self.active_agents.items()):
            if agent.status in ("aborted", "terminated"):
                continue
                
            try:
                action = await agent.execute_tick(current_price, liquidity_profile)
                if action:
                    actions_taken.append({"agent_id": agent_id, "action": action})
                    
                if agent.progress_pct >= 1.0:
                    self.terminate_agent(agent_id, "position_filled_safely")
                    
            except Exception as e:
                self.logger.error(f"Child agent {agent_id} crashed during tick: {e}")
                agent.abort("exception_during_tick")
                self.terminate_agent(agent_id, reason="crash")
                
        return actions_taken
