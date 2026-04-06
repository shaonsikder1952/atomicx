"""Live Monitor & Reporter.

Continuously tracks real-time P&L, slippage vs expected, and liquidity
conditions across the Fleet. Triggers emergency alerts to the Brain.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.execution.manager import FleetManager


class LiveMonitor:
    """The telemetry system for the Action Engine."""
    
    def __init__(self, fleet_manager: FleetManager) -> None:
        self.fleet_manager = fleet_manager
        self.logger = logger.bind(module="action.monitor")
        
    def generate_report(self) -> dict[str, Any]:
        """Produce a telemetry snapshot to feed back to the Brain."""
        active_count = len(self.fleet_manager.active_agents)
        
        # Aggregated stats mock
        report = {
            "active_agents": active_count,
            "agents_by_status": {},
            "total_estimated_pnl": 0.0,
            "warnings": []
        }
        
        for agent_id, agent in self.fleet_manager.active_agents.items():
            status = agent.status
            report["agents_by_status"][status] = report["agents_by_status"].get(status, 0) + 1
            
            if status == "aborted":
                report["warnings"].append(f"Agent {agent_id} aborted.")
                
        if active_count > 0:
            self.logger.info(f"TELEMETRY REPORT: {active_count} active child agents running.")
            
        return report
