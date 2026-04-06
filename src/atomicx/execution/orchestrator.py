"""Action Orchestrator Module.

Translates high-level Intents passed from the Command Receiver into a detailed
execution plan. Spawns the exact combination of specialized child agents needed
to fulfill the Intent safely.
"""

from __future__ import annotations

from loguru import logger

from atomicx.execution.receiver import ReceivedCommand
from atomicx.execution.manager import FleetManager
from atomicx.execution.agents.base import AgentTask


class ActionOrchestrator:
    """Orchestrates the execution plan for validated Intents."""
    
    def __init__(self, fleet_manager: FleetManager) -> None:
        self.fleet_manager = fleet_manager
        self.logger = logger.bind(module="action.orchestrator")

    def execute_intent(self, command: ReceivedCommand) -> list[str]:
        """Convert a validated Intent into a fleet of active child agents."""
        self.logger.info(f"[{command.symbol}] ORCHESTRATOR: Translating Intent '{command.intent.intent_type}' into execution plan.")
        
        spawned_agent_ids = []
        intent_type = command.intent.intent_type
        direction = command.intent.direction
        
        # In a real system, target size would be derived from account balance & risk limits.
        # For Phase 15 skeleton, we mock standard sizes based on conviction.
        base_size = 1000.0 * command.intent.conviction
        
        if intent_type == "quiet_accumulation":
            # Spawn: 1 Accumulator + 1 Guardian
            acc_task = AgentTask(
                symbol=command.symbol,
                direction=direction,
                target_size_usd=base_size,
                urgency="low",
                constraints={"twap_duration_minutes": 120, "max_slippage_bps": 5}
            )
            # Dynamic import to avoid circular dependencies
            from atomicx.execution.agents.accumulator import AccumulatorAgent
            from atomicx.execution.agents.guardian import StopLossGuardian
            
            acc_agent = AccumulatorAgent(task=acc_task)
            guard_agent = StopLossGuardian(task=AgentTask(
                symbol=command.symbol, direction=direction, target_size_usd=0, constraints={"stop_pct": 0.05}
            ))
            
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(acc_agent))
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(guard_agent))
            self.logger.success(f"[{command.symbol}] PLAN GENERATED: Spawned 1x Accumulator + 1x Guardian.")
            
        elif intent_type == "defensive_hedge" or intent_type == "contrarian_hedge":
            # Spawn: 1 Hedger + 1 Guardian
            from atomicx.execution.agents.hedger import HedgerAgent
            from atomicx.execution.agents.guardian import StopLossGuardian
            
            hedge_task = AgentTask(
                symbol=command.symbol,
                direction="bearish" if direction == "bullish" else "bullish", # Hedge is inverse
                target_size_usd=base_size * 0.5, # Hedge 50%
                urgency="high",
                constraints={"max_slippage_bps": 20}
            )
            
            hedge_agent = HedgerAgent(task=hedge_task)
            guard_agent = StopLossGuardian(task=AgentTask(
                symbol=command.symbol, direction=direction, target_size_usd=0, constraints={"stop_pct": 0.02}
            ))
            
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(hedge_agent))
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(guard_agent))
            self.logger.success(f"[{command.symbol}] PLAN GENERATED: Spawned 1x Hedger + 1x Guardian.")

        elif intent_type == "trend_follow":
            # Standard TWAP entry
            from atomicx.execution.agents.accumulator import AccumulatorAgent
            from atomicx.execution.agents.guardian import StopLossGuardian
            
            acc_task = AgentTask(symbol=command.symbol, direction=direction, target_size_usd=base_size, urgency="normal")
            acc_agent = AccumulatorAgent(task=acc_task)
            guard_agent = StopLossGuardian(task=AgentTask(symbol=command.symbol, direction=direction, target_size_usd=0, constraints={"stop_pct": 0.08}))
            
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(acc_agent))
            spawned_agent_ids.append(self.fleet_manager.spawn_agent(guard_agent))
            self.logger.success(f"[{command.symbol}] PLAN GENERATED: Spawned 1x Standard Accumulator + 1x Guardian.")
            
        else:
            self.logger.warning(f"[{command.symbol}] No blueprint defined for Intent type: {intent_type}")
            
        return spawned_agent_ids
