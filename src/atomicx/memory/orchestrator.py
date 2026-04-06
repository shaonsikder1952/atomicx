"""Memory Orchestrator.

The central nervous system for memory. Synchronizes data flowing between
Tiers 0–4 and manages the background cognitive scanning agents.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.memory.tiers import (
    SensoryBufferTier0,
    ShortTermTier1,
    StrategyGenomeTier2,
    CausalKnowledgeTier3,
    EvolutionaryArchiveTier4
)
from atomicx.memory.agents.pattern import PatternMonitorAgent
from atomicx.memory.agents.regime import RegimeDetectorAgent
from atomicx.memory.agents.reflector import MetaReflector
from atomicx.memory.regime_trigger import RegimeChangeTrigger
from atomicx.memory.causal_rl import CausalRLUpdater
from atomicx.memory.genome import PerformanceGenome
from atomicx.memory.evolution import DailyEvolutionCycle
from atomicx.memory.pruner import DataPruner
from atomicx.memory.wiki import TradingWiki


class MemoryOrchestrator:
    """Manages the 5 active tiers of the living Memory Core + LLM Wiki."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="memory.orchestrator")

        # Initialize the 5 Tiers
        self.tier0 = SensoryBufferTier0()
        self.tier1 = ShortTermTier1()
        self.tier2 = StrategyGenomeTier2()
        self.tier3 = CausalKnowledgeTier3()
        self.tier4 = EvolutionaryArchiveTier4()

        # LLM Wiki: Persistent compounding knowledge (Karpathy pattern)
        self.wiki = TradingWiki()
        self.logger.info("[WIKI] Trading Intelligence Wiki initialized")

        # Week 1-2: Continuous Monitoring Agents
        self.pattern_agent = PatternMonitorAgent(tier0=self.tier0, tier1=self.tier1)
        self.regime_agent = RegimeDetectorAgent()
        self.reflector_agent = MetaReflector(tier2=self.tier2, tier3=self.tier3, tier4=self.tier4)

        # Week 3-4: Strategy Updating Loop
        self.regime_trigger = RegimeChangeTrigger(
            tier1=self.tier1, tier2=self.tier2, tier3=self.tier3, tier4=self.tier4
        )
        self.causal_rl = CausalRLUpdater(tier2=self.tier2)
        self._rl_cycle_counter = 0

        # Week 5-6: Performance Genome & Evolution
        self.genome = PerformanceGenome(tier2=self.tier2, tier4=self.tier4)
        self.evolution = DailyEvolutionCycle(genome=self.genome, tier4=self.tier4)

        # Phase 19: Data Pruner (Janitor)
        self.pruner = DataPruner(tier0=self.tier0, tier2=self.tier2)
        
    def stream_raw_data(self, data_packet: dict[str, Any]) -> None:
        """Ingest raw websocket/RPC data directly into Tier 0."""
        self.tier0.add(data_packet)
    
    async def cycle_maintenance(self, memory_service: Any = None) -> None:
        """Pulse the memory core to ensure pruning and synchronization."""
        # 1. Prune old raw data
        self.tier0.prune()
        
        # 2. Scan for new short-term patterns
        await self.pattern_agent.scan_for_patterns()
        
        # 3. Detect macroscopic regime drift → trigger full strategy refresh
        if self.tier0.buffer:
            latest = self.tier0.buffer[-1].data
            drift = await self.regime_agent.analyze_regime_drift(latest)
            if drift:
                self.tier1.active_patterns.clear()
                self.regime_trigger.on_regime_shift(drift)
                self.logger.warning(f"Full strategy refresh executed for regime: {drift}")
                
        # 4. Periodic Causal RL weight adjustment (every ~6 cycles ≈ 30s)
        self._rl_cycle_counter += 1
        if self._rl_cycle_counter % 6 == 0:
            adjustments = self.causal_rl.periodic_weight_adjustment()
            if adjustments:
                self.logger.info(f"[Causal RL] Weight adjustments applied: {adjustments}")
                
        # 5. Meta-Reflection on recent strategies
        await self.reflector_agent.reflect_on_genome()
        
        # 6. Daily Evolution Cycle (every ~50 cycles ≈ ~4 hours)
        if self._rl_cycle_counter % 50 == 0:
            self.logger.info("=== Triggering Daily Evolution Cycle ===")
            self.evolution.run_evolution(memory_service=memory_service)
            
        # 7. Data Pruning / Janitor (every ~100 cycles ≈ ~8 hours)
        if self._rl_cycle_counter % 100 == 0:
            self.logger.info("=== Running Data Pruner (Janitor) ===")
            self.pruner.run_pruning_cycle()
        
    def log_strategy_outcome(self, strategy_id: str, outcome: dict[str, Any]) -> None:
        """Add executed outcome data into the Strategy Genome (Tier 2), Causal RL, and Performance Genome."""
        self.tier2.record_performance(strategy_id, outcome)
        self.causal_rl.update_from_outcome(strategy_id, outcome)
        
        # Feed the Performance Genome
        self.genome.record_trade(
            strategy_id=strategy_id,
            profit=outcome.get("profit", 0),
            regime=outcome.get("regime", "unknown")
        )
        self.logger.info(f"[DMN] Logged strategy performance for {strategy_id} → Tier-2 + Causal RL + Genome")

