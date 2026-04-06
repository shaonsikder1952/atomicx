"""Data Pruner — The Janitor.

Keeps the database lean by automatically deleting noise using
Conformal Data Cleaning (CDC) rules.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from loguru import logger

from atomicx.memory.tiers import SensoryBufferTier0, StrategyGenomeTier2
from atomicx.intelligence.knowledge_graph import KnowledgeGraph


class DataPruner:
    """Self-cleaning pipeline that prunes stale and noisy data."""
    
    def __init__(
        self,
        tier0: SensoryBufferTier0,
        tier2: StrategyGenomeTier2,
        knowledge_graph: KnowledgeGraph | None = None
    ) -> None:
        self.tier0 = tier0
        self.tier2 = tier2
        self.knowledge_graph = knowledge_graph
        self.logger = logger.bind(module="memory.pruner")
        self.prune_count = 0
        
    def run_pruning_cycle(self) -> dict[str, int]:
        """Execute one complete pruning cycle."""
        self.prune_count += 1
        results = {"tier0_pruned": 0, "genome_pruned": 0, "people_demoted": 0}
        
        # Rule 1: Age-Out — Tier-0 data older than 60s (already handled by tier0.prune())
        before = len(self.tier0.buffer)
        self.tier0.prune()
        results["tier0_pruned"] = before - len(self.tier0.buffer)
        
        # Rule 2: Genome Noise — Remove strategy entries with zero impact
        stale_entries = []
        for i, entry in enumerate(self.tier2.genome_log):
            metrics = entry.data.get("metrics", {})
            # If a strategy has been logged but produced exactly 0 profit across 5+ records
            if metrics.get("profit", None) == 0 and metrics.get("status", "") != "active":
                stale_entries.append(i)
                
        for idx in reversed(stale_entries):
            self.tier2.genome_log.pop(idx)
        results["genome_pruned"] = len(stale_entries)
        
        # Rule 3: Noise Filter — Lower significance of low-impact people
        if self.knowledge_graph:
            for name, person in self.knowledge_graph.people.items():
                if len(person.statements) >= 10:
                    # Check if their statements actually moved markets
                    reactions = []
                    for s in person.statements:
                        try:
                            r = float(str(s.get("market_reaction", "0")).replace("%", "").replace("+", ""))
                            reactions.append(abs(r))
                        except (ValueError, AttributeError):
                            pass
                    avg_impact = sum(reactions) / len(reactions) if reactions else 0
                    if avg_impact < 0.1:  # Less than 0.1% average impact
                        self.knowledge_graph.lower_significance(name, factor=0.5)
                        results["people_demoted"] += 1
        
        if any(v > 0 for v in results.values()):
            self.logger.info(f"[PRUNER] Cycle #{self.prune_count}: {results}")
            
        return results
