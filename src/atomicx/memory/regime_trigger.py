"""Regime Change Trigger.

When the RegimeDetectorAgent detects a macro shift, this module orchestrates
the full strategy refresh pipeline:
1. Pause strategies tuned for the old regime
2. Pull best historical strategies from similar past regimes (Tier 4)
3. Run a fast Causal RL simulation on new regime data
4. Push refreshed strategies into Tier 2 and Tier 3
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.memory.tiers import (
    ShortTermTier1,
    StrategyGenomeTier2,
    CausalKnowledgeTier3,
    EvolutionaryArchiveTier4
)


class RegimeChangeTrigger:
    """Orchestrates automatic strategy refresh on regime transitions."""
    
    def __init__(
        self,
        tier1: ShortTermTier1,
        tier2: StrategyGenomeTier2,
        tier3: CausalKnowledgeTier3,
        tier4: EvolutionaryArchiveTier4
    ) -> None:
        self.tier1 = tier1
        self.tier2 = tier2
        self.tier3 = tier3
        self.tier4 = tier4
        self.logger = logger.bind(module="memory.regime_trigger")
        self.last_regime = "bullish_trend"
        
    def on_regime_shift(self, new_regime: str) -> None:
        """Execute the full strategy refresh pipeline."""
        old_regime = self.last_regime
        self.last_regime = new_regime
        
        self.logger.warning(f"REGIME SHIFT DETECTED: '{old_regime}' → '{new_regime}'")
        
        # Step 1: Archive current strategies to Tier-4 (never delete)
        self._archive_current_strategies(old_regime)
        
        # Step 2: Pull best historical strategies for the new regime from Tier-4
        candidates = self._pull_matching_strategies(new_regime)
        
        # Step 3: Simulate and score candidates (fast Causal RL pass)
        refreshed = self._fast_rl_rescore(candidates, new_regime)
        
        # Step 4: Push refreshed strategies into active Tier-2 Genome
        self._activate_refreshed_strategies(refreshed, new_regime)
        
        self.logger.success(f"Strategy refresh complete for regime '{new_regime}'. {len(refreshed)} strategies activated.")
        
    def _archive_current_strategies(self, old_regime: str) -> None:
        """Move all active Tier-2 strategies into Tier-4 cold storage."""
        for entry in self.tier2.genome_log:
            entry.data["archived_from_regime"] = old_regime
            self.tier4.archive_strategy(entry.data)
        
        archived_count = len(self.tier2.genome_log)
        self.tier2.genome_log.clear()
        self.logger.info(f"Archived {archived_count} strategies from regime '{old_regime}' to Tier-4.")
        
    def _pull_matching_strategies(self, target_regime: str) -> list[dict[str, Any]]:
        """Search Tier-4 for strategies that historically worked in a similar regime."""
        candidates = []
        for entry in self.tier4.archive:
            # In production: vector similarity search against regime embeddings
            archived_regime = entry.data.get("archived_from_regime", "")
            if archived_regime == target_regime or archived_regime == "":
                candidates.append(entry.data)
                
        self.logger.info(f"Found {len(candidates)} historical strategy candidates for regime '{target_regime}'.")
        return candidates
        
    def _fast_rl_rescore(self, candidates: list[dict[str, Any]], new_regime: str) -> list[dict[str, Any]]:
        """Run a fast Causal RL simulation pass to score candidates."""
        # MOCK: In production, this runs a lightweight DoWhy + RL policy optimization
        # against the last N hours of data under the new regime fingerprint.
        scored = []
        for c in candidates:
            c["rl_score"] = 0.7  # Placeholder score
            c["regime_fit"] = new_regime
            scored.append(c)
            
        scored.sort(key=lambda x: x.get("rl_score", 0), reverse=True)
        top_k = scored[:5]  # Activate top 5 strategies
        self.logger.info(f"Fast RL scored {len(scored)} candidates. Activating top {len(top_k)}.")
        return top_k
        
    def _activate_refreshed_strategies(self, strategies: list[dict[str, Any]], regime: str) -> None:
        """Push refreshed strategies into Tier-2 as the active genome."""
        for s in strategies:
            self.tier2.record_performance(
                strategy_id=s.get("strategy_id", "auto_refreshed"),
                outcome_metrics={"regime": regime, "rl_score": s.get("rl_score", 0), "status": "active"}
            )
