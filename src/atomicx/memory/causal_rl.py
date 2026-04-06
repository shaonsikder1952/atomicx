"""Causal RL Strategy Updater.

Runs every 30-60 minutes and slightly adjusts active strategy parameters
based on real live performance data. This is the "continuous refinement"
engine that operates even when no regime change has occurred.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.memory.tiers import StrategyGenomeTier2


class CausalRLUpdater:
    """Continuously refines active strategy weights using causal reinforcement learning."""
    
    def __init__(self, tier2: StrategyGenomeTier2) -> None:
        self.tier2 = tier2
        self.logger = logger.bind(module="memory.causal_rl")
        self.update_count = 0
        
    def update_from_outcome(self, strategy_id: str, trade_outcome: dict[str, Any]) -> None:
        """Process a single closed trade and update the Strategy Genome."""
        self.update_count += 1
        
        profit = trade_outcome.get("profit", 0)
        slippage = trade_outcome.get("slippage_bps", 0)
        regime = trade_outcome.get("regime", "unknown")
        
        # Record the raw outcome into Tier-2
        self.tier2.record_performance(strategy_id, {
            "profit": profit,
            "slippage_bps": slippage,
            "regime": regime,
            "update_number": self.update_count
        })
        
        self.logger.info(
            f"[Causal RL] Update #{self.update_count}: Strategy '{strategy_id}' | "
            f"Profit: {profit:.2f} | Slippage: {slippage}bps | Regime: {regime}"
        )
        
    def periodic_weight_adjustment(self) -> dict[str, float]:
        """Run the periodic RL policy optimization pass.
        
        In production, this uses DoWhy to estimate causal effects of
        each strategy parameter on P&L, then adjusts weights accordingly.
        
        Returns updated weight recommendations.
        """
        if len(self.tier2.genome_log) < 5:
            return {}  # Not enough data yet
            
        # MOCK: Analyze recent performance and adjust
        recent_profits = []
        for entry in self.tier2.genome_log[-20:]:
            metrics = entry.data.get("metrics", {})
            recent_profits.append(metrics.get("profit", 0))
            
        avg_profit = sum(recent_profits) / len(recent_profits) if recent_profits else 0
        
        adjustments = {}
        if avg_profit > 0:
            adjustments["momentum_weight"] = min(1.5, 1.0 + avg_profit * 0.01)
            adjustments["causal_trust"] = min(1.2, 1.0 + avg_profit * 0.005)
            self.logger.success(f"[Causal RL] Positive drift detected. Boosting momentum_weight to {adjustments['momentum_weight']:.3f}")
        else:
            adjustments["momentum_weight"] = max(0.5, 1.0 + avg_profit * 0.01)
            adjustments["risk_scaling"] = max(0.7, 1.0 + avg_profit * 0.02)
            self.logger.warning(f"[Causal RL] Negative drift detected. Reducing momentum_weight to {adjustments['momentum_weight']:.3f}")
            
        return adjustments
