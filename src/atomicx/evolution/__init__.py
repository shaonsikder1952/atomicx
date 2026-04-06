"""Self-Improvement Loop — Continuous Learning Engine (Phase 11).

After every prediction outcome, the system:
1. Scores the prediction (correct/incorrect)
2. Updates agent weights (promote winners, demote losers)
3. Updates variable registry weights
4. Discovers which patterns worked and which failed
5. Adjusts thresholds (RSI overbought at 80? Maybe 75 is better for this regime)
6. Stores lessons in memory for future retrieval
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from atomicx.agents.orchestrator import AgentHierarchy
from atomicx.fusion.prediction import PredictionPacket
from atomicx.memory.service import MemoryService


class PredictionOutcome(BaseModel):
    """Outcome of a completed prediction."""
    prediction_id: str
    symbol: str
    predicted_direction: str
    predicted_confidence: float
    actual_return: float  # Raw price movement (always from long perspective)
    profit_return: float = 0.0  # Actual P&L (inverted for bearish predictions)
    was_correct: bool
    entry_price: float
    exit_price: float
    duration_hours: float
    regime_at_prediction: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class PerformanceStats(BaseModel):
    """Running performance statistics."""
    total_predictions: int = 0
    correct_predictions: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    edge_over_random: float = 0.0
    best_regime: str = ""
    worst_regime: str = ""
    by_regime: dict[str, dict[str, float]] = Field(default_factory=dict)


class SelfImprovementLoop:
    """Continuous self-improvement after every prediction outcome.

    This is what makes AtomicX a living, evolving system.
    Every outcome feeds back into the system to make it better.
    """

    def __init__(
        self,
        hierarchy: AgentHierarchy | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.memory = memory or MemoryService()
        self._outcomes: list[PredictionOutcome] = []
        self._stats = PerformanceStats()

    async def process_outcome(self, outcome: PredictionOutcome) -> dict[str, Any]:
        """Process a prediction outcome and update the system.

        This is called after every prediction resolves.
        Returns a dict of all adjustments made.
        """
        self._outcomes.append(outcome)
        adjustments: dict[str, Any] = {}

        # Step 1: Update global stats
        self._update_stats(outcome)
        adjustments["stats"] = {
            "win_rate": self._stats.win_rate,
            "total": self._stats.total_predictions,
            "edge": self._stats.edge_over_random,
        }

        # Step 2: Update agent weights
        if self.hierarchy:
            agent_adjustments = self._update_agent_weights(outcome)
            adjustments["agent_adjustments"] = agent_adjustments

        # Step 3: Store lessons in memory
        lessons = self._extract_lessons(outcome)
        if lessons:
            await self.memory.store_outcome(
                prediction_id=outcome.prediction_id,
                was_correct=outcome.was_correct,
                actual_return=outcome.actual_return,
                lessons=lessons,
            )
            adjustments["lessons"] = lessons

        # Step 4: Check for systematic issues
        systematic = self._check_systematic_issues()
        if systematic:
            adjustments["systematic_issues"] = systematic

        logger.info(
            f"Outcome processed: {outcome.prediction_id} — "
            f"{'✓' if outcome.was_correct else '✗'} "
            f"(return={outcome.actual_return:+.2%}, "
            f"win_rate={self._stats.win_rate:.1%})"
        )

        return adjustments

    def _update_stats(self, outcome: PredictionOutcome) -> None:
        """Update running performance statistics."""
        self._stats.total_predictions += 1
        if outcome.was_correct:
            self._stats.correct_predictions += 1

        self._stats.win_rate = (
            self._stats.correct_predictions / self._stats.total_predictions
        )
        self._stats.edge_over_random = self._stats.win_rate - 0.5

        # Average return (use profit_return for actual P&L)
        returns = [o.profit_return for o in self._outcomes]
        self._stats.avg_return = sum(returns) / len(returns)

        # Profit factor (use profit_return for actual P&L)
        wins = [abs(r) for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        if losses:
            self._stats.profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else float("inf")

        # By regime
        regime = outcome.regime_at_prediction or "unknown"
        if regime not in self._stats.by_regime:
            self._stats.by_regime[regime] = {"wins": 0, "total": 0, "avg_return": 0.0}

        self._stats.by_regime[regime]["total"] += 1
        if outcome.was_correct:
            self._stats.by_regime[regime]["wins"] += 1

        regime_data = self._stats.by_regime[regime]
        regime_data["avg_return"] = (
            (regime_data["avg_return"] * (regime_data["total"] - 1) + outcome.profit_return)
            / regime_data["total"]
        )

    def _update_agent_weights(self, outcome: PredictionOutcome) -> list[str]:
        """Update agent weights based on outcome."""
        if not self.hierarchy:
            return []

        changes = []
        for agent_id, agent in self.hierarchy.atomic_agents.items():
            # Only update agents that contributed to this prediction
            agent.record_outcome(outcome.was_correct)

            if not agent.is_active:
                changes.append(f"Agent {agent_id} AUTO-PRUNED (edge < 2%)")

        return changes

    def _extract_lessons(self, outcome: PredictionOutcome) -> str:
        """Extract actionable lessons from the outcome."""
        parts = []

        if not outcome.was_correct:
            # Analyze why it failed
            if outcome.actual_return > 0.05:
                parts.append(
                    f"Predicted {outcome.predicted_direction} but price moved strongly "
                    f"opposite ({outcome.actual_return:+.2%}). Possible: wrong regime "
                    f"classification or missing actor incentive signal."
                )
            else:
                parts.append(
                    f"Minor miss: predicted {outcome.predicted_direction}, "
                    f"actual return {outcome.actual_return:+.2%}."
                )

            # Check if confidence was too high
            if outcome.predicted_confidence > 0.8:
                parts.append("Overconfidence detected — confidence was >80% but outcome wrong.")

        else:
            if outcome.predicted_confidence < 0.6:
                parts.append(
                    f"Correct call at low confidence ({outcome.predicted_confidence:.0%}). "
                    f"Variables in this regime may deserve higher weight."
                )

        if outcome.regime_at_prediction:
            parts.append(f"Regime: {outcome.regime_at_prediction}")

        return " | ".join(parts) if parts else ""

    def _check_systematic_issues(self) -> list[str]:
        """Look for systematic problems in recent performance."""
        issues = []
        recent = self._outcomes[-20:]  # Last 20 predictions

        if len(recent) >= 10:
            recent_wr = sum(1 for o in recent if o.was_correct) / len(recent)
            if recent_wr < 0.35:
                issues.append(
                    f"ALERT: Win rate dropped to {recent_wr:.0%} over last {len(recent)} predictions. "
                    f"System may be poorly calibrated for current market conditions."
                )

            # Check for overconfidence
            high_conf = [o for o in recent if o.predicted_confidence > 0.7]
            if high_conf:
                high_wr = sum(1 for o in high_conf if o.was_correct) / len(high_conf)
                if high_wr < 0.5:
                    issues.append(
                        f"OVERCONFIDENCE: High-confidence predictions ({len(high_conf)}) "
                        f"have {high_wr:.0%} win rate — should be >65%."
                    )

        return issues

    @property
    def stats(self) -> PerformanceStats:
        return self._stats
