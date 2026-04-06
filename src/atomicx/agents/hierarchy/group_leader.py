"""Group Leader — aggregates 3-8 Atomic Agents (AGNT-02).

Groups atomic agents by category (e.g., all momentum indicators)
and produces a weighted consensus signal.
"""

from __future__ import annotations

import numpy as np

from atomicx.agents.base import AgentConfig, BaseAgent
from atomicx.agents.signals import (
    AgentSignal,
    AggregatedSignal,
    SignalDirection,
)


class GroupLeader(BaseAgent):
    """Aggregates signals from 3-8 atomic agents in a category.

    Aggregation strategy:
    - Weighted average of child confidences
    - Direction by weighted majority vote
    - Consensus strength = agreement ratio among active children
    """

    def __init__(
        self,
        group_id: str,
        name: str,
        children: list[BaseAgent],
    ) -> None:
        super().__init__(
            AgentConfig(
                agent_id=f"group_{group_id}",
                agent_type="group_leader",
                name=name,
            )
        )
        self.children = children

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Aggregate all child agent signals."""
        child_signals: list[AgentSignal] = []
        skipped = 0

        for child in self.children:
            signal = await child.evaluate(symbol, timeframe, context)
            if signal is None or signal.direction == SignalDirection.SKIP:
                skipped += 1
                continue
            child_signals.append(signal)

        if not child_signals:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type="group_leader",
                direction=SignalDirection.SKIP,
                confidence=0.0,
                symbol=symbol,
                timeframe=timeframe,
                reasoning=f"All {len(self.children)} children skipped",
            )

        # Weighted aggregation
        direction, confidence, consensus = self._aggregate(child_signals)

        return AgentSignal(
            agent_id=self.agent_id,
            agent_type="group_leader",
            direction=direction,
            confidence=confidence,
            symbol=symbol,
            timeframe=timeframe,
            reasoning=(
                f"{len(child_signals)} active / {skipped} skipped, "
                f"consensus: {consensus:.0%}"
            ),
            contributing_signals=[s.agent_id for s in child_signals],
            metadata={
                "consensus": consensus,
                "active": len(child_signals),
                "skipped": skipped,
            },
        )

    def _aggregate(
        self, signals: list[AgentSignal]
    ) -> tuple[SignalDirection, float, float]:
        """Compute weighted consensus direction and confidence."""
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0

        for s in signals:
            w = s.weight * s.confidence
            if s.direction == SignalDirection.BULLISH:
                bullish_weight += w
            elif s.direction == SignalDirection.BEARISH:
                bearish_weight += w
            total_weight += w

        if total_weight == 0:
            return SignalDirection.NEUTRAL, 0.0, 0.0

        # Direction by weighted majority
        if bullish_weight > bearish_weight:
            direction = SignalDirection.BULLISH
            confidence = bullish_weight / total_weight
        elif bearish_weight > bullish_weight:
            direction = SignalDirection.BEARISH
            confidence = bearish_weight / total_weight
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.1

        # Consensus — how much children agree
        max_weight = max(bullish_weight, bearish_weight)
        consensus = max_weight / total_weight if total_weight > 0 else 0

        # Scale confidence by average child confidence
        avg_conf = np.mean([s.confidence for s in signals])
        confidence = min(confidence * avg_conf * 1.5, 1.0)

        return direction, float(confidence), float(consensus)
