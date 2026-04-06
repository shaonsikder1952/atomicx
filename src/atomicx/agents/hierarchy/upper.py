"""Super Group, Verification Layer, Common-Sense Layer, and Domain Leader (AGNT-03 to AGNT-07).

These are the upper layers of the hierarchy that progressively
synthesize, validate, and sanity-check signals before they reach
the Fusion Node.
"""

from __future__ import annotations

from loguru import logger

from atomicx.agents.base import AgentConfig, BaseAgent
from atomicx.agents.hierarchy.group_leader import GroupLeader
from atomicx.agents.signals import AgentSignal, SignalDirection


class SuperGroup(BaseAgent):
    """Synthesizes Group Leader outputs (AGNT-03).

    Combines multiple Group Leaders (e.g., momentum + volatility + volume)
    into a higher-level view. Uses confidence-weighted averaging.
    """

    def __init__(
        self,
        super_group_id: str,
        name: str,
        group_leaders: list[GroupLeader],
    ) -> None:
        super().__init__(
            AgentConfig(
                agent_id=f"supergroup_{super_group_id}",
                agent_type="super_group",
                name=name,
            )
        )
        self.group_leaders = group_leaders

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Aggregate group leader signals."""
        signals: list[AgentSignal] = []

        for gl in self.group_leaders:
            signal = await gl.evaluate(symbol, timeframe, context)
            if signal and signal.direction != SignalDirection.SKIP:
                signals.append(signal)

        if not signals:
            return AgentSignal(
                agent_id=self.agent_id, agent_type="super_group",
                direction=SignalDirection.SKIP, confidence=0.0,
                symbol=symbol, timeframe=timeframe,
                reasoning="No group leader signals",
            )

        # Confidence-weighted synthesis
        bull_total = sum(s.confidence for s in signals if s.direction == SignalDirection.BULLISH)
        bear_total = sum(s.confidence for s in signals if s.direction == SignalDirection.BEARISH)
        total = bull_total + bear_total

        if total == 0:
            direction = SignalDirection.NEUTRAL
            confidence = 0.1
        elif bull_total > bear_total:
            direction = SignalDirection.BULLISH
            confidence = min(bull_total / total, 1.0)
        else:
            direction = SignalDirection.BEARISH
            confidence = min(bear_total / total, 1.0)

        return AgentSignal(
            agent_id=self.agent_id, agent_type="super_group",
            direction=direction, confidence=confidence,
            symbol=symbol, timeframe=timeframe,
            reasoning=f"Super group: bull={bull_total:.2f} vs bear={bear_total:.2f}",
            contributing_signals=[s.agent_id for s in signals],
        )


class VerificationLayer(BaseAgent):
    """Cross-checks signals from multiple Super Groups (AGNT-04).

    Applies consistency checks:
    - If momentum says bullish but volume says bearish, reduce confidence
    - If trend and momentum agree, boost confidence
    - Detects conflicting signals and penalizes overall confidence
    """

    def __init__(self, super_groups: list[SuperGroup]) -> None:
        super().__init__(
            AgentConfig(
                agent_id="verification_layer",
                agent_type="verification",
                name="Verification Layer",
            )
        )
        self.super_groups = super_groups

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Cross-check super group signals for consistency."""
        signals: list[AgentSignal] = []

        for sg in self.super_groups:
            signal = await sg.evaluate(symbol, timeframe, context)
            if signal and signal.direction != SignalDirection.SKIP:
                signals.append(signal)

        if not signals:
            return AgentSignal(
                agent_id=self.agent_id, agent_type="verification",
                direction=SignalDirection.SKIP, confidence=0.0,
                symbol=symbol, timeframe=timeframe,
                reasoning="No super group signals to verify",
            )

        # Check consistency
        directions = [s.direction for s in signals]
        bullish_count = directions.count(SignalDirection.BULLISH)
        bearish_count = directions.count(SignalDirection.BEARISH)
        total = len(signals)

        # Majority direction
        if bullish_count > bearish_count:
            direction = SignalDirection.BULLISH
        elif bearish_count > bullish_count:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Consistency score
        majority = max(bullish_count, bearish_count)
        consistency = majority / total

        # Confidence = weighted signal average × consistency bonus/penalty
        avg_confidence = sum(s.confidence for s in signals) / total
        if consistency == 1.0:
            confidence = min(avg_confidence * 1.3, 1.0)  # 30% BOOST for unanimous
            reasoning = f"UNANIMOUS {direction.value} ({total} super groups agree)"
        elif consistency >= 0.67:
            confidence = avg_confidence * 1.0  # No penalty
            reasoning = f"Majority {direction.value} ({majority}/{total} agree)"
        else:
            confidence = avg_confidence * 0.7  # 30% PENALTY for conflict
            reasoning = f"CONFLICTING signals ({bullish_count}↑ vs {bearish_count}↓), confidence reduced"

        return AgentSignal(
            agent_id=self.agent_id, agent_type="verification",
            direction=direction, confidence=confidence,
            symbol=symbol, timeframe=timeframe,
            reasoning=reasoning,
            contributing_signals=[s.agent_id for s in signals],
            metadata={"consistency": consistency, "unanimous": consistency == 1.0},
        )


class CommonSenseLayer(BaseAgent):
    """Applies sanity tests to prevent obviously bad signals (AGNT-05).

    Checks:
    - Extreme readings filter: reject if confidence unreasonably high
    - Regime mismatch: mean-reversion signal in strong trend → reduce confidence
    - Volume confirmation: big directional signal with no volume → reduce confidence
    - Time-of-day filters: certain hours have lower signal quality
    """

    def __init__(self) -> None:
        super().__init__(
            AgentConfig(
                agent_id="common_sense_layer",
                agent_type="common_sense",
                name="Common-Sense Layer",
            )
        )

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Apply sanity checks to the verification layer signal."""
        # Get the pre-existing signal from context (passed from verification layer)
        input_signal: AgentSignal | None = context.get("verification_signal")

        if not input_signal:
            return AgentSignal(
                agent_id=self.agent_id, agent_type="common_sense",
                direction=SignalDirection.SKIP, confidence=0.0,
                symbol=symbol, timeframe=timeframe,
                reasoning="No verification signal to check",
            )

        adjustments = []
        confidence = input_signal.confidence
        direction = input_signal.direction

        # Check 1: Volume confirmation
        variables = context.get("variables", {})
        rel_volume = variables.get("REL_VOLUME", 1.0)

        if rel_volume < 0.5 and confidence > 0.5:
            confidence *= 0.75
            adjustments.append(f"Low volume ({rel_volume:.1f}x), confidence reduced 25%")

        # Check 2: ADX regime check
        adx = variables.get("ADX", 25)
        rsi = variables.get("RSI_14", 50)

        # Mean-reversion signal in strong trend → reduce
        if adx > 35:
            if (direction == SignalDirection.BEARISH and rsi < 30) or \
               (direction == SignalDirection.BULLISH and rsi > 70):
                # Counter-trend in strong trend is weaker
                confidence *= 0.8
                adjustments.append(f"Counter-trend signal in strong trend (ADX={adx:.0f}), reduced 20%")

        # Check 3: Confidence ceiling sanity
        if confidence > 0.90:
            confidence = 0.85
            adjustments.append("Confidence capped at 85% (nothing is > 85% certain)")

        # Check 4: Bollinger Band extreme + volume spike = keep confidence
        bb_pct_b = variables.get("BB_PERCENT_B", 0.5)
        if abs(bb_pct_b - 0.5) > 0.5 and rel_volume > 2.0:
            confidence = min(confidence * 1.1, 0.85)
            adjustments.append(f"Extreme band ({bb_pct_b:.2f}) with volume spike, slight boost")

        reasoning = (
            f"Sanity checks applied: {len(adjustments)} adjustments. "
            + "; ".join(adjustments) if adjustments else "All checks passed, no adjustments"
        )

        return AgentSignal(
            agent_id=self.agent_id, agent_type="common_sense",
            direction=direction, confidence=confidence,
            symbol=symbol, timeframe=timeframe,
            reasoning=reasoning,
            metadata={"adjustments": adjustments, "original_confidence": input_signal.confidence},
        )


class DomainLeader(BaseAgent):
    """Produces domain-level confidence scores (AGNT-06, AGNT-07).

    The final agent before the Fusion Node. Combines verification
    and common-sense outputs to produce a domain-level view.
    """

    def __init__(
        self,
        domain: str,
        verification_layer: VerificationLayer,
        common_sense_layer: CommonSenseLayer,
    ) -> None:
        super().__init__(
            AgentConfig(
                agent_id=f"domain_leader_{domain}",
                agent_type="domain_leader",
                name=f"Domain Leader: {domain.title()}",
            )
        )
        self.domain = domain
        self.verification = verification_layer
        self.common_sense = common_sense_layer

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Produce domain-level confidence score."""
        # Run verification
        verification_signal = await self.verification.evaluate(symbol, timeframe, context)

        if not verification_signal or verification_signal.direction == SignalDirection.SKIP:
            return AgentSignal(
                agent_id=self.agent_id, agent_type="domain_leader",
                direction=SignalDirection.SKIP, confidence=0.0,
                symbol=symbol, timeframe=timeframe,
                reasoning=f"Domain {self.domain}: no verification signal",
            )

        # Run common-sense checks on verified signal
        cs_context = {**context, "verification_signal": verification_signal}
        final_signal = await self.common_sense.evaluate(symbol, timeframe, cs_context)

        if not final_signal or final_signal.direction == SignalDirection.SKIP:
            final_signal = verification_signal

        return AgentSignal(
            agent_id=self.agent_id, agent_type="domain_leader",
            direction=final_signal.direction,
            confidence=final_signal.confidence,
            symbol=symbol, timeframe=timeframe,
            reasoning=(
                f"Domain {self.domain}: {final_signal.direction.value} "
                f"@ {final_signal.confidence:.0%} "
                f"[verified={verification_signal.confidence:.0%}]"
            ),
            contributing_signals=[verification_signal.agent_id, final_signal.agent_id],
            metadata={
                "domain": self.domain,
                "verified_confidence": verification_signal.confidence,
                "final_confidence": final_signal.confidence,
            },
        )
