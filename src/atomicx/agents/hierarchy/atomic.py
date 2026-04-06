"""Atomic Agent — one agent per variable (AGNT-01).

The lowest level of the hierarchy. Each atomic agent owns exactly
one variable and produces a directional score based on its current value,
historical context, and causal relationships.
"""

from __future__ import annotations

from atomicx.agents.base import AgentConfig, BaseAgent
from atomicx.agents.signals import AgentSignal, SignalDirection


class AtomicAgent(BaseAgent):
    """Owns a single variable and produces directional signals.

    Signal logic is based on:
    1. Current variable value vs historical distribution
    2. Rate of change (momentum)
    3. Causal weight (from the causal DAG)
    4. Regime-dependent thresholds
    """

    def __init__(self, variable_id: str, config: AgentConfig | None = None) -> None:
        self.variable_id = variable_id
        cfg = config or AgentConfig(
            agent_id=f"atomic_{variable_id}",
            agent_type="atomic",
            name=f"Atomic Agent: {variable_id}",
        )
        super().__init__(cfg)

        # Configurable thresholds — adjusted by the self-improvement loop
        self._bullish_threshold: float = 0.6
        self._bearish_threshold: float = 0.4

    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Generate directional signal based on variable value.

        Context expected keys:
            variables: dict[str, float]  — current variable snapshot
            causal_weight: float — weight from causal DAG
            regime: str — current market regime
        """
        variables = context.get("variables", {})
        value = variables.get(self.variable_id)

        if value is None:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type="atomic",
                direction=SignalDirection.SKIP,
                confidence=0.0,
                symbol=symbol,
                timeframe=timeframe,
                variable_id=self.variable_id,
                reasoning=f"No data for {self.variable_id}",
            )

        # Compute directional score
        direction, confidence, reasoning = self._analyze_value(
            value, variables, context
        )

        return AgentSignal(
            agent_id=self.agent_id,
            agent_type="atomic",
            direction=direction,
            confidence=confidence,
            symbol=symbol,
            timeframe=timeframe,
            variable_id=self.variable_id,
            reasoning=reasoning,
        )

    def _analyze_value(
        self, value: float, variables: dict, context: dict
    ) -> tuple[SignalDirection, float, str]:
        """Analyze variable value and produce signal.

        Uses rule-based logic for common indicators,
        with generic z-score fallback for others.
        """
        var_id = self.variable_id.upper()

        # RSI-family
        if "RSI" in var_id:
            return self._analyze_rsi(value)

        # MACD-family
        if "MACD" in var_id:
            return self._analyze_macd(value, variables)

        # Bollinger-family
        if "BB_PERCENT_B" in var_id:
            return self._analyze_bollinger(value)

        # Volume-based
        if "REL_VOLUME" in var_id:
            return self._analyze_volume(value)

        # Trend strength
        if "ADX" in var_id:
            return self._analyze_adx(value)

        # Funding rate
        if "FUNDING" in var_id:
            return self._analyze_funding(value)

        # Order book
        if "OB_IMBALANCE" in var_id:
            return self._analyze_orderbook(value)

        # Generic: use causal weight to determine direction
        return self._analyze_generic(value, context)

    def _analyze_rsi(self, value: float) -> tuple[SignalDirection, float, str]:
        """RSI with MOMENTUM interpretation for intraday 5-min trading.

        FIX: Changed from mean-reversion (contrarian) to momentum interpretation.
        High RSI = strong uptrend = BULLISH (not bearish mean-reversion).
        """
        if value > 80:
            # Strong uptrend momentum
            conf = min(0.3 + (value - 80) / 40, 0.85)
            return SignalDirection.BULLISH, conf, f"RSI strong uptrend momentum ({value:.1f})"
        elif value > 70:
            # Uptrend momentum
            conf = 0.2 + (value - 70) / 50
            return SignalDirection.BULLISH, conf, f"RSI uptrend momentum ({value:.1f})"
        elif value < 20:
            # Strong downtrend momentum
            conf = min(0.3 + (20 - value) / 40, 0.85)
            return SignalDirection.BEARISH, conf, f"RSI strong downtrend momentum ({value:.1f})"
        elif value < 30:
            # Downtrend momentum
            conf = 0.2 + (30 - value) / 50
            return SignalDirection.BEARISH, conf, f"RSI downtrend momentum ({value:.1f})"
        else:
            return SignalDirection.NEUTRAL, 0.1, f"RSI neutral at {value:.1f}"

    def _analyze_macd(
        self, value: float, variables: dict
    ) -> tuple[SignalDirection, float, str]:
        histogram = variables.get("MACD_HISTOGRAM", value)
        if histogram > 0:
            conf = min(abs(histogram) * 10, 0.7)
            return SignalDirection.BULLISH, conf, f"MACD histogram positive: {histogram:.4f}"
        elif histogram < 0:
            conf = min(abs(histogram) * 10, 0.7)
            return SignalDirection.BEARISH, conf, f"MACD histogram negative: {histogram:.4f}"
        return SignalDirection.NEUTRAL, 0.1, "MACD histogram at zero"

    def _analyze_bollinger(self, value: float) -> tuple[SignalDirection, float, str]:
        """Bollinger %B with MOMENTUM interpretation for intraday trading.

        FIX: Changed from mean-reversion to momentum (breakout continuation).
        Breakout above = BULLISH continuation (not bearish reversion).
        """
        if value > 1.0:
            # Breakout above upper band = bullish continuation
            conf = min(0.3 + (value - 1.0), 0.75)
            return SignalDirection.BULLISH, conf, f"Breakout above upper BB (momentum, %B={value:.2f})"
        elif value < 0.0:
            # Breakdown below lower band = bearish continuation
            conf = min(0.3 + abs(value), 0.75)
            return SignalDirection.BEARISH, conf, f"Breakdown below lower BB (momentum, %B={value:.2f})"
        elif value > 0.8:
            # Near upper band = bullish bias
            return SignalDirection.BULLISH, 0.2, f"Near upper BB (bullish bias, %B={value:.2f})"
        elif value < 0.2:
            # Near lower band = bearish bias
            return SignalDirection.BEARISH, 0.2, f"Near lower BB (bearish bias, %B={value:.2f})"
        return SignalDirection.NEUTRAL, 0.1, f"Mid-band neutral (%B={value:.2f})"

    def _analyze_volume(self, value: float) -> tuple[SignalDirection, float, str]:
        if value > 3.0:
            return SignalDirection.BULLISH, 0.5, f"Volume spike: {value:.1f}x average (watch for breakout)"
        elif value > 2.0:
            return SignalDirection.BULLISH, 0.3, f"High volume: {value:.1f}x average"
        elif value < 0.5:
            return SignalDirection.NEUTRAL, 0.15, f"Low volume: {value:.1f}x average (thin market)"
        return SignalDirection.NEUTRAL, 0.1, f"Normal volume: {value:.1f}x average"

    def _analyze_adx(self, value: float) -> tuple[SignalDirection, float, str]:
        if value > 40:
            return SignalDirection.NEUTRAL, 0.6, f"Strong trend (ADX={value:.1f}), follow direction"
        elif value > 25:
            return SignalDirection.NEUTRAL, 0.4, f"Moderate trend (ADX={value:.1f})"
        return SignalDirection.NEUTRAL, 0.2, f"Weak/no trend (ADX={value:.1f}), mean-revert regime"

    def _analyze_funding(self, value: float) -> tuple[SignalDirection, float, str]:
        if value > 0.05:
            return SignalDirection.BEARISH, 0.6, f"Extreme positive funding ({value:.4f}), overleveraged longs"
        elif value > 0.01:
            return SignalDirection.BEARISH, 0.3, f"Positive funding ({value:.4f})"
        elif value < -0.05:
            return SignalDirection.BULLISH, 0.6, f"Extreme negative funding ({value:.4f}), overleveraged shorts"
        elif value < -0.01:
            return SignalDirection.BULLISH, 0.3, f"Negative funding ({value:.4f})"
        return SignalDirection.NEUTRAL, 0.1, f"Neutral funding ({value:.4f})"

    def _analyze_orderbook(self, value: float) -> tuple[SignalDirection, float, str]:
        if value > 0.3:
            return SignalDirection.BULLISH, 0.5, f"Strong buy pressure (imbalance={value:.2f})"
        elif value > 0.1:
            return SignalDirection.BULLISH, 0.25, f"Mild buy pressure (imbalance={value:.2f})"
        elif value < -0.3:
            return SignalDirection.BEARISH, 0.5, f"Strong sell pressure (imbalance={value:.2f})"
        elif value < -0.1:
            return SignalDirection.BEARISH, 0.25, f"Mild sell pressure (imbalance={value:.2f})"
        return SignalDirection.NEUTRAL, 0.1, f"Balanced book (imbalance={value:.2f})"

    def _analyze_generic(
        self, value: float, context: dict
    ) -> tuple[SignalDirection, float, str]:
        """Generic analysis using causal weight as guide."""
        weights = context.get("causal_weights", {})
        causal_w = weights.get(self.variable_id, 0.5)

        if causal_w > 0:
            direction = SignalDirection.BULLISH if value > 0 else SignalDirection.BEARISH
        elif causal_w < 0:
            direction = SignalDirection.BEARISH if value > 0 else SignalDirection.BULLISH
        else:
            direction = SignalDirection.NEUTRAL

        confidence = min(abs(causal_w) * abs(value) * 0.1, 0.6)
        return direction, confidence, f"{self.variable_id}={value:.4f} (causal_w={causal_w:.3f})"
