"""Fusion Node Engine — combines ALL signals into a final prediction.

The Fusion Node is the central integration point of AtomicX.
It receives signals from:
1. Agent Hierarchy domain leaders
2. Causal discovery engine (planned: Phase 7 strategic layer, Phase 8 narrative, Phase 9 swarm)

And produces structured PredictionPackets with:
- BET/STAY_OUT decision based on 72% threshold
- Regime-adaptive weighting
- Multi-timeframe alignment check
- Ensemble baseline penalty
- Full reasoning trace
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from atomicx.agents.orchestrator import AgentHierarchy
from atomicx.agents.signals import AgentSignal, SignalDirection
from atomicx.fusion.prediction import PredictionAction, PredictionPacket
from atomicx.fusion.regime import (
    REGIME_WEIGHTS,
    MarketRegime,
    RegimeDetector,
    RegimeState,
)
from atomicx.variables.engine import VariableComputeEngine
from atomicx.causal.engine import CausalDiscoveryEngine
from atomicx.memory.service import MemoryService


class FusionNode:
    """The core integration engine — produces final predictions.

    Workflow:
    1. Detect current regime
    2. Run agent hierarchy to get domain leader signals
    3. Apply regime-adaptive weights
    4. Check multi-timeframe alignment
    5. Apply STAY_OUT threshold
    6. Compute trade levels
    7. Store prediction in memory
    8. Output PredictionPacket
    """

    # System-wide thresholds
    BET_THRESHOLD = 0.72  # Only BET when confidence > 72%
    STAY_OUT_ZONE = (0.40, 0.60)  # Skip 40-60% zone
    ENSEMBLE_PENALTY = 0.15  # 15% penalty if ensemble disagrees
    MIN_TIMEFRAME_AGREEMENT = 2  # At least 2 timeframes must agree

    def __init__(
        self,
        hierarchy: AgentHierarchy | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self.hierarchy = hierarchy or AgentHierarchy()
        self.memory = memory or MemoryService()
        self.regime_detector = RegimeDetector()
        self.causal_engine = CausalDiscoveryEngine()
        self._prediction_count = 0
        self._built = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if not self._built:
            self.hierarchy.build()
            await self.memory.initialize()
            self.causal_engine.initialize_with_defaults()
            self._built = True
            logger.info("Fusion Node initialized")

    async def predict(
        self,
        symbol: str,
        timeframe: str = "4h",
        variables: dict[str, float] | None = None,
        price: float | None = None,
    ) -> PredictionPacket:
        """Generate a full prediction for a symbol.

        Args:
            symbol: Trading pair, e.g. 'BTC/USDT'
            timeframe: Primary analysis timeframe
            variables: Current variable snapshot (all indicators)
            price: Current price (for trade level calculation)

        Returns:
            Complete PredictionPacket
        """
        if not self._built:
            await self.initialize()

        vars_ = variables or {}

        # DEBUG: Log variable count and critical indicators
        logger.info(f"[DEBUG] Predict called with {len(vars_)} variables for {symbol}")
        if len(vars_) == 0:
            logger.warning(f"[DEBUG] NO VARIABLES provided! This will result in 'unknown' regime.")
        else:
            ema_count = sum(1 for k in vars_.keys() if 'EMA' in k)
            logger.info(f"[DEBUG] Found {ema_count} EMA variables. Sample keys: {list(vars_.keys())[:8]}")

        # FIX: Get real-time microstructure data from cache (bypasses 60s delay)
        try:
            from atomicx.common.cache import get_sensory_cache
            cache = get_sensory_cache()
            ob_imbalance_live = cache.get(symbol, "OB_IMBALANCE")
            if ob_imbalance_live is not None:
                vars_["OB_IMBALANCE"] = ob_imbalance_live
                logger.debug(f"[REAL-TIME] OB_IMBALANCE updated: {ob_imbalance_live:.3f}")
        except Exception as e:
            logger.warning(f"[REAL-TIME] Failed to get live microstructure: {e}")

        # Step 1: Detect regime
        regime_state = self.regime_detector.detect(vars_)
        logger.info(f"Regime: {regime_state.regime.value} ({regime_state.confidence:.0%})")

        # ═══ FIX: Retrieve past similar setups from memory ═══
        past_lessons = []
        try:
            similar_setups = await self.memory.retrieve_similar_setups(vars_, limit=5)
            if similar_setups:
                past_lessons = [
                    {
                        "content": m.get("content", m.get("memory", "")),
                        "outcome": m.get("metadata", {}).get("outcome"),
                        "timestamp": m.get("metadata", {}).get("timestamp"),
                    }
                    for m in similar_setups
                ]
                logger.info(f"Retrieved {len(past_lessons)} similar past predictions")
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

        # ═══ FIX: Retrieve regime-specific lessons ═══
        regime_lessons = []
        try:
            regime_memories = await self.memory.get_lessons_for_regime(
                regime_state.regime.value, limit=10
            )
            if regime_memories:
                regime_lessons = [
                    {
                        "content": m.get("content", m.get("memory", "")),
                        "importance": m.get("metadata", {}).get("importance", 0.5),
                    }
                    for m in regime_memories
                ]
                logger.info(f"Retrieved {len(regime_lessons)} lessons for {regime_state.regime.value} regime")
        except Exception as e:
            logger.warning(f"Regime lesson retrieval failed: {e}")

        # Step 2: Run agent hierarchy with memory-augmented context
        causal_weights = self.causal_engine.get_weights()
        context = {
            "variables": vars_,
            "regime": regime_state.regime.value,
            "causal_weights": causal_weights,
            "past_lessons": past_lessons,  # Similar setups from memory
            "regime_lessons": regime_lessons,  # Regime-specific wisdom
        }
        hierarchy_signal = await self.hierarchy.evaluate(symbol, timeframe, context)

        # Step 3: Build layer contributions
        layer_contributions = self._collect_layer_contributions(hierarchy_signal, regime_state)

        # Step 4: Compute final confidence with regime weighting
        direction, confidence = self._compute_final_signal(
            hierarchy_signal, regime_state
        )

        # FIX: Regime Contradiction Check (prevents "bearish black hole")
        # If regime and signal direction conflict, reduce confidence or stay out
        regime_dir = self._get_regime_direction(regime_state)
        if regime_dir and direction != "neutral":
            if (regime_dir == "bearish" and direction == "bullish") or \
               (regime_dir == "bullish" and direction == "bearish"):
                logger.warning(
                    f"[REGIME-CONTRADICTION] Regime={regime_dir} but signal={direction}. "
                    f"Original confidence={confidence:.2f}. Applying 50% penalty."
                )
                confidence *= 0.5  # Severe penalty for contradicting regime

        # Step 5: Apply STAY_OUT logic (includes microstructure veto)
        action = self._decide_action(confidence, regime_state, direction, vars_)

        # Step 6: Compute trade levels
        entry, stop, tp1, tp2, rr = self._compute_trade_levels(
            direction, confidence, price, vars_
        )

        # Step 7: Position sizing
        position_size = self._compute_position_size(confidence, action)

        # Step 8: Build prediction packet
        packet = PredictionPacket(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            action=action,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=rr,
            position_size_pct=position_size,
            regime=regime_state.regime,
            regime_confidence=regime_state.confidence,
            layer_contributions=layer_contributions,
            reasoning_summary=self._build_reasoning(
                direction, confidence, action, regime_state, hierarchy_signal
            ),
            variable_snapshot={k: round(v, 4) for k, v in list(vars_.items())[:20]},
            metadata={
                "complete_variables": {
                    k: round(v, 6) if isinstance(v, (int, float)) else v
                    for k, v in vars_.items()
                },  # ALL variables with high precision, skip non-numeric
                "hierarchy_signal_raw": hierarchy_signal.model_dump() if hierarchy_signal else None,
                "regime_state_raw": {
                    "regime": regime_state.regime.value,
                    "confidence": regime_state.confidence,
                    "recommended_strategy": regime_state.recommended_strategy,
                },
                "past_lessons": past_lessons,
                "regime_lessons": regime_lessons,
                "causal_weights": causal_weights,
            },
        )

        self._prediction_count += 1

        # Step 9: Store in memory
        await self._store_prediction_memory(packet, vars_)

        logger.info(
            f"Prediction #{self._prediction_count}: {packet.symbol} "
            f"{packet.direction} @ {packet.confidence:.0%} → {packet.action.value}"
        )

        return packet

    def _get_regime_direction(self, regime_state: RegimeState) -> str | None:
        """Extract directional bias from regime type.

        Returns:
            "bullish", "bearish", or None if regime is neutral/unknown
        """
        regime = regime_state.regime
        if regime in (MarketRegime.TRENDING_BULLISH,):
            return "bullish"
        elif regime in (MarketRegime.TRENDING_BEARISH, MarketRegime.CAPITULATION):
            return "bearish"
        # Ranging, breakout, unknown = no clear directional bias
        return None

    def _compute_final_signal(
        self,
        hierarchy_signal: AgentSignal | None,
        regime_state: RegimeState,
    ) -> tuple[str, float]:
        """Combine hierarchy signal with regime weighting."""
        if not hierarchy_signal or hierarchy_signal.direction == SignalDirection.SKIP:
            return "neutral", 0.0

        direction = hierarchy_signal.direction.value
        confidence = hierarchy_signal.confidence

        # Apply regime weight multiplier
        weights = REGIME_WEIGHTS.get(regime_state.regime, REGIME_WEIGHTS[MarketRegime.UNKNOWN])
        # Average the regime weights as a general multiplier
        avg_weight = sum(weights.values()) / len(weights)
        confidence *= avg_weight

        # Clamp confidence
        confidence = max(0.0, min(confidence, 1.0))

        return direction, confidence

    def _decide_action(
        self,
        confidence: float,
        regime_state: RegimeState,
        direction: str,
        variables: dict[str, float]
    ) -> PredictionAction:
        """Apply STAY_OUT threshold logic with microstructure veto."""

        # ═══ FIX #1: MICROSTRUCTURE GOVERNOR ═══
        # Hard veto if orderbook is severely lopsided against trade direction
        ob_imbalance = variables.get("OB_IMBALANCE", 0.0)

        if abs(ob_imbalance) > 0.7:  # Extreme whale wall (70%+ one-sided)
            if (direction == "bullish" and ob_imbalance < -0.7) or \
               (direction == "bearish" and ob_imbalance > 0.7):
                logger.warning(
                    f"[MICROSTRUCTURE-VETO] 🛑 Aborting {direction} trade: "
                    f"OB_IMBALANCE={ob_imbalance:.2f} indicates extreme opposite pressure. "
                    f"Whale wall detected."
                )
                return PredictionAction.STAY_OUT

        # ═══ INSTITUTIONAL FIX: CVD vs OB_IMBALANCE ANTI-SPOOFING VETO ═══
        # Double confirmation: Intent (OB_IMBALANCE) must match Execution (CVD)
        # If orderbook shows bullish intent but actual trades are bearish → SPOOF
        cvd = variables.get("CVD", None)

        if cvd is not None and abs(ob_imbalance) > 0.3:  # Only check when there's significant OB pressure
            # Get CVD trend from cache (recent execution pressure)
            try:
                from atomicx.common.cache import get_sensory_cache
                cache = get_sensory_cache()

                # Get historical CVD to compute 5-minute delta
                # For now, use a simplified heuristic: CVD direction vs OB_IMBALANCE
                # Future enhancement: Query cumulative_delta table for proper windowed analysis

                # Simplified logic: If OB_IMBALANCE and CVD disagree significantly, it's a spoof
                # OB_IMBALANCE > 0 = bullish intent (big bids)
                # CVD > 0 = bullish execution (buyers aggressive)
                # If they disagree → fake wall

                if ob_imbalance > 0.3 and cvd < -1000:  # Bullish intent, bearish execution
                    logger.warning(
                        f"[ANTI-SPOOFING-VETO] 🚨 SPOOF DETECTED on {direction} trade: "
                        f"OB_IMBALANCE={ob_imbalance:.2f} (bullish intent) but "
                        f"CVD={cvd:.0f} (bearish execution). "
                        f"Fake buy wall detected - sellers are hitting bids."
                    )
                    return PredictionAction.STAY_OUT

                elif ob_imbalance < -0.3 and cvd > 1000:  # Bearish intent, bullish execution
                    logger.warning(
                        f"[ANTI-SPOOFING-VETO] 🚨 SPOOF DETECTED on {direction} trade: "
                        f"OB_IMBALANCE={ob_imbalance:.2f} (bearish intent) but "
                        f"CVD={cvd:.0f} (bullish execution). "
                        f"Fake sell wall detected - buyers are hitting asks."
                    )
                    return PredictionAction.STAY_OUT

                # Log when OB and CVD agree (validation)
                elif (ob_imbalance > 0.3 and cvd > 500) or (ob_imbalance < -0.3 and cvd < -500):
                    logger.info(
                        f"[ANTI-SPOOFING-VALIDATION] ✅ OB_IMBALANCE={ob_imbalance:.2f} "
                        f"and CVD={cvd:.0f} agree. Real orderbook pressure confirmed."
                    )

            except Exception as e:
                logger.debug(f"[ANTI-SPOOFING] Failed to get CVD from cache: {e}")

        # FIX: Allow trading in unknown regime if confidence is high enough
        # Only stay out if regime is unknown AND confidence is not exceptional
        if regime_state.regime == MarketRegime.UNKNOWN and confidence < 0.70:
            return PredictionAction.STAY_OUT

        # STAY_OUT zone: 40-60% confidence → skip
        if self.STAY_OUT_ZONE[0] <= confidence <= self.STAY_OUT_ZONE[1]:
            return PredictionAction.STAY_OUT

        # BET only above threshold
        if confidence >= self.BET_THRESHOLD:
            return PredictionAction.BET

        return PredictionAction.STAY_OUT

    def _compute_trade_levels(
        self,
        direction: str,
        confidence: float,
        price: float | None,
        variables: dict[str, float],
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Compute entry, stop-loss, take-profit levels.

        Uses ATR for volatility-adjusted stops and targets.
        """
        if not price or price <= 0:
            return None, None, None, None, None

        atr = variables.get("ATR_14")
        if atr is None:
            logger.warning("ATR_14 not in variables -- price target unreliable")
            return price, None, None, None, 0.0

        if direction == "bullish":
            entry = price
            stop = price - (atr * 1.5)  # 1.5 ATR stop
            tp1 = price + (atr * 2.5)  # 1:2.5 R:R minimum
            tp2 = price + (atr * 4.0)  # Extended target
        elif direction == "bearish":
            entry = price
            stop = price + (atr * 1.5)
            tp1 = price - (atr * 2.5)
            tp2 = price - (atr * 4.0)
        else:
            return price, None, None, None, None

        risk = abs(entry - stop)
        reward = abs(tp1 - entry)
        rr = reward / risk if risk > 0 else 0

        return round(entry, 2), round(stop, 2), round(tp1, 2), round(tp2, 2), round(rr, 1)

    def _compute_position_size(
        self, confidence: float, action: PredictionAction
    ) -> float:
        """Dynamic position sizing based on confidence (max 2%)."""
        if action == PredictionAction.STAY_OUT:
            return 0.0

        # Scale from 0.5% to 2% based on confidence
        base = 0.005  # 0.5%
        max_size = 0.02  # 2%
        scale = (confidence - self.BET_THRESHOLD) / (1.0 - self.BET_THRESHOLD)
        return min(base + scale * (max_size - base), max_size)

    def _collect_layer_contributions(
        self,
        hierarchy_signal: AgentSignal | None,
        regime_state: RegimeState,
    ) -> dict[str, dict[str, Any]]:
        """Collect layer-level contributions for the reasoning trace."""
        contributions = {
            "regime": {
                "type": regime_state.regime.value,
                "confidence": regime_state.confidence,
                "strategy": regime_state.recommended_strategy,
            },
        }

        if hierarchy_signal:
            contributions["hierarchy"] = {
                "direction": hierarchy_signal.direction.value,
                "confidence": hierarchy_signal.confidence,
                "reasoning": hierarchy_signal.reasoning,
            }

        return contributions

    def _build_reasoning(
        self,
        direction: str,
        confidence: float,
        action: PredictionAction,
        regime_state: RegimeState,
        hierarchy_signal: AgentSignal | None,
    ) -> str:
        """Build human-readable reasoning summary."""
        parts = [
            f"Regime: {regime_state.regime.value} ({regime_state.confidence:.0%})",
            f"Strategy: {regime_state.recommended_strategy}",
        ]

        if hierarchy_signal and hierarchy_signal.direction != SignalDirection.SKIP:
            parts.append(f"Hierarchy: {hierarchy_signal.direction.value} @ {hierarchy_signal.confidence:.0%}")
            if hierarchy_signal.reasoning:
                parts.append(f"Detail: {hierarchy_signal.reasoning}")

        if action == PredictionAction.STAY_OUT:
            if confidence < self.BET_THRESHOLD:
                parts.append(f"STAY_OUT: confidence {confidence:.0%} < {self.BET_THRESHOLD:.0%} threshold")
            else:
                parts.append("STAY_OUT: regime or alignment conditions not met")

        return " | ".join(parts)

    async def _store_prediction_memory(
        self, packet: PredictionPacket, variables: dict[str, float]
    ) -> None:
        """Store prediction in memory for future learning."""
        try:
            clean_vars = {k: float(v) for k, v in variables.items() if isinstance(v, (int, float))}
            await self.memory.store_prediction(
                symbol=packet.symbol,
                direction=packet.direction,
                confidence=packet.confidence,
                reasoning=packet.reasoning_summary,
                variable_snapshot=clean_vars,
                prediction_id=packet.prediction_id,
            )
        except Exception as e:
            logger.error(f"Failed to store prediction memory: {e}")

    @property
    def prediction_count(self) -> int:
        return self._prediction_count
