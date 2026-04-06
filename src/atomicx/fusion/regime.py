"""Regime Detector — classifies current market regime.

The regime determines how signals are weighted and which strategies
are most appropriate. Detects:
- Trend direction (bullish/bearish)
- Trend strength (strong/weak)
- Volatility level (high/low/normal)
- Market phase (accumulation/markup/distribution/markdown)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """High-level market regime classification."""

    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT = "breakout"
    CAPITULATION = "capitulation"
    UNKNOWN = "unknown"


class RegimeState(BaseModel):
    """Current regime detection result."""

    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Component scores
    trend_direction: float = Field(
        default=0.0, description="+1 = strong bull, -1 = strong bear, 0 = no trend"
    )
    trend_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="ADX-based trend strength"
    )
    volatility_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=calm, 1=extreme"
    )
    volume_intensity: float = Field(
        default=0.0, ge=0.0, description="Relative volume vs average"
    )

    # Strategy recommendation
    recommended_strategy: str = "wait"  # trend_follow, mean_revert, breakout, wait
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegimeDetector:
    """Detects the current market regime from variable values."""

    def detect(self, variables: dict[str, float]) -> RegimeState:
        """Classify the current market regime.

        Uses ADX (trend strength), Bollinger Bandwidth (volatility),
        RSI (momentum), and Relative Volume.
        """
        adx = variables.get("ADX", 25)
        bb_bandwidth = variables.get("BB_BANDWIDTH", 0.05)
        rsi = variables.get("RSI_14", 50)
        rel_volume = variables.get("REL_VOLUME", 1.0)
        ema_9 = variables.get("EMA_9", 0)
        ema_21 = variables.get("EMA_21", 0)
        ema_50 = variables.get("EMA_50", 0)

        # FIX: Anti-lag mechanism using price velocity
        # Check if price moved >2% in opposite direction of EMA-based trend
        price = variables.get("PRICE", 0)
        recent_prices = variables.get("RECENT_PRICES", [])

        # Trend direction from EMA stack (lagging indicator)
        if ema_9 > ema_21 > ema_50 and ema_50 > 0:
            trend_dir = 1.0
        elif ema_9 < ema_21 < ema_50 and ema_50 > 0:
            trend_dir = -1.0
        else:
            trend_dir = 0.0

        # Override with price velocity if recent price action contradicts EMAs
        if price > 0 and len(recent_prices) >= 15:
            price_15_ago = recent_prices[-15] if len(recent_prices) >= 15 else price
            price_change_pct = (price - price_15_ago) / price_15_ago

            # If price pumped >2% but EMAs say bearish → early reversal
            if price_change_pct > 0.02 and trend_dir < 0:
                trend_dir = 0.5  # Weak bullish (reversal in progress)
            # If price dumped >2% but EMAs say bullish → early reversal
            elif price_change_pct < -0.02 and trend_dir > 0:
                trend_dir = -0.5  # Weak bearish (reversal in progress)

        # Trend strength (normalize ADX to 0-1)
        trend_strength = min(adx / 50, 1.0)

        # FIX: Add ADX slope to detect weakening trends early
        # If ADX is falling, trend is losing strength even if ADX value is high
        adx_slope = 0.0
        if len(recent_prices) >= 20:
            # Approximate ADX slope from recent price volatility changes
            recent_range = max(recent_prices[-10:]) - min(recent_prices[-10:])
            prev_range = max(recent_prices[-20:-10]) - min(recent_prices[-20:-10])
            if prev_range > 0:
                range_change = (recent_range - prev_range) / prev_range
                adx_slope = range_change  # Positive = expanding range, Negative = contracting

        # Volatility (normalize bandwidth)
        vol_level = min(bb_bandwidth / 0.1, 1.0)

        # Volume intensity
        vol_intensity = rel_volume

        # Classify regime
        regime, strategy, confidence = self._classify(
            trend_dir, trend_strength, vol_level, vol_intensity, rsi, adx_slope
        )

        regime_state = RegimeState(
            regime=regime,
            confidence=confidence,
            trend_direction=trend_dir,
            trend_strength=trend_strength,
            volatility_level=vol_level,
            volume_intensity=vol_intensity,
            recommended_strategy=strategy,
            metadata={
                "adx_slope": adx_slope,
                "rsi": rsi,
                "price_velocity_15": (price - recent_prices[-15]) / recent_prices[-15] if len(recent_prices) >= 15 and recent_prices[-15] > 0 else 0,
            }
        )

        # FIX: Add debug logging for regime detection
        logger.debug(
            f"[REGIME-DETECTION] trend_dir={trend_dir:.2f}, strength={trend_strength:.2f}, "
            f"adx_slope={adx_slope:.2f}, rsi={rsi:.1f} → {regime.value} (conf={confidence:.0%})"
        )

        return regime_state

    def _classify(
        self,
        trend_dir: float,
        trend_strength: float,
        vol_level: float,
        vol_intensity: float,
        rsi: float,
        adx_slope: float = 0.0,
    ) -> tuple[MarketRegime, str, float]:
        """Rule-based regime classification with ADX slope consideration."""

        # Capitulation: extreme RSI + high volume + bearish
        if rsi < 20 and vol_intensity > 2.5 and trend_dir < 0:
            return MarketRegime.CAPITULATION, "wait", 0.8

        # Breakout: high volume + expanding volatility + trending
        if vol_intensity > 2.0 and vol_level > 0.6 and trend_strength > 0.5:
            return MarketRegime.BREAKOUT, "breakout", 0.7

        # FIX: Adjust trend strength based on ADX slope
        # If ADX is declining (slope < -0.2), reduce effective trend strength
        effective_trend_strength = trend_strength
        if adx_slope < -0.2:
            effective_trend_strength *= 0.7  # Reduce by 30% if trend weakening
            logger.debug(f"[REGIME] ADX declining (slope={adx_slope:.2f}), reducing trend strength")

        # Strong trend (but check if weakening)
        if effective_trend_strength > 0.6:
            if trend_dir > 0:
                return MarketRegime.TRENDING_BULLISH, "trend_follow", 0.75
            elif trend_dir < 0:
                # FIX: For bearish trends, be more conservative
                # Only classify as strong bearish if RSI confirms (not overbought)
                if rsi < 60:  # RSI confirms bearish momentum
                    return MarketRegime.TRENDING_BEARISH, "trend_follow", 0.75
                else:
                    # Potential false bearish signal (RSI doesn't confirm)
                    return MarketRegime.RANGING_HIGH_VOL, "mean_revert", 0.5

        # Range: weak trend
        if effective_trend_strength < 0.4:
            if vol_level > 0.6:
                return MarketRegime.RANGING_HIGH_VOL, "mean_revert", 0.6
            else:
                return MarketRegime.RANGING_LOW_VOL, "mean_revert", 0.65

        # Moderate trend
        if trend_dir > 0:
            return MarketRegime.TRENDING_BULLISH, "trend_follow", 0.55
        elif trend_dir < 0:
            # FIX: Additional check for moderate bearish - require RSI confirmation
            if rsi < 55:
                return MarketRegime.TRENDING_BEARISH, "trend_follow", 0.55
            else:
                # RSI doesn't support bearish bias, classify as ranging
                return MarketRegime.RANGING_HIGH_VOL, "mean_revert", 0.45

        return MarketRegime.UNKNOWN, "wait", 0.3


# ── Regime-Adaptive Weights ──────────────────────────────────


REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.TRENDING_BULLISH: {
        "technical": 1.2,
        "microstructure": 0.8,
        "macro": 1.0,
    },
    MarketRegime.TRENDING_BEARISH: {
        "technical": 1.2,
        "microstructure": 0.9,
        "macro": 1.0,
    },
    MarketRegime.RANGING_HIGH_VOL: {
        "technical": 0.8,
        "microstructure": 1.3,
        "macro": 0.9,
    },
    MarketRegime.RANGING_LOW_VOL: {
        "technical": 1.0,
        "microstructure": 0.7,
        "macro": 1.1,
    },
    MarketRegime.BREAKOUT: {
        "technical": 1.0,
        "microstructure": 1.4,
        "macro": 0.8,
    },
    MarketRegime.CAPITULATION: {
        "technical": 0.6,
        "microstructure": 1.5,
        "macro": 0.7,
    },
    MarketRegime.UNKNOWN: {
        "technical": 0.8,
        "microstructure": 0.8,
        "macro": 0.8,
    },
}
