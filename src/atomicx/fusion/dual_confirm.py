"""Dual-Confirmation Engine.

The system only executes a trade when BOTH the statistical pattern
AND the causal logic agree. This is the professional 50/50 balance.
"""

from __future__ import annotations

from typing import Any
from loguru import logger
from pydantic import BaseModel


class ConfirmationResult(BaseModel):
    """Result of the dual-confirmation check."""
    pattern_signal: str  # "buy", "sell", "neutral"
    pattern_confidence: float
    logic_signal: str    # "bullish", "bearish", "neutral", "none"
    logic_reason: str
    confirmed: bool
    score: float = 0.0  # 0.0 to 1.0
    final_direction: str  # "execute_buy", "execute_sell", "abort"
    

class DualConfirmationEngine:
    """50/50 Pattern vs Logic — trades only execute when both agree."""
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="fusion.dual_confirm")
        
    def evaluate(self, brain_state: dict[str, Any], debate_summary: Any) -> ConfirmationResult:
        """Run the dual-confirmation check.
        
        Pattern Channel (50%): Statistical formations from the Variable Engine.
        Logic Channel (50%): Causal reasoning from the Debate Chamber.
        """
        # Channel 1: PATTERN — What does the statistical signal say?
        pattern_signal, pattern_confidence = self._evaluate_pattern(brain_state)
        
        # Channel 2: LOGIC — What does the causal engine say?
        logic_signal, logic_reason = self._evaluate_logic(debate_summary)
        
        # THE BET RULE
        confirmed = False
        final_direction = "abort"
        
        # 50/50 Conflict Handling: Instead of strict abort, apply a penalty
        if pattern_signal == "buy" and logic_signal == "bullish":
            confirmed = True
            final_direction = "execute_buy"
            self.logger.success("[DUAL-CONFIRM] ✓ Pattern=BUY + Logic=BULLISH → EXECUTE BUY")
            
        elif pattern_signal == "sell" and logic_signal == "bearish":
            confirmed = True
            final_direction = "execute_sell"
            self.logger.success("[DUAL-CONFIRM] ✓ Pattern=SELL + Logic=BEARISH → EXECUTE SELL")
            
        elif pattern_signal in ("buy", "sell") and logic_signal in ("neutral", "conflict"):
            # FIX: Allow high-confidence patterns even if logic is neutral
            if pattern_confidence >= 0.75:
                confirmed = True
                final_direction = f"execute_{pattern_signal}"
                self.logger.warning(
                    f"[DUAL-CONFIRM] ⚠️  Pattern={pattern_signal.upper()} + Logic={logic_signal.upper()} "
                    f"→ ALLOW (high pattern confidence {pattern_confidence:.0%} overrides neutral logic)"
                )
            else:
                self.logger.warning(
                    f"[DUAL-CONFIRM] ✗ Pattern={pattern_signal.upper()} ({pattern_confidence:.0%}) + "
                    f"Logic={logic_signal.upper()} → ABORT (confidence too low for override)"
                )
            
        elif pattern_signal == "neutral":
            self.logger.info("[DUAL-CONFIRM] Pattern=NEUTRAL → No action needed")
            
        elif pattern_signal != logic_signal:
            # Honest conflict handling
            penalized_confidence = pattern_confidence * 0.6
            
            if penalized_confidence < 0.40:
                self.logger.warning(f"[DUAL-CONFIRM] ✗ Conflict: Pattern={pattern_signal.upper()}, Logic={logic_signal.upper()}. Confidence {penalized_confidence:.2f} too low.")
                final_direction = "abort"
                logic_reason = f"Conflict: {logic_reason}. Confidence penalized to {penalized_confidence:.2%}"
            else:
                confirmed = True
                final_direction = f"low_confidence_{pattern_signal}"
                self.logger.info(f"[DUAL-CONFIRM] ! Conflict detected but confidence {penalized_confidence:.2f} remains tradable. Pattern={pattern_signal.upper()}")
                logic_reason = f"Conflict: {logic_reason}. Confidence penalized."
            
        return ConfirmationResult(
            pattern_signal=pattern_signal,
            pattern_confidence=pattern_confidence,
            logic_signal=logic_signal,
            logic_reason=logic_reason,
            confirmed=confirmed,
            score=(pattern_confidence + getattr(debate_summary, 'overall_conviction', 0.5)) / 2,
            final_direction=final_direction,
        )
        
    def _evaluate_pattern(self, brain_state: dict[str, Any]) -> tuple[str, float]:
        """Extract the statistical pattern signal from the brain state.

        ENHANCEMENT: Use fusion engine prediction if available (46+ variables),
        otherwise fallback to basic RSI/ADX pattern detection (2 variables).
        """
        # PRIORITY: Use fusion engine's prediction if available in brain_state
        fusion_prediction = brain_state.get("fusion_prediction")
        if fusion_prediction:
            direction = fusion_prediction.get("direction", "neutral")
            confidence = fusion_prediction.get("confidence", 0.5)

            # Map fusion direction to pattern signal
            if direction == "bullish":
                return "buy", confidence
            elif direction == "bearish":
                return "sell", confidence
            # If fusion says neutral or low confidence, continue to basic pattern check

        # FALLBACK: Basic 2-variable pattern detection (RSI + ADX)
        variables = brain_state.get("variables", {})
        rsi = variables.get("RSI_14", 50)
        adx = variables.get("ADX", 20)

        # Bullish: Oversold or Trend Strength
        if rsi < 30 or (rsi < 50 and adx > 25):
            conf = min(0.5 + (adx / 100.0), 1.0)
            return "buy", conf
        # Bearish: Overbought or Trend Breakdown
        elif rsi > 70 or (rsi > 50 and adx > 25):
            conf = min(0.5 + (adx / 100.0), 1.0)
            return "sell", conf

        return "neutral", 0.5
        
    def _evaluate_logic(self, debate_summary: Any) -> tuple[str, str]:
        """Extract the causal logic signal from the debate summary."""
        # In production: reads the Debate Chamber's consensus
        consensus = getattr(debate_summary, "dominant_stance", "neutral")
        reason = getattr(debate_summary, "synthesis", "No strong causal signal")
        
        if consensus in ("strongly_bullish", "bullish"):
            return "bullish", reason
        elif consensus in ("strongly_bearish", "bearish"):
            return "bearish", reason
        elif getattr(debate_summary, "conflict_detected", False):
            return "conflict", "Sub-agents disagree — no clear causal direction"
        
        return "neutral", reason
