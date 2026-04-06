"""Decider Core — The Executive Function.

Translates the output of the DebateChamber and the Meta-Orchestrator's
sensory state into a high-level INTENT (e.g., 'quiet_accumulation',
'defensive_hedge', 'full_exit').
"""

from __future__ import annotations

from typing import Any
from loguru import logger
from pydantic import BaseModel

from atomicx.brain.debate import DebateSummary


class DecisionIntent(BaseModel):
    """The formalized intent of the Brain."""
    intent_type: str  # e.g., "quiet_accumulation", "trend_follow", "defensive_hedge"
    direction: str    # "bullish", "bearish", "stay_out"
    conviction: float
    reasoning: str
    expected_rr: float = 2.0  # Risk to Reward target
    
    def is_actionable(self) -> bool:
        return self.intent_type not in ("stay_out", "neutral", "observe")


class DeciderCore:
    """The Executive Function of the Brain."""
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="brain.decider")

    def decide(self, brain_state: dict[str, Any], debate: DebateSummary) -> DecisionIntent:
        """Formulate a high-level intent based on the debate and current constraints."""
        regime = brain_state.get("variables", {}).get("REGIME", "unknown")
        
        # 1. Immediate Stay Out / Veto evaluation
        if debate.dominant_stance == "stay_out":
            return DecisionIntent(
                intent_type="observe_only",
                direction="stay_out",
                conviction=debate.overall_conviction,
                reasoning=f"Debate resulted in hard veto: {debate.synthesis}. Action aborted."
            )
            
        if debate.conflict_detected:
            # Reduce conviction but don't abort unless it drops too low
            penalized = debate.overall_conviction * 0.6
            if penalized < 0.40:
                return DecisionIntent(
                    intent_type="observe_only",
                    direction="stay_out",
                    conviction=penalized,
                    reasoning=f"Conflict detected, conviction too low after penalty: {penalized:.0%}. {debate.synthesis}"
                )
            # Otherwise continue with penalized conviction
            debate.overall_conviction = penalized
            self.logger.warning(f"CONFLICT PENALTY APPLIED: New conviction {penalized:.0%}")
            
        # 2. Determine Intent Archetype based on Regime and Stance
        intent_type = "trend_follow"
        reasoning = debate.synthesis
        
        if regime in ("range_bound", "low_volatility"):
            intent_type = "quiet_accumulation"
            reasoning = f"Low volatility regime detected. Shifting to {intent_type}. {debate.synthesis}"
        elif regime in ("high_volatility", "capitulation"):
            # If the market is crashing but we want to bet, it's a defensive/contrarian hedge
            if debate.dominant_stance == "bullish":
                intent_type = "contrarian_hedge"
                reasoning = f"High volatility dump detected, but debate forces bullish entry. This is a {intent_type}. Expect high slippage."
            else:
                intent_type = "momentum_short"
                
        # 3. Compile intent
        intent = DecisionIntent(
            intent_type=intent_type,
            direction=debate.dominant_stance,
            conviction=debate.overall_conviction,
            reasoning=reasoning,
            expected_rr=2.5 if intent_type == "contrarian_hedge" else 2.0
        )
        
        self.logger.success(f"DECISION REACHED [Intent: {intent.intent_type.upper()}] | Direction: {intent.direction.upper()} | Conviction: {intent.conviction:.0%}")
        return intent
