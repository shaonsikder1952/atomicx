"""Prediction Packet — the structured output of the Fusion Node.

Every prediction is a complete, self-documenting packet that includes:
- Direction, confidence, and STAY_OUT/BET decision
- Entry, stop-loss, take-profit levels
- Risk sizing
- Full reasoning trace showing which layers contributed
- Regime context
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from atomicx.fusion.regime import MarketRegime, RegimeState


class PredictionAction(str, Enum):
    """The binary decision: BET or STAY_OUT."""

    BET = "BET"  # High confidence — take this trade
    STAY_OUT = "STAY_OUT"  # Uncertainty — skip this setup


class PredictionPacket(BaseModel):
    """Complete prediction output from the Fusion Node."""

    # Identity
    prediction_id: str = Field(default_factory=lambda: f"pred_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Core prediction
    symbol: str
    timeframe: str
    direction: str  # "bullish" or "bearish"
    confidence: float = Field(ge=0.0, le=1.0)
    action: PredictionAction = PredictionAction.STAY_OUT

    # Trade levels
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    risk_reward_ratio: float | None = None

    # Risk management
    position_size_pct: float = Field(
        default=0.0, ge=0.0, le=0.02,
        description="Position size as % of account (max 2%)",
    )

    # Regime context
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0

    # Reasoning trace
    layer_contributions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Each layer's direction, confidence, and reasoning",
    )
    reasoning_summary: str = ""

    # Multi-timeframe alignment
    timeframe_alignment: dict[str, str] = Field(
        default_factory=dict,
        description="Direction per timeframe (4H, 12H, 1D, 1W)",
    )
    alignment_count: int = Field(
        default=0,
        description="How many timeframes agree with the prediction",
    )

    # Validation
    ensemble_agrees: bool = Field(
        default=True,
        description="Whether XGBoost+LSTM ensemble agrees with the prediction",
    )
    ensemble_penalty_applied: bool = Field(
        default=False,
        description="Whether -15% penalty was applied for ensemble disagreement",
    )

    # Metadata
    variable_snapshot: dict[str, float] = Field(default_factory=dict)
    causal_chains_used: list[str] = Field(default_factory=list)
    memory_retrievals: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_summary(self) -> str:
        """Human-readable prediction summary."""
        emoji = "🟢" if self.direction == "bullish" else "🔴"
        action_emoji = "✅ BET" if self.action == PredictionAction.BET else "⛔ STAY OUT"

        lines = [
            f"{emoji} {self.symbol} — {self.direction.upper()} @ {self.confidence:.0%}",
            f"Action: {action_emoji}",
            f"Regime: {self.regime.value} ({self.regime_confidence:.0%})",
        ]

        if self.entry_price:
            lines.append(f"Entry: ${self.entry_price:,.2f}")
        if self.stop_loss:
            lines.append(f"Stop: ${self.stop_loss:,.2f}")
        if self.take_profit_1:
            lines.append(f"TP1: ${self.take_profit_1:,.2f}")
        if self.risk_reward_ratio:
            lines.append(f"R:R: 1:{self.risk_reward_ratio:.1f}")
        if self.position_size_pct > 0:
            lines.append(f"Position: {self.position_size_pct:.1%} of account")
        if not self.ensemble_agrees:
            lines.append("⚠️ Ensemble disagrees (-15% penalty applied)")

        lines.append(f"\nReasoning: {self.reasoning_summary}")

        return "\n".join(lines)
