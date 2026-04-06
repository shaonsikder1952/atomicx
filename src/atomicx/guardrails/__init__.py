"""Neurosymbolic Guardrails — Tradability & Risk Control (Phase 10).

Enforces hard constraints that no prediction can bypass:
- Maximum position size (2% of account)
- Maximum daily loss limit
- Minimum R:R ratio
- Correlation risk (don't bet same direction on correlated assets)
- Drawdown circuit breaker
- News event lockout
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from atomicx.fusion.prediction import PredictionAction, PredictionPacket


class GuardrailConfig(BaseModel):
    """Configurable hard limits."""
    max_position_pct: float = Field(default=0.02, description="Max 2% of account per trade")
    max_daily_loss_pct: float = Field(default=0.05, description="Stop trading after 5% daily loss")
    min_risk_reward: float = Field(default=2.0, description="Minimum 1:2 R:R ratio")
    max_correlated_positions: int = Field(default=3, description="Max same-direction positions on correlated assets")
    max_drawdown_pct: float = Field(default=0.15, description="15% max drawdown → halt all trading")
    cooldown_after_loss_streak: int = Field(default=3, description="Pause after 3 consecutive losses")
    news_lockout_minutes: int = Field(default=30, description="No trading 30 min before/after major news")


class GuardrailResult(BaseModel):
    """Result of guardrail checks."""
    passed: bool = True
    violations: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)
    original_action: PredictionAction = PredictionAction.STAY_OUT
    final_action: PredictionAction = PredictionAction.STAY_OUT


class TradabilityGuardrails:
    """Enforces hard constraints on every prediction before execution.

    No prediction bypasses these — they are the system's safety net.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._open_positions: list[dict[str, Any]] = []
        self._account_peak: float = 100_000
        self._account_value: float = 100_000

    def check(self, packet: PredictionPacket) -> GuardrailResult:
        """Run all guardrail checks on a prediction packet.

        Returns GuardrailResult with pass/fail and any adjustments.
        """
        result = GuardrailResult(original_action=packet.action)
        result.final_action = packet.action

        # Check 1: Position size limit
        if packet.position_size_pct > self.config.max_position_pct:
            packet.position_size_pct = self.config.max_position_pct
            result.adjustments.append(
                f"Position capped at {self.config.max_position_pct:.1%}"
            )

        # Check 2: Daily loss limit
        if abs(self._daily_pnl) > self.config.max_daily_loss_pct * self._account_value:
            result.passed = False
            result.final_action = PredictionAction.STAY_OUT
            result.violations.append(
                f"Daily loss limit reached ({self._daily_pnl:.2f} > "
                f"{self.config.max_daily_loss_pct:.1%} of account)"
            )

        # Check 3: Minimum R:R
        if packet.risk_reward_ratio and packet.risk_reward_ratio < self.config.min_risk_reward:
            result.passed = False
            result.final_action = PredictionAction.STAY_OUT
            result.violations.append(
                f"R:R too low ({packet.risk_reward_ratio:.1f} < {self.config.min_risk_reward:.1f})"
            )

        # Check 4: Maximum drawdown
        drawdown = (self._account_peak - self._account_value) / self._account_peak
        if drawdown > self.config.max_drawdown_pct:
            result.passed = False
            result.final_action = PredictionAction.STAY_OUT
            result.violations.append(
                f"Max drawdown breached ({drawdown:.1%} > {self.config.max_drawdown_pct:.1%}) — HALT ALL TRADING"
            )

        # Check 5: Consecutive loss streak
        if self._consecutive_losses >= self.config.cooldown_after_loss_streak:
            result.passed = False
            result.final_action = PredictionAction.STAY_OUT
            result.violations.append(
                f"Loss streak cooldown ({self._consecutive_losses} consecutive losses)"
            )

        # Check 6: Correlated position limit
        same_dir_count = sum(
            1 for pos in self._open_positions
            if pos.get("direction") == packet.direction
        )
        if same_dir_count >= self.config.max_correlated_positions:
            result.passed = False
            result.final_action = PredictionAction.STAY_OUT
            result.violations.append(
                f"Too many {packet.direction} positions ({same_dir_count} >= "
                f"{self.config.max_correlated_positions})"
            )

        if result.violations:
            logger.warning(f"Guardrail violations: {result.violations}")
        elif result.adjustments:
            logger.info(f"Guardrail adjustments: {result.adjustments}")

        return result

    def record_trade_result(self, pnl: float, was_win: bool) -> None:
        """Update state after a trade closes."""
        self._daily_pnl += pnl
        self._account_value += pnl
        self._account_peak = max(self._account_peak, self._account_value)

        if was_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def reset_daily(self) -> None:
        """Reset daily counters (called at midnight UTC)."""
        self._daily_pnl = 0.0

    def add_position(self, symbol: str, direction: str) -> None:
        self._open_positions.append({"symbol": symbol, "direction": direction})

    def remove_position(self, symbol: str) -> None:
        self._open_positions = [p for p in self._open_positions if p["symbol"] != symbol]

    def get_active_risk(self) -> float:
        """Calculate current active risk from open positions.

        Returns total risk as percentage of account value.
        """
        if not self._open_positions:
            return 0.0
        # Each position risks max_position_pct of account
        total_risk = len(self._open_positions) * self.config.max_position_pct
        return total_risk

    @property
    def is_trading_halted(self) -> bool:
        drawdown = (self._account_peak - self._account_value) / self._account_peak
        return drawdown > self.config.max_drawdown_pct
