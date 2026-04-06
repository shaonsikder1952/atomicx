"""Agent signal types — the data structures agents pass up the hierarchy.

Every agent produces an AgentSignal that flows UP to its parent:
  AtomicAgent → AgentSignal → GroupLeader → GroupSignal → ...
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    """Agent's directional view."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    SKIP = "skip"  # Agent has no signal for this query


class SignalConfidence(str, Enum):
    """Confidence tier."""

    HIGH = "high"  # >72% — enables BET
    MODERATE = "moderate"  # 55-72% — contributes to consensus
    LOW = "low"  # 40-55% — weak, discounted
    NO_SIGNAL = "no_signal"  # <40% — agent skipped


class AgentSignal(BaseModel):
    """Output from any agent in the hierarchy."""

    agent_id: str
    agent_type: str  # atomic, group_leader, super_group, verification, common_sense, domain_leader
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Signal
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0, description="Raw confidence 0-1")
    confidence_tier: SignalConfidence = SignalConfidence.NO_SIGNAL

    # Context
    symbol: str = ""
    timeframe: str = ""
    variable_id: str | None = None  # Only for atomic agents

    # Reasoning
    reasoning: str = ""
    contributing_signals: list[str] = Field(
        default_factory=list, description="Agent IDs that contributed"
    )

    # Performance tracking
    weight: float = Field(default=1.0, ge=0.0, description="Current weight in parent's aggregation")
    edge: float = Field(default=0.0, description="Historical performance edge")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def classify_confidence(self) -> None:
        """Set the confidence tier based on raw confidence value."""
        if self.confidence >= 0.72:
            self.confidence_tier = SignalConfidence.HIGH
        elif self.confidence >= 0.55:
            self.confidence_tier = SignalConfidence.MODERATE
        elif self.confidence >= 0.40:
            self.confidence_tier = SignalConfidence.LOW
        else:
            self.confidence_tier = SignalConfidence.NO_SIGNAL
            self.direction = SignalDirection.SKIP


class AggregatedSignal(BaseModel):
    """Aggregated signal from a Group Leader or higher."""

    agent_id: str
    agent_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Aggregated signal
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_tier: SignalConfidence = SignalConfidence.NO_SIGNAL

    # Component breakdown
    child_signals: list[AgentSignal] = Field(default_factory=list)
    consensus_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How much children agree (1.0 = unanimous)",
    )
    active_children: int = Field(default=0)
    skipped_children: int = Field(default=0)

    # Context
    symbol: str = ""
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
