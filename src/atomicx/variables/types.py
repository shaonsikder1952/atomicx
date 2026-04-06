"""Variable base types — defines what a variable is in the AtomicX system.

Every measurable quantity in the system is a Variable. Each variable has:
- A unique identifier and human-readable name
- A domain (Economic, Social, Behavioral, etc.)
- A data source and update frequency
- A causal half-life (how long its signal persists)
- A reliability score (how trustworthy the source is)
- Weight history (how its importance has changed over time)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VariableDomain(str, Enum):
    """The 7 universal domains for atomic variables."""

    PHYSICAL = "physical"  # Market microstructure, order flow
    BIOLOGICAL = "biological"  # Growth/decay patterns, network effects
    PSYCHOLOGICAL = "psychological"  # Sentiment, fear/greed, narrative
    BEHAVIORAL = "behavioral"  # Trading patterns, whale movements
    ECONOMIC = "economic"  # Price, volume, funding, on-chain metrics
    SOCIAL = "social"  # Social media, news, community signals
    TEMPORAL = "temporal"  # Time-based patterns, seasonality, cycles


class VariableTimeframe(str, Enum):
    """Supported computation timeframes."""

    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"


class VariableStatus(str, Enum):
    """Variable lifecycle status."""

    ACTIVE = "active"  # Currently producing signals
    OBSERVATION = "observation"  # Tracked but not influencing predictions
    DEMOTED = "demoted"  # Auto-demoted for poor performance
    RETIRED = "retired"  # Permanently removed from system


class VariableDefinition(BaseModel):
    """Defines a variable in the system — its identity and metadata."""

    # Identity
    id: str = Field(description="Unique identifier, e.g. 'RSI_14_4H'")
    name: str = Field(description="Human-readable name, e.g. 'RSI (14-period, 4H)'")
    description: str = Field(default="", description="What this variable measures")

    # Classification
    domain: VariableDomain = Field(description="Which of the 7 domains this belongs to")
    category: str = Field(description="Sub-category, e.g. 'momentum', 'volatility', 'on-chain'")
    tags: list[str] = Field(default_factory=list, description="Free-form tags for search")

    # Data source
    source: str = Field(description="Where the data comes from, e.g. 'binance', 'coingecko'")
    source_field: str = Field(default="", description="Specific field within source data")
    update_frequency: VariableTimeframe = Field(
        default=VariableTimeframe.M1,
        description="How often this variable updates",
    )

    # Signal characteristics
    causal_half_life: float = Field(
        default=24.0,
        description="Hours until signal strength decays 50%",
    )
    reliability_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How reliable the data source is (0-1)",
    )
    lookback_periods: int = Field(
        default=14,
        description="Number of periods needed for computation",
    )

    # Configuration
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Computation parameters, e.g. {'period': 14, 'std_dev': 2.0}",
    )
    symbol_specific: bool = Field(
        default=True,
        description="Whether this variable is per-symbol or global",
    )

    # Lifecycle
    status: VariableStatus = Field(default=VariableStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class VariableValue(BaseModel):
    """A computed variable value at a point in time."""

    variable_id: str
    symbol: str
    timestamp: datetime
    value: float
    timeframe: VariableTimeframe = VariableTimeframe.M1
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VariableWeightRecord(BaseModel):
    """Tracks how a variable's weight has changed over time."""

    variable_id: str
    timestamp: datetime
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="")
    performance_edge: float = Field(
        default=0.0,
        description="Percentage improvement over baseline when this variable is included",
    )
    prediction_count: int = Field(default=0)
