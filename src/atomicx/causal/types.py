"""Causal graph types and data models.

Defines the structure of the causal DAG and individual causal edges.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CausalStrength(str, Enum):
    """Strength classification of a causal relationship."""

    STRONG = "strong"  # High confidence, replicated across regimes
    MODERATE = "moderate"  # Consistent but regime-dependent
    WEAK = "weak"  # Statistically significant but small effect
    SPURIOUS = "spurious"  # Failed refutation tests — likely not causal


class CausalDirection(str, Enum):
    """Direction of causal influence on the target."""

    POSITIVE = "positive"  # Cause↑ → Effect↑
    NEGATIVE = "negative"  # Cause↑ → Effect↓
    NONLINEAR = "nonlinear"  # Complex relationship


class CausalEdge(BaseModel):
    """A causal relationship between two variables."""

    source: str = Field(description="Cause variable ID")
    target: str = Field(description="Effect variable ID")
    weight: float = Field(ge=0.0, le=1.0, description="Edge strength (0-1)")
    direction: CausalDirection = Field(description="Direction of causal influence")
    strength: CausalStrength = Field(default=CausalStrength.MODERATE)

    # Discovery metadata
    algorithm: str = Field(description="Which algorithm discovered this edge")
    p_value: float | None = Field(default=None, description="Statistical significance")
    lag_periods: int = Field(default=0, description="Temporal lag of causal effect")

    # Validation
    refutation_passed: bool = Field(
        default=False, description="Whether DoWhy refutation tests passed"
    )
    regime_stable: bool = Field(
        default=False, description="Whether edge holds across market regimes"
    )

    # Timestamps
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_validated: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Context
    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalChain(BaseModel):
    """A multi-step causal chain: A → B → C → price."""

    chain_id: str
    edges: list[CausalEdge]
    total_strength: float = Field(description="Product of edge weights along chain")
    reasoning: str = Field(default="", description="Human-readable explanation")
    prediction_impact: float = Field(
        default=0.0, description="How much this chain contributes to predictions"
    )


class CausalDAG(BaseModel):
    """The complete causal DAG — all known causal relationships."""

    edges: list[CausalEdge] = Field(default_factory=list)
    chains: list[CausalChain] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    algorithm_versions: dict[str, str] = Field(default_factory=dict)
    variable_count: int = Field(default=0)

    def get_causes_of(self, target: str) -> list[CausalEdge]:
        """Get all variables that cause a given target."""
        return [e for e in self.edges if e.target == target and e.strength != CausalStrength.SPURIOUS]

    def get_effects_of(self, source: str) -> list[CausalEdge]:
        """Get all variables affected by a given source."""
        return [e for e in self.edges if e.source == source and e.strength != CausalStrength.SPURIOUS]

    def get_strongest_paths_to(self, target: str, top_k: int = 10) -> list[CausalChain]:
        """Get the strongest causal chains leading to a target."""
        relevant = [c for c in self.chains if c.edges[-1].target == target]
        return sorted(relevant, key=lambda c: c.total_strength, reverse=True)[:top_k]

    def prune_weak_edges(self, min_weight: float = 0.02) -> int:
        """Remove edges below the weight threshold. Returns count removed."""
        before = len(self.edges)
        self.edges = [e for e in self.edges if e.weight >= min_weight]
        removed = before - len(self.edges)
        return removed
