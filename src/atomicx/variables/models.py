"""Database models for the Variable Registry.

Persists variable definitions, computed values, and weight history
for tracking performance and enabling the auto-demotion system.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from atomicx.data.storage.database import Base


class VariableRegistryEntry(Base):
    """Persisted variable definition in the registry."""

    __tablename__ = "variable_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variable_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)

    # Classification
    domain: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    tags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Data source
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    update_frequency: Mapped[str] = mapped_column(String(10), nullable=False, default="1m")

    # Signal characteristics
    causal_half_life: Mapped[float] = mapped_column(Float, nullable=False, default=24.0)
    reliability_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    lookback_periods: Mapped[int] = mapped_column(Integer, nullable=False, default=14)

    # Configuration
    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    symbol_specific: Mapped[bool] = mapped_column(nullable=False, default=True)

    # Performance tracking
    current_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    performance_edge: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prediction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ComputedVariable(Base):
    """Stores computed variable values — TimescaleDB hypertable."""

    __tablename__ = "computed_variables"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    variable_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="1m")
    value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_cv_varid_sym_ts", "variable_id", "symbol", "timestamp"),
        Index("ix_cv_sym_tf_ts", "symbol", "timeframe", "timestamp"),
    )


class VariableWeightHistory(Base):
    """Tracks weight changes over time for performance analysis."""

    __tablename__ = "variable_weight_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    variable_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    performance_edge: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prediction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_vwh_varid_ts", "variable_id", "timestamp"),
    )
