"""TimescaleDB models for market data storage.

Hypertables are created via Alembic migration after table creation.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from atomicx.data.storage.database import Base


class OHLCV(Base):
    """OHLCV candlestick data — TimescaleDB hypertable.

    Stores price candles at multiple timeframes for all tracked symbols.
    Partitioned by time for efficient time-range queries.
    """

    __tablename__ = "ohlcv"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(
        String(10), nullable=False, default="1m"
    )  # 1m, 5m, 15m, 1h, 4h, 1d
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quote_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("timestamp", "symbol", "timeframe", name="uq_ohlcv_ts_sym_tf"),
        Index("ix_ohlcv_symbol_timeframe_ts", "symbol", "timeframe", "timestamp"),
    )


class TickData(Base):
    """Real-time tick/trade data — TimescaleDB hypertable.

    Stores individual trades as they arrive from the WebSocket stream.
    High-frequency: can be millions of rows per day.
    """

    __tablename__ = "tick_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    is_buyer_maker: Mapped[bool | None] = mapped_column(nullable=True)
    trade_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index("ix_tick_symbol_ts", "symbol", "timestamp"),
    )


class OrderBookSnapshot(Base):
    """Order book depth snapshots — TimescaleDB hypertable.

    Stores periodic order book snapshots for liquidity analysis.
    """

    __tablename__ = "order_book_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    bids: Mapped[dict] = mapped_column(JSONB, nullable=False)  # [[price, qty], ...]
    asks: Mapped[dict] = mapped_column(JSONB, nullable=False)  # [[price, qty], ...]
    bid_total_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    ask_total_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    spread: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    mid_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    __table_args__ = (
        Index("ix_orderbook_symbol_ts", "symbol", "timestamp"),
    )


class FundingRate(Base):
    """Perpetual futures funding rate data."""

    __tablename__ = "funding_rates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    funding_rate: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    index_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    __table_args__ = (
        UniqueConstraint("timestamp", "symbol", name="uq_funding_ts_sym"),
        Index("ix_funding_symbol_ts", "symbol", "timestamp"),
    )


class OnChainMetric(Base):
    """On-chain analytics data from providers like CoinGecko, Glassnode."""

    __tablename__ = "onchain_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[Decimal] = mapped_column(Numeric(30, 10), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_onchain_sym_metric_ts", "symbol", "metric_name", "timestamp"),
    )


class DataFreshness(Base):
    """Tracks data freshness per source for circuit breaker monitoring."""

    __tablename__ = "data_freshness"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    last_update: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="healthy"
    )  # healthy, stale, failed
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)


class Liquidation(Base):
    """Global liquidation events (Pain Map) — TimescaleDB hypertable.

    Tracks where retail traders are getting liquidated. This reveals hidden
    support/resistance levels and institutional pain zones.

    INSTITUTIONAL: Liquidations cluster at key levels where leveraged
    positions fail, indicating true market structure vs spoofed walls.
    """

    __tablename__ = "liquidations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # SELL = long liq, BUY = short liq
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    notional_usd: Mapped[float] = mapped_column(sa.Float, nullable=False)  # qty * price
    order_type: Mapped[str | None] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        Index("ix_liq_sym_ts", "symbol", "timestamp"),
        Index("ix_liq_notional", "notional_usd"),
    )


class CumulativeDelta(Base):
    """Cumulative Volume Delta (CVD) — Anti-spoofing metric.

    Tracks ACTUAL executed volume (aggressive buys vs sells) to detect
    if orderbook walls are real or fake.

    INSTITUTIONAL: If OB_IMBALANCE shows bullish intent (big bids) but CVD
    shows bearish execution (sellers hitting bids), it's a spoof — abort trade.
    """

    __tablename__ = "cumulative_delta"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    cvd_value: Mapped[float] = mapped_column(sa.Float, nullable=False)  # Running total
    period_delta: Mapped[float] = mapped_column(sa.Float, nullable=False)  # Change since last
    reset_flag: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="false")

    __table_args__ = (
        Index("ix_cvd_sym_ts", "symbol", "timestamp"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING STATE PERSISTENCE (Fix for persistence gaps)
# ═══════════════════════════════════════════════════════════════════════════


class AgentPerformance(Base):
    """Agent learning state — persists across restarts.

    Stores prediction counts, win rates, trust weights, and active status
    for all 62 agents in the hierarchy. Critical for auto-pruning and
    maintaining agent trust scores across sessions.
    """

    __tablename__ = "agent_performance"

    agent_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    performance_edge: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), nullable=False, default=0.5
    )
    weight: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False, default=1.0)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    last_prediction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_agent_perf_active", "is_active", "performance_edge"),
    )


class StrategyGenome(Base):
    """Strategy evolution genome — persists across restarts.

    Stores performance metrics, regime-specific scores, and mutation parameters
    for all evolved strategies. Enables cross-session strategy evolution.
    """

    __tablename__ = "strategy_genome"

    strategy_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    gene_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_profit: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0.0)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0.0)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False, default=0.0)
    expectancy: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0.0)
    sharpe_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False, default=0.0)
    edge_decay_rate: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False, default=0.0)
    regime_scores: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    best_regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    worst_regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_gene_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active"
    )  # active, degrading, retired
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_genome_status", "status", "win_rate"),
    )


class PredictionOutcome(Base):
    """Prediction tracking and verification — persists across restarts.

    Stores all predictions and their verified outcomes. Enables cross-session
    win rate tracking and prevents abandoning in-flight predictions on restart.
    """

    __tablename__ = "prediction_outcomes"

    prediction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_direction: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    verification_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    was_correct: Mapped[bool | None] = mapped_column(nullable=True)
    actual_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    profit_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    variable_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_outcomes_verified", "verified_at"),
        Index("ix_outcomes_symbol_time", "symbol", "predicted_at"),
        Index("ix_outcomes_regime", "regime", "was_correct"),
    )


class RegimeHistory(Base):
    """Regime transition history — persists across restarts.

    Tracks all market regime changes for drift detection and regime-specific
    strategy adaptation. Enables analysis of regime persistence patterns.
    """

    __tablename__ = "regime_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    old_regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    new_regime: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    drift_score: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    trigger_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_regime_history_symbol", "symbol", "timestamp"),
        Index("ix_regime_transition", "old_regime", "new_regime"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMOUS EVOLUTION SYSTEM (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════


class CausalWeight(Base):
    """Causal weight persistence per variable per regime."""

    __tablename__ = "causal_weights"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    variable_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    weight: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    updated_by: Mapped[str] = mapped_column(String(100), nullable=False)
    update_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    previous_weight: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        UniqueConstraint("variable_id", "regime", name="uq_causal_weights_var_regime"),
    )


class PendingPredictionDB(Base):
    """Pending predictions awaiting verification (DB-backed)."""

    __tablename__ = "pending_predictions"

    prediction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    variables_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    packet_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("ix_pending_predictions_symbol", "symbol", "created_at"),
    )


class LiveConfig(Base):
    """Live configuration with regime-specific overrides."""

    __tablename__ = "live_config"

    config_key: Mapped[str] = mapped_column(String(200), nullable=False)
    regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    config_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    component: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    default_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    updated_by: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default="system"
    )
    update_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    performance_delta: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    __table_args__ = (
        sa.PrimaryKeyConstraint("config_key", "regime", name="pk_live_config"),
        UniqueConstraint("config_key", "regime", name="uq_config_key_regime"),
    )


class EvolutionProposal(Base):
    """Evolution proposals from diagnosis engine."""

    __tablename__ = "evolution_proposals"

    proposal_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    component: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    parameter_path: Mapped[str] = mapped_column(String(200), nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    proposed_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    expected_improvement: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_proposals_status", "status", "created_at"),
    )


class ABTestResult(Base):
    """A/B test results for shadow testing."""

    __tablename__ = "ab_test_results"

    test_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    shadow_predictions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    shadow_win_rate: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    live_win_rate: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    delta: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    cycles_tested: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)


class DiagnosisLog(Base):
    """System health diagnosis results."""

    __tablename__ = "diagnosis_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    diagnosed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    win_rate_by_regime: Mapped[dict] = mapped_column(JSONB, nullable=False)
    win_rate_by_agent: Mapped[dict] = mapped_column(JSONB, nullable=False)
    win_rate_by_variable: Mapped[dict] = mapped_column(JSONB, nullable=False)
    win_rate_by_timeframe: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default='{}')
    worst_component: Mapped[str | None] = mapped_column(String(100), nullable=True)
    best_component: Mapped[str | None] = mapped_column(String(100), nullable=True)
    recommended_actions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    system_health_score: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)


class CodeSnapshot(Base):
    """Code modification tracking and rollback."""

    __tablename__ = "code_snapshots"

    change_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    original_code: Mapped[str] = mapped_column(Text, nullable=False)
    new_code: Mapped[str] = mapped_column(Text, nullable=False)
    change_type: Mapped[str] = mapped_column(String(50), nullable=False)
    proposed_by: Mapped[str] = mapped_column(String(100), nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    shadow_win_rate_before: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    shadow_win_rate_after: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    live_win_rate_before: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    live_win_rate_after: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rollback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_code_snapshots_status", "status", "applied_at"),
    )


class EvolutionReport(Base):
    """Periodic evolution progress reports."""

    __tablename__ = "evolution_reports"

    report_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    changes_made: Mapped[dict] = mapped_column(JSONB, nullable=False)
    health_score: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    top_improvements: Mapped[dict] = mapped_column(JSONB, nullable=False)
    top_weaknesses: Mapped[dict] = mapped_column(JSONB, nullable=False)
    win_rate_trend: Mapped[dict] = mapped_column(JSONB, nullable=False)
    estimated_trajectory: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    next_planned: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class AgentEvolutionLog(Base):
    """Agent parameter mutation history."""

    __tablename__ = "agent_evolution_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    old_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    win_rate_before: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    win_rate_after: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_agent_evolution_log_agent", "agent_id", "created_at"),
    )


class MetaLearningLog(Base):
    """Meta-learning insights — learning from learning."""

    __tablename__ = "meta_learning_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    insight_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_meta_learning_log_type", "insight_type", "created_at"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO MANAGEMENT SYSTEM (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioAsset(Base):
    """Portfolio assets — tracks which assets are in the user's portfolio.

    Supports crypto, stocks, commodities, forex, etc.
    Separate from trading signals — this is the user's actual holdings.

    Status flow: pending → initializing → backfilling → active → error
    """

    __tablename__ = "portfolio_assets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    asset_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="crypto"
    )  # crypto, stock, commodity, forex
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    # ═══ Multi-Asset Support ═══
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )  # pending, initializing, backfilling, active, error
    data_source: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)  # binance, yahoo, alpha_vantage
    backfill_progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0-100%
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # Extensibility (renamed from metadata to avoid SQLAlchemy conflict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_portfolio_assets_status_update", "status", "last_update"),
    )


class Position(Base):
    """Trading positions — tracks individual trades (long/short).

    Records entry/exit prices, P&L, and status.
    Separate from prediction tracking — this is actual money deployed.
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_asset_id: Mapped[int] = mapped_column(
        BigInteger, sa.ForeignKey("portfolio_assets.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # long, short
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", index=True
    )  # open, closed, stopped
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    pnl_percentage: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_positions_status_entry", "status", "entry_time"),
        Index("ix_positions_symbol_status", "symbol", "status"),
    )


class PortfolioStats(Base):
    """Portfolio statistics — cached aggregated metrics.

    Updated on position open/close for fast retrieval.
    Can be global (symbol=NULL) or per-asset.
    """

    __tablename__ = "portfolio_stats"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(
        String(20), nullable=True, unique=True, index=True
    )  # NULL = global
    total_invested: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0.0)
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0.0)
    total_pnl_percentage: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=0.0)
    total_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    closed_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=0.0)
    avg_win: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    avg_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    largest_win: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    largest_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
