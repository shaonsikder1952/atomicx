"""Initial schema — TimescaleDB hypertables for market data.

Revision ID: 001
Revises:
Create Date: 2026-03-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # ── OHLCV ────────────────────────────────────────────────
    op.create_table(
        "ohlcv",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(20, 8), nullable=False),
        sa.Column("quote_volume", sa.Numeric(20, 8), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "timeframe", name="pk_ohlcv"),
    )
    op.create_index("ix_ohlcv_symbol_timeframe_ts", "ohlcv", ["symbol", "timeframe", "timestamp"])

    # Convert to hypertable
    op.execute("SELECT create_hypertable('ohlcv', 'timestamp', migrate_data => true)")

    # ── Tick Data ────────────────────────────────────────────
    op.create_table(
        "tick_data",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("is_buyer_maker", sa.Boolean(), nullable=True),
        sa.Column("trade_id", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "id", name="pk_tick_data"),
    )
    op.create_index("ix_tick_symbol_ts", "tick_data", ["symbol", "timestamp"])
    op.execute("SELECT create_hypertable('tick_data', 'timestamp', migrate_data => true)")

    # ── Order Book Snapshots ─────────────────────────────────
    op.create_table(
        "order_book_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("bids", postgresql.JSONB(), nullable=False),
        sa.Column("asks", postgresql.JSONB(), nullable=False),
        sa.Column("bid_total_volume", sa.Numeric(20, 8), nullable=True),
        sa.Column("ask_total_volume", sa.Numeric(20, 8), nullable=True),
        sa.Column("spread", sa.Numeric(20, 8), nullable=True),
        sa.Column("mid_price", sa.Numeric(20, 8), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "id", name="pk_order_book_snapshots"),
    )
    op.create_index("ix_orderbook_symbol_ts", "order_book_snapshots", ["symbol", "timestamp"])
    op.execute(
        "SELECT create_hypertable('order_book_snapshots', 'timestamp', migrate_data => true)"
    )

    # ── Funding Rates ────────────────────────────────────────
    op.create_table(
        "funding_rates",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("funding_rate", sa.Numeric(20, 10), nullable=False),
        sa.Column("mark_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("index_price", sa.Numeric(20, 8), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "id", name="pk_funding_rates"),
    )
    op.create_index("ix_funding_symbol_ts", "funding_rates", ["symbol", "timestamp"])
    op.execute("SELECT create_hypertable('funding_rates', 'timestamp', migrate_data => true)")

    # ── On-Chain Metrics ─────────────────────────────────────
    op.create_table(
        "onchain_metrics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("value", sa.Numeric(30, 10), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "metric_name", "id", name="pk_onchain_metrics"),
    )
    op.create_index("ix_onchain_sym_metric_ts", "onchain_metrics", ["symbol", "metric_name", "timestamp"])
    op.execute("SELECT create_hypertable('onchain_metrics', 'timestamp', migrate_data => true)")

    # ── Data Freshness ───────────────────────────────────────
    op.create_table(
        "data_freshness",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_name", sa.String(100), nullable=False),
        sa.Column(
            "last_update",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="healthy"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_data_freshness"),
        sa.UniqueConstraint("source_name", name="uq_data_freshness_source_name"),
    )

    # ── Compression Policies ─────────────────────────────────
    # Compress tick data older than 7 days
    op.execute(
        "ALTER TABLE tick_data SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'symbol')"
    )
    op.execute(
        "SELECT add_compression_policy('tick_data', INTERVAL '7 days')"
    )

    # Compress order book snapshots older than 3 days
    op.execute(
        "ALTER TABLE order_book_snapshots SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'symbol')"
    )
    op.execute(
        "SELECT add_compression_policy('order_book_snapshots', INTERVAL '3 days')"
    )


def downgrade() -> None:
    op.drop_table("data_freshness")
    op.drop_table("onchain_metrics")
    op.drop_table("funding_rates")
    op.drop_table("order_book_snapshots")
    op.drop_table("tick_data")
    op.drop_table("ohlcv")
