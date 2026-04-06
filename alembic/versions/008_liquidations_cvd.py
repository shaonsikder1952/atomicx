"""Phase 8 — Liquidations and CVD tables.

Revision ID: 008_liquidations_cvd
Revises: 007_causality_tracking
Create Date: 2026-04-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008_liquidations_cvd"
down_revision: Union[str, None] = "007_causality_tracking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Liquidations (Pain Map) ──────────────────────────────
    op.create_table(
        "liquidations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),  # SELL = long liq, BUY = short liq
        sa.Column("price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("notional_usd", sa.Float(), nullable=False),  # qty * price for filtering
        sa.Column("order_type", sa.String(20), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "id", name="pk_liquidations"),
    )
    op.create_index("ix_liq_sym_ts", "liquidations", ["symbol", "timestamp"])
    op.create_index("ix_liq_notional", "liquidations", ["notional_usd"])
    op.execute("SELECT create_hypertable('liquidations', 'timestamp', migrate_data => true)")

    # Compress after 30 days
    op.execute(
        "ALTER TABLE liquidations SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'symbol')"
    )
    op.execute("SELECT add_compression_policy('liquidations', INTERVAL '30 days')")

    # ── Cumulative Volume Delta (Anti-Spoofing) ──────────────
    op.create_table(
        "cumulative_delta",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("cvd_value", sa.Float(), nullable=False),  # Running total
        sa.Column("period_delta", sa.Float(), nullable=False),  # Change since last record
        sa.Column("reset_flag", sa.Boolean(), nullable=False, server_default="false"),  # Daily reset marker
        sa.PrimaryKeyConstraint("timestamp", "symbol", "id", name="pk_cumulative_delta"),
    )
    op.create_index("ix_cvd_sym_ts", "cumulative_delta", ["symbol", "timestamp"])
    op.execute("SELECT create_hypertable('cumulative_delta', 'timestamp', migrate_data => true)")

    # Compress after 7 days
    op.execute(
        "ALTER TABLE cumulative_delta SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'symbol')"
    )
    op.execute("SELECT add_compression_policy('cumulative_delta', INTERVAL '7 days')")


def downgrade() -> None:
    op.drop_table("cumulative_delta")
    op.drop_table("liquidations")
