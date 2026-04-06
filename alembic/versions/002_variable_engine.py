"""Phase 2 — Variable Engine tables.

Revision ID: 002
Revises: 001
Create Date: 2026-03-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Variable Registry ────────────────────────────────────
    op.create_table(
        "variable_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("variable_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(50), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("update_frequency", sa.String(10), nullable=False, server_default="1m"),
        sa.Column("causal_half_life", sa.Float(), nullable=False, server_default="24.0"),
        sa.Column("reliability_score", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("lookback_periods", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("symbol_specific", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("current_weight", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("performance_edge", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("prediction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_variable_registry"),
        sa.UniqueConstraint("variable_id", name="uq_variable_registry_variable_id"),
    )
    op.create_index("ix_vr_domain", "variable_registry", ["domain"])
    op.create_index("ix_vr_category", "variable_registry", ["category"])

    # ── Computed Variables (hypertable) ──────────────────────
    op.create_table(
        "computed_variables",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("variable_id", sa.String(100), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False, server_default="1m"),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "variable_id", "symbol", "id", name="pk_computed_variables"),
    )
    op.create_index("ix_cv_varid_sym_ts", "computed_variables", ["variable_id", "symbol", "timestamp"])
    op.create_index("ix_cv_sym_tf_ts", "computed_variables", ["symbol", "timeframe", "timestamp"])
    op.execute("SELECT create_hypertable('computed_variables', 'timestamp', migrate_data => true)")

    # Compress after 7 days
    op.execute(
        "ALTER TABLE computed_variables SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'variable_id,symbol')"
    )
    op.execute("SELECT add_compression_policy('computed_variables', INTERVAL '7 days')")

    # ── Variable Weight History ──────────────────────────────
    op.create_table(
        "variable_weight_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("variable_id", sa.String(100), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("performance_edge", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("prediction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "variable_id", "id", name="pk_variable_weight_history"),
    )
    op.create_index("ix_vwh_varid_ts", "variable_weight_history", ["variable_id", "timestamp"])
    op.execute("SELECT create_hypertable('variable_weight_history', 'timestamp', migrate_data => true)")


def downgrade() -> None:
    op.drop_table("variable_weight_history")
    op.drop_table("computed_variables")
    op.drop_table("variable_registry")
