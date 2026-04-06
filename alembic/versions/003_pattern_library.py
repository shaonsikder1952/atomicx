"""Create pattern_library table for historical pattern tracking

Revision ID: 003_pattern_library
Revises: 002
Create Date: 2026-04-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '003_pattern_library'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create pattern_library table and convert to TimescaleDB hypertable."""

    # Create pattern_library table
    op.create_table(
        'pattern_library',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('pattern_id', sa.String(100), nullable=False, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('timeframe', sa.String(10), nullable=False, index=True),
        sa.Column('pattern_name', sa.String(100), nullable=False, index=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('variables_snapshot', JSONB, nullable=False),
        sa.Column('regime', sa.String(50), nullable=True),
        sa.Column('confidence_score', sa.Numeric(5, 4), nullable=False),

        # Outcome tracking (filled after verification window)
        sa.Column('outcome_direction', sa.String(20), nullable=True),  # bullish/bearish/neutral
        sa.Column('outcome_return', sa.Numeric(10, 6), nullable=True),  # actual return %
        sa.Column('outcome_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('verification_candles', sa.Integer(), nullable=True),  # candles until verified
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),

        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('pattern_metadata', JSONB, nullable=True),

        sa.PrimaryKeyConstraint('id', 'detected_at'),
        sa.Index('ix_pattern_lib_sym_tf_detected', 'symbol', 'timeframe', 'detected_at'),
        sa.Index('ix_pattern_lib_name_regime', 'pattern_name', 'regime'),
        sa.Index('ix_pattern_lib_verified', 'outcome_verified', 'pattern_name'),
    )

    # Convert to TimescaleDB hypertable (partitioned by detected_at)
    op.execute("""
        SELECT create_hypertable(
            'pattern_library',
            'detected_at',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    """)

    # Create materialized view for pattern performance stats
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS pattern_performance AS
        SELECT
            pattern_name,
            regime,
            symbol,
            timeframe,
            COUNT(*) as total_occurrences,
            COUNT(*) FILTER (WHERE outcome_verified = true) as verified_count,
            COUNT(*) FILTER (WHERE outcome_verified = true AND outcome_direction = 'bullish') as bullish_outcomes,
            COUNT(*) FILTER (WHERE outcome_verified = true AND outcome_direction = 'bearish') as bearish_outcomes,
            AVG(outcome_return) FILTER (WHERE outcome_verified = true) as avg_return,
            STDDEV(outcome_return) FILTER (WHERE outcome_verified = true) as return_stddev,
            AVG(confidence_score) as avg_confidence,
            MAX(detected_at) as last_seen
        FROM pattern_library
        GROUP BY pattern_name, regime, symbol, timeframe;

        CREATE INDEX ON pattern_performance (pattern_name, regime, symbol, timeframe);
    """)

    # Create refresh function for continuous aggregate
    op.execute("""
        CREATE OR REPLACE FUNCTION refresh_pattern_performance()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW pattern_performance;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Drop pattern_library table and related objects."""
    op.execute("DROP FUNCTION IF EXISTS refresh_pattern_performance();")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS pattern_performance;")
    op.drop_table('pattern_library')
