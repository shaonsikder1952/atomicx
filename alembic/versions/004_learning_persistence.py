"""Add learning state persistence tables

Revision ID: 004_learning_persistence
Revises: 003_pattern_library
Create Date: 2026-04-05

Adds tables for:
- Agent performance (trust weights, prediction counts)
- Strategy genome (evolution history, regime-specific scores)
- Prediction outcomes (verified predictions, in-flight tracking)
- Regime history (regime transitions, drift detection)

Fixes critical persistence gaps where learning state was lost on restart.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_learning_persistence'
down_revision = '003_pattern_library'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add learning state persistence tables."""

    # ═══ Agent Performance Table ═══
    op.create_table(
        'agent_performance',
        sa.Column('agent_id', sa.String(length=50), nullable=False),
        sa.Column('total_predictions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('correct_predictions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('performance_edge', sa.Numeric(precision=10, scale=6), nullable=False, server_default='0.5'),
        sa.Column('weight', sa.Numeric(precision=10, scale=6), nullable=False, server_default='1.0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_prediction_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('agent_id')
    )
    op.create_index('ix_agent_perf_active', 'agent_performance', ['is_active', 'performance_edge'])

    # ═══ Strategy Genome Table ═══
    op.create_table(
        'strategy_genome',
        sa.Column('strategy_id', sa.String(length=100), nullable=False),
        sa.Column('gene_id', sa.String(length=20), nullable=False),
        sa.Column('total_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('winning_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_profit', sa.Numeric(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('max_drawdown', sa.Numeric(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('win_rate', sa.Numeric(precision=10, scale=6), nullable=False, server_default='0.0'),
        sa.Column('expectancy', sa.Numeric(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('sharpe_ratio', sa.Numeric(precision=10, scale=6), nullable=False, server_default='0.0'),
        sa.Column('edge_decay_rate', sa.Numeric(precision=10, scale=6), nullable=False, server_default='0.0'),
        sa.Column('regime_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('best_regime', sa.String(length=50), nullable=True),
        sa.Column('worst_regime', sa.String(length=50), nullable=True),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('generation', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('parent_gene_id', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('strategy_id'),
        sa.UniqueConstraint('gene_id', name='uq_strategy_gene_id')
    )
    op.create_index('ix_genome_status', 'strategy_genome', ['status', 'win_rate'])

    # ═══ Prediction Outcomes Table ═══
    op.create_table(
        'prediction_outcomes',
        sa.Column('prediction_id', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('predicted_direction', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('verification_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('was_correct', sa.Boolean(), nullable=True),
        sa.Column('actual_return', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('profit_return', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('regime', sa.String(length=50), nullable=True),
        sa.Column('predicted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('variable_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('prediction_id')
    )
    op.create_index('ix_outcomes_verified', 'prediction_outcomes', ['verified_at'])
    op.create_index('ix_outcomes_symbol_time', 'prediction_outcomes', ['symbol', 'predicted_at'])
    op.create_index('ix_outcomes_regime', 'prediction_outcomes', ['regime', 'was_correct'])

    # ═══ Regime History Table ═══
    op.create_table(
        'regime_history',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('old_regime', sa.String(length=50), nullable=True),
        sa.Column('new_regime', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('drift_score', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('trigger_reason', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_regime_history_symbol', 'regime_history', ['symbol', 'timestamp'])
    op.create_index('ix_regime_history_ts', 'regime_history', ['timestamp'])
    op.create_index('ix_regime_transition', 'regime_history', ['old_regime', 'new_regime'])


def downgrade() -> None:
    """Remove learning state persistence tables."""
    op.drop_table('regime_history')
    op.drop_table('prediction_outcomes')
    op.drop_table('strategy_genome')
    op.drop_table('agent_performance')
