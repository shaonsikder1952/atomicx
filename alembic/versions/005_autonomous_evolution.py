"""Add autonomous evolution system tables

Revision ID: 005_autonomous_evolution
Revises: 004_learning_persistence
Create Date: 2026-04-05

Adds tables for:
- Causal weights persistence (per-variable, per-regime)
- Pending predictions (table-based with expiry)
- Live configuration management (all thresholds and weights)
- Evolution proposals (diagnosis → proposals → A/B testing → deployment)
- A/B test results (shadow testing)
- Diagnosis logs (health monitoring)
- Code snapshots (self-modification tracking)
- Evolution reports (what changed, why, results)
- Agent evolution log (parameter mutations)
- Meta-learning log (learning from learning)

Enables autonomous self-improvement with safety mechanisms.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_autonomous_evolution'
down_revision = '004_learning_persistence'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add autonomous evolution system tables."""

    # ═══ Causal Weights Table ═══
    op.create_table(
        'causal_weights',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('variable_id', sa.String(length=50), nullable=False),
        sa.Column('weight', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('regime', sa.String(length=50), nullable=True),
        sa.Column('updated_by', sa.String(length=100), nullable=False),
        sa.Column('update_reason', sa.Text(), nullable=True),
        sa.Column('previous_weight', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('variable_id', 'regime', name='uq_causal_weights_var_regime')
    )
    op.create_index('ix_causal_weights_var', 'causal_weights', ['variable_id'])
    op.create_index('ix_causal_weights_updated', 'causal_weights', ['updated_at'])

    # ═══ Pending Predictions Table ═══
    op.create_table(
        'pending_predictions',
        sa.Column('prediction_id', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('direction', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('regime', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('variables_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('packet_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('prediction_id')
    )
    op.create_index('ix_pending_predictions_expires', 'pending_predictions', ['expires_at'])
    op.create_index('ix_pending_predictions_symbol', 'pending_predictions', ['symbol', 'created_at'])

    # ═══ Live Config Table ═══
    op.create_table(
        'live_config',
        sa.Column('config_key', sa.String(length=200), nullable=False),
        sa.Column('config_value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('regime', sa.String(length=50), nullable=True),
        sa.Column('default_value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_by', sa.String(length=100), nullable=False, server_default='system'),
        sa.Column('update_reason', sa.Text(), nullable=True),
        sa.Column('performance_delta', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.PrimaryKeyConstraint('config_key', 'regime', name='pk_live_config'),
        sa.UniqueConstraint('config_key', 'regime', name='uq_config_key_regime')
    )
    op.create_index('ix_live_config_component', 'live_config', ['component'])
    op.create_index('ix_live_config_updated', 'live_config', ['updated_at'])

    # ═══ Evolution Proposals Table ═══
    op.create_table(
        'evolution_proposals',
        sa.Column('proposal_id', sa.String(length=50), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('action_type', sa.String(length=50), nullable=False),
        sa.Column('parameter_path', sa.String(length=200), nullable=False),
        sa.Column('old_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('proposed_value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('expected_improvement', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejected_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('proposal_id')
    )
    op.create_index('ix_proposals_status', 'evolution_proposals', ['status', 'created_at'])
    op.create_index('ix_proposals_component', 'evolution_proposals', ['component'])

    # ═══ A/B Test Results Table ═══
    op.create_table(
        'ab_test_results',
        sa.Column('test_id', sa.String(length=50), nullable=False),
        sa.Column('proposal_id', sa.String(length=50), nullable=False),
        sa.Column('shadow_predictions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('shadow_win_rate', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('live_win_rate', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('delta', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('cycles_tested', sa.Integer(), nullable=False),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('test_id'),
        sa.ForeignKeyConstraint(['proposal_id'], ['evolution_proposals.proposal_id'])
    )
    op.create_index('ix_ab_test_proposal', 'ab_test_results', ['proposal_id'])
    op.create_index('ix_ab_test_decided', 'ab_test_results', ['decided_at'])

    # ═══ Diagnosis Log Table ═══
    op.create_table(
        'diagnosis_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('diagnosed_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('win_rate_by_regime', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('win_rate_by_agent', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('win_rate_by_variable', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('win_rate_by_timeframe', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('worst_component', sa.String(length=100), nullable=True),
        sa.Column('best_component', sa.String(length=100), nullable=True),
        sa.Column('recommended_actions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('system_health_score', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_diagnosis_log_time', 'diagnosis_log', ['diagnosed_at'])

    # ═══ Code Snapshots Table ═══
    op.create_table(
        'code_snapshots',
        sa.Column('change_id', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('original_code', sa.Text(), nullable=False),
        sa.Column('new_code', sa.Text(), nullable=False),
        sa.Column('change_type', sa.String(length=50), nullable=False),
        sa.Column('proposed_by', sa.String(length=100), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('shadow_win_rate_before', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('shadow_win_rate_after', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('live_win_rate_before', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('live_win_rate_after', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rolled_back_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rollback_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('change_id')
    )
    op.create_index('ix_code_snapshots_status', 'code_snapshots', ['status', 'applied_at'])
    op.create_index('ix_code_snapshots_file', 'code_snapshots', ['file_path'])

    # ═══ Evolution Reports Table ═══
    op.create_table(
        'evolution_reports',
        sa.Column('report_id', sa.String(length=50), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('changes_made', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('health_score', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('top_improvements', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('top_weaknesses', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('win_rate_trend', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('estimated_trajectory', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('next_planned', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('report_id')
    )
    op.create_index('ix_evolution_reports_time', 'evolution_reports', ['generated_at'])

    # ═══ Agent Evolution Log Table ═══
    op.create_table(
        'agent_evolution_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('agent_id', sa.String(length=50), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('old_params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('win_rate_before', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('win_rate_after', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_evolution_log_agent', 'agent_evolution_log', ['agent_id', 'created_at'])
    op.create_index('ix_agent_evolution_log_type', 'agent_evolution_log', ['event_type'])

    # ═══ Meta Learning Log Table ═══
    op.create_table(
        'meta_learning_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('insight_type', sa.String(length=100), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_meta_learning_log_type', 'meta_learning_log', ['insight_type', 'created_at'])


def downgrade() -> None:
    """Remove autonomous evolution system tables."""
    op.drop_table('meta_learning_log')
    op.drop_table('agent_evolution_log')
    op.drop_table('evolution_reports')
    op.drop_table('code_snapshots')
    op.drop_table('diagnosis_log')
    op.drop_table('ab_test_results')
    op.drop_table('evolution_proposals')
    op.drop_table('live_config')
    op.drop_table('pending_predictions')
    op.drop_table('causal_weights')
