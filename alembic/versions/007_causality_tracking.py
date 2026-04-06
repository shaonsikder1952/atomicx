"""Add causality tracking and decision audit

Revision ID: 007_causality_tracking
Revises: 006_news_intelligence
Create Date: 2026-04-05
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '007_causality_tracking'
down_revision = '006_news_intelligence'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add causality tracking fields and decision audit table."""

    # Add causality fields to news_events
    op.add_column('news_events', sa.Column('decision_reasoning', sa.JSON(), nullable=True))
    op.add_column('news_events', sa.Column('contributing_factors', sa.JSON(), nullable=True))
    op.add_column('news_events', sa.Column('variable_impacts', sa.JSON(), nullable=True))
    op.add_column('news_events', sa.Column('confidence_breakdown', sa.JSON(), nullable=True))
    op.add_column('news_events', sa.Column('learning_notes', sa.JSON(), nullable=True))

    # Add learning tracking to news_patterns
    op.add_column('news_patterns', sa.Column('learning_iterations', sa.JSON(), nullable=True))
    op.add_column('news_patterns', sa.Column('improvement_log', sa.JSON(), nullable=True))
    op.add_column('news_patterns', sa.Column('failed_attempts', sa.JSON(), nullable=True))

    # Create decision_audits table
    op.create_table(
        'decision_audits',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('audit_id', sa.String(64), unique=True, nullable=False, index=True),
        sa.Column('decision_type', sa.String(50), nullable=False, index=True),
        sa.Column('decision_timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('decision_outcome', sa.String(50), nullable=True),
        sa.Column('news_event_id', sa.Integer(), sa.ForeignKey('news_events.id'), nullable=True),
        sa.Column('pattern_id', sa.Integer(), sa.ForeignKey('news_patterns.id'), nullable=True),
        sa.Column('causal_chain', sa.JSON(), nullable=True),
        sa.Column('reasoning_tree', sa.JSON(), nullable=True),
        sa.Column('factors_analyzed', sa.JSON(), nullable=True),
        sa.Column('variables_changed', sa.JSON(), nullable=True),
        sa.Column('thinking_log', sa.JSON(), nullable=True),
        sa.Column('problems_found', sa.JSON(), nullable=True),
        sa.Column('predicted_outcome', sa.String(50), nullable=True),
        sa.Column('actual_outcome', sa.String(50), nullable=True),
        sa.Column('was_correct', sa.Boolean(), nullable=True),
        sa.Column('error_magnitude', sa.Float(), nullable=True),
        sa.Column('learning_insights', sa.JSON(), nullable=True),
        sa.Column('system_improvements', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Create indexes for decision_audits
    op.create_index('idx_decision_type_time', 'decision_audits', ['decision_type', 'decision_timestamp'])
    op.create_index('idx_decision_correctness', 'decision_audits', ['was_correct', 'decision_timestamp'])


def downgrade() -> None:
    """Remove causality tracking."""

    # Drop decision_audits table
    op.drop_index('idx_decision_correctness', table_name='decision_audits')
    op.drop_index('idx_decision_type_time', table_name='decision_audits')
    op.drop_table('decision_audits')

    # Remove columns from news_patterns
    op.drop_column('news_patterns', 'failed_attempts')
    op.drop_column('news_patterns', 'improvement_log')
    op.drop_column('news_patterns', 'learning_iterations')

    # Remove columns from news_events
    op.drop_column('news_events', 'learning_notes')
    op.drop_column('news_events', 'confidence_breakdown')
    op.drop_column('news_events', 'variable_impacts')
    op.drop_column('news_events', 'contributing_factors')
    op.drop_column('news_events', 'decision_reasoning')
