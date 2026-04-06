"""Add news intelligence tables

Revision ID: 006_news_intelligence
Revises: 005_autonomous_evolution
Create Date: 2026-04-05

Adds tables for:
- News events (news stories with analysis and outcomes)
- News patterns (learned causal patterns from historical events)
- News variables (dynamic trading signals from news patterns)

Enables learning causal relationships: "When X news happens, Y price move follows with Z confidence"
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_news_intelligence'
down_revision = '005_autonomous_evolution'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add news intelligence tables."""

    # ═══ News Events Table ═══
    op.create_table(
        'news_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.String(length=64), nullable=False),

        # Story metadata
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=255), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('discovered_at', sa.DateTime(timezone=True), nullable=False),

        # Significance
        sa.Column('significance_score', sa.Float(), nullable=False),
        sa.Column('keywords_matched', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Deep-dive analysis results
        sa.Column('article_content', sa.Text(), nullable=True),
        sa.Column('sentiment', sa.String(length=50), nullable=True),
        sa.Column('people_mentioned', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('entities_mentioned', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Impact prediction
        sa.Column('predicted_impact', sa.String(length=50), nullable=True),
        sa.Column('predicted_magnitude', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),

        # Actual outcome (filled after event)
        sa.Column('outcome_tracked', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('price_change_1h', sa.Float(), nullable=True),
        sa.Column('price_change_4h', sa.Float(), nullable=True),
        sa.Column('price_change_24h', sa.Float(), nullable=True),
        sa.Column('volatility_change', sa.Float(), nullable=True),
        sa.Column('volume_change', sa.Float(), nullable=True),

        # Learning
        sa.Column('prediction_correct', sa.Boolean(), nullable=True),
        sa.Column('pattern_id', sa.Integer(), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_news_events_event_id', 'news_events', ['event_id'], unique=True)
    op.create_index('ix_news_events_source', 'news_events', ['source'])
    op.create_index('ix_news_events_discovered_at', 'news_events', ['discovered_at'])
    op.create_index('ix_news_events_significance_score', 'news_events', ['significance_score'])
    op.create_index('ix_news_events_sentiment', 'news_events', ['sentiment'])
    op.create_index('ix_news_events_outcome_tracked', 'news_events', ['outcome_tracked'])
    op.create_index('idx_news_significance_time', 'news_events', ['significance_score', 'discovered_at'])
    op.create_index('idx_news_outcome', 'news_events', ['outcome_tracked', 'discovered_at'])
    op.create_index('idx_news_sentiment_time', 'news_events', ['sentiment', 'discovered_at'])

    # ═══ News Patterns Table ═══
    op.create_table(
        'news_patterns',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('pattern_id', sa.String(length=64), nullable=False),

        # Pattern definition
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('trigger_keywords', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('source_filter', sa.String(length=255), nullable=True),

        # Pattern characteristics
        sa.Column('avg_price_impact', sa.Float(), nullable=True),
        sa.Column('avg_timeframe_hours', sa.Float(), nullable=True),
        sa.Column('volatility_impact', sa.Float(), nullable=True),

        # Statistics
        sa.Column('occurrences', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('correct_predictions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('win_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_magnitude', sa.Float(), nullable=True),

        # Pattern strength
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=True),

        # Pattern details
        sa.Column('examples', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_news_patterns_pattern_id', 'news_patterns', ['pattern_id'], unique=True)
    op.create_index('ix_news_patterns_pattern_type', 'news_patterns', ['pattern_type'])
    op.create_index('ix_news_patterns_win_rate', 'news_patterns', ['win_rate'])
    op.create_index('ix_news_patterns_confidence', 'news_patterns', ['confidence'])
    op.create_index('ix_news_patterns_last_seen', 'news_patterns', ['last_seen'])
    op.create_index('idx_pattern_confidence_win_rate', 'news_patterns', ['confidence', 'win_rate'])
    op.create_index('idx_pattern_type_confidence', 'news_patterns', ['pattern_type', 'confidence'])

    # ═══ News Variables Table ═══
    op.create_table(
        'news_variables',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('variable_name', sa.String(length=255), nullable=False),

        # Variable definition
        sa.Column('pattern_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),

        # Current state
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('current_value', sa.Float(), nullable=True),
        sa.Column('last_triggered', sa.DateTime(timezone=True), nullable=True),

        # Performance
        sa.Column('total_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('winning_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('edge', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),

        # Meta
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_news_variables_variable_name', 'news_variables', ['variable_name'], unique=True)
    op.create_index('ix_news_variables_is_active', 'news_variables', ['is_active'])
    op.create_index('idx_news_var_active_edge', 'news_variables', ['is_active', 'edge'])

    # Add foreign key for pattern_id in news_events
    op.create_foreign_key(
        'fk_news_events_pattern_id',
        'news_events', 'news_patterns',
        ['pattern_id'], ['id']
    )

    # Add foreign key for pattern_id in news_variables
    op.create_foreign_key(
        'fk_news_variables_pattern_id',
        'news_variables', 'news_patterns',
        ['pattern_id'], ['id']
    )


def downgrade() -> None:
    """Remove news intelligence tables."""
    op.drop_table('news_variables')
    op.drop_table('news_events')
    op.drop_table('news_patterns')
