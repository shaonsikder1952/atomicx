"""Portfolio and position tracking system.

Revision ID: 009_portfolio_system
Revises: 008_liquidations_cvd
Create Date: 2026-04-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '009_portfolio_system'
down_revision = '008_liquidations_cvd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Portfolio table - tracks assets in user's portfolio
    op.create_table(
        'portfolio_assets',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('asset_type', sa.String(20), nullable=False, default='crypto'),  # crypto, stock, commodity, forex
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('symbol', name='uq_portfolio_asset_symbol')
    )

    # Positions table - tracks individual trades
    op.create_table(
        'positions',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('portfolio_asset_id', sa.BigInteger(), sa.ForeignKey('portfolio_assets.id'), nullable=False, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('side', sa.String(10), nullable=False),  # long, short
        sa.Column('entry_time', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('entry_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('quantity', sa.Numeric(20, 8), nullable=False),
        sa.Column('exit_time', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('exit_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='open', index=True),  # open, closed, stopped
        sa.Column('pnl', sa.Numeric(20, 8), nullable=True),  # Calculated on close
        sa.Column('pnl_percentage', sa.Numeric(10, 4), nullable=True),  # Calculated on close
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=True),  # For strategy tags, signals, etc.
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_positions_status_entry', 'positions', ['status', 'entry_time'])
    op.create_index('ix_positions_symbol_status', 'positions', ['symbol', 'status'])

    # Portfolio statistics cache (materialized view-like table)
    op.create_table(
        'portfolio_stats',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=True, index=True),  # NULL = global stats
        sa.Column('total_invested', sa.Numeric(20, 8), nullable=False, default=0.0),
        sa.Column('total_pnl', sa.Numeric(20, 8), nullable=False, default=0.0),
        sa.Column('total_pnl_percentage', sa.Numeric(10, 4), nullable=False, default=0.0),
        sa.Column('total_positions', sa.Integer(), nullable=False, default=0),
        sa.Column('open_positions', sa.Integer(), nullable=False, default=0),
        sa.Column('closed_positions', sa.Integer(), nullable=False, default=0),
        sa.Column('winning_positions', sa.Integer(), nullable=False, default=0),
        sa.Column('losing_positions', sa.Integer(), nullable=False, default=0),
        sa.Column('win_rate', sa.Numeric(10, 4), nullable=False, default=0.0),
        sa.Column('avg_win', sa.Numeric(20, 8), nullable=True),
        sa.Column('avg_loss', sa.Numeric(20, 8), nullable=True),
        sa.Column('largest_win', sa.Numeric(20, 8), nullable=True),
        sa.Column('largest_loss', sa.Numeric(20, 8), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('symbol', name='uq_portfolio_stats_symbol')
    )


def downgrade() -> None:
    op.drop_table('portfolio_stats')
    op.drop_table('positions')
    op.drop_table('portfolio_assets')
