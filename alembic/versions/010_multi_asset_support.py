"""Multi-asset support with universal data connectors.

Adds status tracking, data source routing, and initialization progress
for autonomous multi-asset intelligence system.

Revision ID: 010_multi_asset_support
Revises: 009_portfolio_system
Create Date: 2026-04-06

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '010_multi_asset_support'
down_revision = '009_portfolio_system'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add multi-asset tracking columns to portfolio_assets
    op.add_column('portfolio_assets',
        sa.Column('status', sa.String(20), nullable=False, server_default='pending', index=True))
    op.add_column('portfolio_assets',
        sa.Column('data_source', sa.String(50), nullable=True, index=True))
    op.add_column('portfolio_assets',
        sa.Column('backfill_progress', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('portfolio_assets',
        sa.Column('error_message', sa.Text(), nullable=True))
    op.add_column('portfolio_assets',
        sa.Column('last_update', sa.DateTime(timezone=True), nullable=True, index=True))
    op.add_column('portfolio_assets',
        sa.Column('extra_data', sa.JSON(), nullable=True))

    # Set existing assets to 'active' status with 100% progress
    op.execute("""
        UPDATE portfolio_assets
        SET status = 'active',
            backfill_progress = 100,
            last_update = NOW()
        WHERE is_active = true
    """)

    # Create index for fast status queries
    op.create_index('ix_portfolio_assets_status_update', 'portfolio_assets', ['status', 'last_update'])


def downgrade() -> None:
    op.drop_index('ix_portfolio_assets_status_update', 'portfolio_assets')
    op.drop_column('portfolio_assets', 'extra_data')
    op.drop_column('portfolio_assets', 'last_update')
    op.drop_column('portfolio_assets', 'error_message')
    op.drop_column('portfolio_assets', 'backfill_progress')
    op.drop_column('portfolio_assets', 'data_source')
    op.drop_column('portfolio_assets', 'status')
