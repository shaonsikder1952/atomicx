"""Add extra_data and deployed_at fields to evolution_proposals for AGI system.

Revision ID: 011_agi_proposal_fields
Revises: 010_multi_asset_support
Create Date: 2026-04-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '011_agi_proposal_fields'
down_revision = '010_multi_asset_support'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add deployed_at timestamp for tracking deployment time
    op.add_column('evolution_proposals',
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True))

    # Add extra_data JSONB field for AGI-specific metadata (connector files, costs, etc.)
    op.add_column('evolution_proposals',
        sa.Column('extra_data', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('evolution_proposals', 'extra_data')
    op.drop_column('evolution_proposals', 'deployed_at')
