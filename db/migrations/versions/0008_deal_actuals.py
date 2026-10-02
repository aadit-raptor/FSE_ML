"""deal actuals: what happened, for plan vs actual (PLAN.md 2.7)

Revision ID: 0008
Revises: 0007
Created: 2026-10-02

Two nullable columns, so the previous release keeps working while this one
rolls out: it never names them, and the rows it writes leave them empty.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = '0008'
down_revision: Union[str, None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('deals', sa.Column('actuals', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('deals', sa.Column('actuals_updated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('deals', 'actuals_updated_at')
    op.drop_column('deals', 'actuals')
