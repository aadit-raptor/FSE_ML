"""model stamp on saved deals and versions (PLAN.md 3.1)

Revision ID: 0009
Revises: 0008
Created: 2026-10-04

Which model a deal was saved with, and the IRR and MOIC it gave then
(core/model_version.py ``saved_stamp``), so reopening it can say whether its
results have changed. Nullable: the previous release never names the column,
and every deal saved before this one keeps NULL (``unknown`` on reopening).
About 200 bytes a row.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = '0009'
down_revision: Union[str, None] = '0008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('deals', sa.Column('model', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('deal_versions', sa.Column('model', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('deal_versions', 'model')
    op.drop_column('deals', 'model')
