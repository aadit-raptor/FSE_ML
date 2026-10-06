"""benchmark tables: published industry averages by region and country tax rates (PLAN.md 4.3)

Revision ID: 0013
Revises: 0012
Created: 2026-10-07

A new table only, so the previous release keeps working while this one
rolls out (it never reads it). The restricted app role gets row rights on it
through the default privileges migration 0005 set up. Public data, kept
compact (db/benchmarks.py): one row per table, replaced by each refresh.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0013'
down_revision: Union[str, None] = '0012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('benchmark_tables',
    sa.Column('name', sa.String(length=40), nullable=False),
    sa.Column('published', sa.Date(), nullable=True),
    sa.Column('url', sa.String(length=300), nullable=False),
    sa.Column('rows', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('refreshed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('name', name=op.f('pk_benchmark_tables'))
    )


def downgrade() -> None:
    op.drop_table('benchmark_tables')
