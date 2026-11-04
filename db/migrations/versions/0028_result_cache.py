"""result_cache: cached model results when Upstash can't be used (PLAN.md 8.1)

Revision ID: 0028
Revises: 0027
Created: 2026-11-04

A new table, the fallback store of api/result_cache.py. Its rows hold an
encrypted answer under a hash, nothing readable; they expire within hours and
the table is held to a few megabytes. The previous release never reads it, so
it keeps working while this one rolls out. The restricted app role gets row
rights on it through migration 0005's default privileges.

Downgrade drops the table: nothing in it is anything but a cache.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0028'
down_revision: Union[str, None] = '0027'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('result_cache',
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('value', sa.LargeBinary(), nullable=False),
    sa.Column('size', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('key', name=op.f('pk_result_cache'))
    )
    op.create_index(op.f('ix_result_cache_created_at'), 'result_cache', ['created_at'], unique=False)
    op.create_index(op.f('ix_result_cache_expires_at'), 'result_cache', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_result_cache_expires_at'), table_name='result_cache')
    op.drop_index(op.f('ix_result_cache_created_at'), table_name='result_cache')
    op.drop_table('result_cache')
