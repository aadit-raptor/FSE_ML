"""ai_usage: what each person's AI requests used, a day (PLAN.md R9)

Revision ID: 0023
Revises: 0022
Created: 2026-10-20

One new table: pages read, requests, tokens and their cost at the paid price
list, per person, UTC day, kind (a results read, a note's sentences) and
tier (free or billed). Counts only, never anything from a document. The
previous release never reads it, so it keeps working while this one rolls
out; the restricted app role gets row rights through migration 0005's
default privileges. About 100 bytes a row: a person reading every day for
the 400 days kept is under 50 KB.

Downgrade drops the table (the counts with it).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = '0023'
down_revision: Union[str, None] = '0022'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('ai_usage',
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('day', sa.Date(), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('tier', sa.String(length=16), nullable=False),
    sa.Column('requests', sa.Integer(), server_default='0', nullable=False),
    sa.Column('pages', sa.Integer(), server_default='0', nullable=False),
    sa.Column('input_tokens', sa.BigInteger(), server_default='0', nullable=False),
    sa.Column('output_tokens', sa.BigInteger(), server_default='0', nullable=False),
    sa.Column('unpriced', sa.Integer(), server_default='0', nullable=False),
    sa.Column('cost_amount', sa.Numeric(precision=24, scale=6), server_default='0', nullable=False),
    sa.Column('cost_currency', sa.String(length=3), server_default='USD', nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind IN ('read', 'write')", name=op.f('ck_ai_usage_kind')),
    sa.CheckConstraint("tier IN ('free', 'billed')", name=op.f('ck_ai_usage_tier')),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_ai_usage_owner_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('owner_id', 'day', 'kind', 'tier', name=op.f('pk_ai_usage'))
    )
    op.create_index('ix_ai_usage_day', 'ai_usage', ['day'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_ai_usage_day', table_name='ai_usage')
    op.drop_table('ai_usage')
