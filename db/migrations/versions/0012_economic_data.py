"""economic data: series by country and source, the ECB's exchange rates (PLAN.md 4.2)

Revision ID: 0012
Revises: 0011
Created: 2026-10-07

New tables only, so the previous release keeps working while this one rolls
out (it never reads them). The restricted app role gets row rights on them
through the default privileges migration 0005 set up. Public data, kept
compact (db/economy.py): the newest observations per series, a rolling year
of exchange rates.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0012'
down_revision: Union[str, None] = '0011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('economic_series',
    sa.Column('key', sa.String(length=48), nullable=False),
    sa.Column('indicator', sa.String(length=20), nullable=False),
    sa.Column('area', sa.String(length=8), nullable=False),
    sa.Column('source', sa.String(length=12), nullable=False),
    sa.Column('source_series', sa.String(length=80), nullable=False),
    sa.Column('frequency', sa.String(length=1), nullable=False),
    sa.Column('url', sa.String(length=300), nullable=False),
    sa.Column('observations', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('refreshed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('key', name=op.f('pk_economic_series'))
    )
    op.create_table('exchange_rates',
    sa.Column('rate_date', sa.Date(), nullable=False),
    sa.Column('rates', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('source', sa.String(length=12), server_default=sa.text("'ecb'"), nullable=False),
    sa.Column('refreshed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('rate_date', name=op.f('pk_exchange_rates'))
    )


def downgrade() -> None:
    op.drop_table('exchange_rates')
    op.drop_table('economic_series')
