"""usage_counters: shared usage counters when Upstash Redis can't be used

Revision ID: 0004
Revises: 0003
Created: 2026-09-16

A new table only, so the previous release keeps working while this one rolls
out (it never reads it). Rows expire within a day or two (api/usage.py).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "usage_counters",
        sa.Column("key", sa.String(length=160), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_usage_counters")),
    )
    # Pruning expired counters
    op.create_index(op.f("ix_usage_counters_expires_at"), "usage_counters", ["expires_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_usage_counters_expires_at"), table_name="usage_counters")
    op.drop_table("usage_counters")
