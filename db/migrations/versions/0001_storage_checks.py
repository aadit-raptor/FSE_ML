"""storage checks: database size readings against the free storage limit

Revision ID: 0001
Revises:
Created: 2026-09-16
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "storage_checks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("database_bytes", sa.BigInteger(), nullable=False),
        sa.Column("limit_bytes", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_storage_checks")),
    )
    op.create_index(op.f("ix_storage_checks_checked_at"), "storage_checks", ["checked_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_storage_checks_checked_at"), table_name="storage_checks")
    op.drop_table("storage_checks")
