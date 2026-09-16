"""deals and deal_versions: saved deals with version history; users.settings

Revision ID: 0003
Revises: 0002
Created: 2026-09-16

Adds two tables and one column with a default, so the previous release keeps
working while this one rolls out (it never reads them). Deleting an account
deletes its deals, and deleting a deal its versions (ON DELETE CASCADE).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Settings overrides move from the browser to the account
    op.add_column("users", sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()),
                                     server_default=sa.text("'{}'::jsonb"), nullable=False))

    op.create_table(
        "deals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("latest_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name=op.f("fk_deals_owner_id_users"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_deals")),
    )
    # The deal list: one owner's deals, most recently edited first
    op.create_index("ix_deals_owner_id_updated_at", "deals", ["owner_id", "updated_at"])

    op.create_table(
        "deal_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("deal_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=True),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.CheckConstraint("kind IN ('created', 'saved', 'auto', 'restored')",
                           name=op.f("ck_deal_versions_kind")),
        sa.ForeignKeyConstraint(["deal_id"], ["deals.id"],
                                name=op.f("fk_deal_versions_deal_id_deals"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_deal_versions")),
        # Also the index for "this deal's versions"
        sa.UniqueConstraint("deal_id", "number", name="uq_deal_versions_deal_id_number"),
    )


def downgrade() -> None:
    op.drop_table("deal_versions")
    op.drop_index("ix_deals_owner_id_updated_at", table_name="deals")
    op.drop_table("deals")
    op.drop_column("users", "settings")
