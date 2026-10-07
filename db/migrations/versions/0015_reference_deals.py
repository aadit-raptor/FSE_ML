"""reference deals: proposed reference transactions and their reviews (PLAN.md 4.5b)

Revision ID: 0015
Revises: 0014
Created: 2026-10-07

Two new tables and two more audit actions. The previous release keeps
working while this one rolls out: it never reads the new tables and never
writes the new actions. The restricted app role gets row rights on both
tables through migration 0005's default privileges; ``audit_events`` stays
append-only (the check constraint is replaced, not the grants).

Downgrade removes the new actions' entries, since the older constraint
can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0015'
down_revision: Union[str, None] = '0014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched")
NEW_ACTIONS = OLD_ACTIONS + ("reference_proposed", "reference_reviewed")


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('reference_deals',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('key', sa.String(length=80), nullable=False),
    sa.Column('content', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('origin', sa.String(length=12), nullable=False),
    sa.Column('status', sa.String(length=12), server_default='proposed', nullable=False),
    sa.Column('proposed_by', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('proposed', 'approved', 'rejected', 'superseded')",
                       name=op.f('ck_reference_deals_status')),
    sa.CheckConstraint("origin IN ('repository', 'user')", name=op.f('ck_reference_deals_origin')),
    sa.ForeignKeyConstraint(['proposed_by'], ['users.id'], name=op.f('fk_reference_deals_proposed_by_users'),
                            ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_reference_deals')),
    sa.UniqueConstraint('key', 'content_hash', name='uq_reference_deals_key_content_hash')
    )
    op.create_index('ix_reference_deals_status', 'reference_deals', ['status'], unique=False)
    op.create_table('reference_reviews',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('reference_id', sa.Uuid(), nullable=False),
    sa.Column('reviewer_id', sa.BigInteger(), nullable=True),
    sa.Column('verdict', sa.String(length=8), nullable=False),
    sa.Column('reason', sa.String(length=20), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("verdict IN ('approve', 'reject')", name=op.f('ck_reference_reviews_verdict')),
    sa.CheckConstraint("reason IS NULL OR reason IN ('figure_wrong', 'source_wrong', 'not_a_buyout', "
                       "'duplicate', 'other')", name=op.f('ck_reference_reviews_reason')),
    sa.ForeignKeyConstraint(['reference_id'], ['reference_deals.id'],
                            name=op.f('fk_reference_reviews_reference_id_reference_deals'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['reviewer_id'], ['users.id'], name=op.f('fk_reference_reviews_reviewer_id_users'),
                            ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_reference_reviews')),
    sa.UniqueConstraint('reference_id', 'reviewer_id', name='uq_reference_reviews_reference_id_reviewer_id')
    )
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN ('reference_proposed', 'reference_reviewed')")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_table('reference_reviews')
    op.drop_index('ix_reference_deals_status', table_name='reference_deals')
    op.drop_table('reference_deals')
