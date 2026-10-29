"""deal_memos: investment memos of saved deal versions (PLAN.md 6.3)

Revision ID: 0025
Revises: 0024
Created: 2026-10-26

A new table, three more audit actions and a fourth AI usage kind.
``deal_memos`` holds one memo per saved version of a deal: the version's model
run as the memo reads it and the linking sentences last written. The previous
release never reads the table, writes the new actions or counts the new kind,
so it keeps working while this one rolls out. The restricted app role gets row
rights on the table through migration 0005's default privileges;
``audit_events`` stays append-only (the check constraint is replaced, not the
grants).

Downgrade drops the table (the memos with it) and deletes the new actions'
audit entries and the ``memo`` usage rows, since the older constraints can't
hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0025'
down_revision: Union[str, None] = '0024'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched",
               "file_uploaded", "file_deleted",
               "coverage_added", "coverage_removed", "period_accepted", "period_removed",
               "model_saved", "model_versioned", "model_restored", "model_rolled",
               "report_saved", "report_details_saved")
NEW = ("memo_drafted", "memo_written", "memo_deleted")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('deal_memos',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('deal_id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('run', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('sentences', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['deal_id'], ['deals.id'], name=op.f('fk_deal_memos_deal_id_deals'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_deal_memos_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_deal_memos')),
    sa.UniqueConstraint('deal_id', 'version', name='uq_deal_memos_deal_id_version')
    )
    op.create_index('ix_deal_memos_owner_id', 'deal_memos', ['owner_id'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))
    op.drop_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', type_='check')
    op.create_check_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', "kind IN ('read', 'write', 'extract', 'memo')")


def downgrade() -> None:
    op.execute("DELETE FROM ai_usage WHERE kind = 'memo'")
    op.drop_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', type_='check')
    op.create_check_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', "kind IN ('read', 'write', 'extract')")
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_deal_memos_owner_id', table_name='deal_memos')
    op.drop_table('deal_memos')
