"""files: uploaded documents and where each stands (PLAN.md 6.1)

Revision ID: 0018
Revises: 0017
Created: 2026-10-11

One new table and two more audit actions. The bytes of a file are in the
file store, never here: a row holds the owner, the name the person gave it,
its kind, size and scan status. The previous release keeps working while
this one rolls out: it never reads the table and never writes the new
actions. The restricted app role gets row rights on the table through
migration 0005's default privileges; ``audit_events`` stays append-only (the
check constraint is replaced, not the grants).

Downgrade drops the table (the stored bytes then belong to no row, and the
file store's own retention or a later sweep removes them) and removes the
new actions' entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0018'
down_revision: Union[str, None] = '0017'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched")
NEW_ACTIONS = OLD_ACTIONS + ("file_uploaded", "file_deleted")


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('files',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('kind', sa.String(length=4), nullable=False),
    sa.Column('size', sa.BigInteger(), nullable=False),
    sa.Column('status', sa.String(length=8), server_default='pending', nullable=False),
    sa.Column('reason', sa.String(length=16), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('queued_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('scanned_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("kind IN ('pdf', 'xlsx', 'docx', 'xls', 'doc')", name=op.f('ck_files_kind')),
    sa.CheckConstraint("status IN ('pending', 'scanning', 'clean', 'refused')", name=op.f('ck_files_status')),
    sa.CheckConstraint("reason IS NULL OR reason IN ('wrong_type', 'empty', 'too_large', 'encrypted', "
                       "'macros', 'active_content', 'archive_bomb', 'unreadable', 'scan_failed')",
                       name=op.f('ck_files_reason')),
    sa.CheckConstraint('size > 0', name=op.f('ck_files_size')),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_files_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_files'))
    )
    op.create_index('ix_files_owner_id_created_at', 'files', ['owner_id', 'created_at'], unique=False)
    op.create_index('ix_files_expires_at', 'files', ['expires_at'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN ('file_uploaded', 'file_deleted')")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_files_expires_at', table_name='files')
    op.drop_index('ix_files_owner_id_created_at', table_name='files')
    op.drop_table('files')
