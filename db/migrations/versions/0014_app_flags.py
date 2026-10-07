"""app flags: site-wide switches, the first being the reference library's (PLAN.md 4.5)

Revision ID: 0014
Revises: 0013
Created: 2026-10-07

A new table and one more audit action. The previous release keeps working
while this one rolls out: it never reads ``app_flags`` and never writes the
new action. The restricted app role gets row rights on the table through
migration 0005's default privileges; ``audit_events`` stays append-only (the
check constraint is replaced, not the grants).

Downgrade removes any ``library_switched`` entries, since the older
constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0014'
down_revision: Union[str, None] = '0013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared")
NEW_ACTIONS = OLD_ACTIONS + ("library_switched",)


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('app_flags',
    sa.Column('name', sa.String(length=40), nullable=False),
    sa.Column('enabled', sa.Boolean(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_by', sa.BigInteger(), nullable=True),
    sa.ForeignKeyConstraint(['updated_by'], ['users.id'], name=op.f('fk_app_flags_updated_by_users'),
                            ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('name', name=op.f('pk_app_flags'))
    )
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action = 'library_switched'")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_table('app_flags')
