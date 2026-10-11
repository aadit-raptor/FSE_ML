"""personalized defaults: the account's off switch (PLAN.md 5.9)

Revision ID: 0017
Revises: 0016
Created: 2026-10-11

One column on ``users`` and one more audit action. Suggestions of a person's
usual figures are worked out from their own saved deals each time they are
asked for, so there is no table of learned figures: the only thing stored is
whether they want them. The previous release keeps working while this one
rolls out: the column has a default it never needs to write, and it never
writes the new action. ``audit_events`` stays append-only (the check
constraint is replaced, not the grants).

Downgrade removes the new action's entries, since the older constraint can't
hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0017'
down_revision: Union[str, None] = '0016'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out")
NEW_ACTIONS = OLD_ACTIONS + ("suggestions_switched",)


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.add_column('users', sa.Column('personalized_defaults', sa.Boolean(), server_default=sa.true(),
                                     nullable=False))
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action = 'suggestions_switched'")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_column('users', 'personalized_defaults')
