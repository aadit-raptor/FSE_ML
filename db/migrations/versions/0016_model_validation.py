"""model validation: the opt-in, the first-actuals date and the reports (PLAN.md 4.6)

Revision ID: 0016
Revises: 0015
Created: 2026-10-08

Two columns on ``deals``, one new table and two more audit actions. The
previous release keeps working while this one rolls out: the new columns
have defaults it never needs to write, and it never reads the new table or
writes the new actions. The restricted app role gets row rights on the new
table through migration 0005's default privileges; ``audit_events`` stays
append-only (the check constraint is replaced, not the grants).

``actuals_first_saved_at`` is left empty for deals that already have
actuals: when they were first entered isn't known, so validation counts
those deals as in-sample, never as newer data.

Downgrade removes the new actions' entries, since the older constraint
can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0016'
down_revision: Union[str, None] = '0015'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed")
NEW_ACTIONS = OLD_ACTIONS + ("validation_opted_in", "validation_opted_out")


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.add_column('deals', sa.Column('actuals_first_saved_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('deals', sa.Column('validation_opt_in', sa.Boolean(), server_default=sa.false(),
                                     nullable=False))
    op.create_table('validation_reports',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('generated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('report', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_validation_reports'))
    )
    op.create_index('ix_validation_reports_generated_at', 'validation_reports', ['generated_at'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN ('validation_opted_in', 'validation_opted_out')")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_validation_reports_generated_at', table_name='validation_reports')
    op.drop_table('validation_reports')
    op.drop_column('deals', 'validation_opt_in')
    op.drop_column('deals', 'actuals_first_saved_at')
