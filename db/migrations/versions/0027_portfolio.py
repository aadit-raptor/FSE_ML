"""portfolio: owned deals, their quarterly results and reforecasts (PLAN.md 7.4)

Revision ID: 0027
Revises: 0026
Created: 2026-11-01

Four nullable columns on ``deals`` and six more audit actions. The previous release never reads the columns or
writes the actions, so it keeps working while this one rolls out. ``audit_events`` stays append-only (the check
constraint is replaced, not the grants).

Downgrade drops the columns (holdings, quarters and reforecasts with them) and deletes the new actions' audit
entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0027'
down_revision: Union[str, None] = '0026'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched",
               "file_uploaded", "file_deleted",
               "coverage_added", "coverage_removed", "period_accepted", "period_removed",
               "model_saved", "model_versioned", "model_restored", "model_rolled",
               "report_saved", "report_details_saved",
               "memo_drafted", "memo_written", "memo_deleted",
               "unshared", "commented", "comment_deleted",
               "team_created", "team_renamed", "team_deleted", "team_invited", "team_joined", "team_left",
               "team_member_removed", "team_role_changed")
NEW = ("portfolio_added", "portfolio_changed", "portfolio_removed", "quarters_saved", "quarters_cleared",
       "reforecast_run")
NEW_ACTIONS = OLD_ACTIONS + NEW
COLUMNS = ("portfolio", "quarters", "quarters_updated_at", "reforecast")


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.add_column('deals', sa.Column('portfolio', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('deals', sa.Column('quarters', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('deals', sa.Column('quarters_updated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('deals', sa.Column('reforecast', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    for column in reversed(COLUMNS):
        op.drop_column('deals', column)
