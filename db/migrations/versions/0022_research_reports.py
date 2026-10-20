"""research_reports: results notes and the account's report details (PLAN.md R8)

Revision ID: 0022
Revises: 0021
Created: 2026-10-19

Two new tables and two more audit actions. ``research_reports`` is a covered
company's results note as the analyst keeps it (what they typed for it and
the linking sentences last written; never a figure, which is read from the
saved model when the note is drawn); ``research_report_details`` the firm
name and disclosures every note of the account carries. The previous release
never reads the tables and never writes the new actions, so it keeps working
while this one rolls out. The restricted app role gets row rights on the
tables through migration 0005's default privileges; ``audit_events`` stays
append-only (the check constraint is replaced, not the grants).

Downgrade drops the tables (the notes with them) and removes the new
actions' entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0022'
down_revision: Union[str, None] = '0021'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched",
               "file_uploaded", "file_deleted",
               "coverage_added", "coverage_removed", "period_accepted", "period_removed",
               "model_saved", "model_versioned", "model_restored", "model_rolled")
NEW = ("report_saved", "report_details_saved")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('research_reports',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('company_id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('fields', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('sentences', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['company_id'], ['coverage_companies.id'],
                            name=op.f('fk_research_reports_company_id_coverage_companies'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_research_reports_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_research_reports')),
    sa.UniqueConstraint('company_id', name='uq_research_reports_company_id')
    )
    op.create_index('ix_research_reports_owner_id', 'research_reports', ['owner_id'], unique=False)
    op.create_table('research_report_details',
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('firm', sa.String(length=200), server_default='', nullable=False),
    sa.Column('disclosures', sa.Text(), server_default='', nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_research_report_details_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('owner_id', name=op.f('pk_research_report_details'))
    )
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_table('research_report_details')
    op.drop_index('ix_research_reports_owner_id', table_name='research_reports')
    op.drop_table('research_reports')
