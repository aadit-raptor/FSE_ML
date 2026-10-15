"""coverage: covered companies and their accepted periods (PLAN.md R2)

Revision ID: 0019
Revises: 0018
Created: 2026-10-15

Two new tables and four more audit actions. ``coverage_companies`` is a
company an analyst follows; ``coverage_periods`` one accepted period of it
(a quarter or a year, standalone or consolidated) with its checked figures
as JSONB, written only when the analyst accepts a read. The previous release
keeps working while this one rolls out: it never reads the tables and never
writes the new actions. The restricted app role gets row rights on the
tables through migration 0005's default privileges; ``audit_events`` stays
append-only (the check constraint is replaced, not the grants).

Downgrade drops the tables (the analysts' accepted figures with them) and
removes the new actions' entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0019'
down_revision: Union[str, None] = '0018'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched",
               "file_uploaded", "file_deleted")
NEW = ("coverage_added", "coverage_removed", "period_accepted", "period_removed")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('coverage_companies',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_coverage_companies_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_coverage_companies'))
    )
    op.create_index('ix_coverage_companies_owner_id_created_at', 'coverage_companies',
                    ['owner_id', 'created_at'], unique=False)
    op.create_table('coverage_periods',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('company_id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.Column('months', sa.SmallInteger(), nullable=False),
    sa.Column('scope', sa.String(length=12), nullable=False),
    sa.Column('unit', sa.String(length=9), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=True),
    sa.Column('file_id', sa.Uuid(), nullable=True),
    sa.Column('figures', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('accepted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("scope IN ('consolidated', 'standalone')", name=op.f('ck_coverage_periods_scope')),
    sa.CheckConstraint("unit IN ('units', 'thousands', 'lakhs', 'millions', 'crores', 'billions')",
                       name=op.f('ck_coverage_periods_unit')),
    sa.CheckConstraint('months IN (3, 6, 9, 12)', name=op.f('ck_coverage_periods_months')),
    sa.ForeignKeyConstraint(['company_id'], ['coverage_companies.id'],
                            name=op.f('fk_coverage_periods_company_id_coverage_companies'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['file_id'], ['files.id'], name=op.f('fk_coverage_periods_file_id_files'),
                            ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_coverage_periods_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_coverage_periods')),
    sa.UniqueConstraint('company_id', 'period_end', 'months', 'scope',
                        name='uq_coverage_periods_company_period_scope')
    )
    op.create_index('ix_coverage_periods_owner_id_accepted_at', 'coverage_periods',
                    ['owner_id', 'accepted_at'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_coverage_periods_owner_id_accepted_at', table_name='coverage_periods')
    op.drop_table('coverage_periods')
    op.drop_index('ix_coverage_companies_owner_id_created_at', table_name='coverage_companies')
    op.drop_table('coverage_companies')
