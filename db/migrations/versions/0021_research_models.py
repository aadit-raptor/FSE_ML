"""research_models: saved sector models and their versions (PLAN.md R5)

Revision ID: 0021
Revises: 0020
Created: 2026-10-19

Two new tables and four more audit actions. ``research_models`` is a covered
company's sector model as the analyst keeps it (one per company);
``research_model_versions`` its history, a ``rolled`` version holding the
estimate set beside the results a roll forward brought. The previous release
never reads the tables and never writes the new actions, so it keeps working
while this one rolls out. The restricted app role gets row rights on the
tables through migration 0005's default privileges; ``audit_events`` stays
append-only (the check constraint is replaced, not the grants).

Downgrade drops the tables (the saved models with them) and removes the new
actions' entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0021'
down_revision: Union[str, None] = '0020'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
               "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared",
               "library_switched", "reference_proposed", "reference_reviewed",
               "validation_opted_in", "validation_opted_out", "suggestions_switched",
               "file_uploaded", "file_deleted",
               "coverage_added", "coverage_removed", "period_accepted", "period_removed")
NEW = ("model_saved", "model_versioned", "model_restored", "model_rolled")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('research_models',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('company_id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('scope', sa.String(length=12), nullable=False),
    sa.Column('structure', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('company', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('as_of', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('latest_version', sa.Integer(), server_default='0', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("scope IN ('consolidated', 'standalone')", name=op.f('ck_research_models_scope')),
    sa.ForeignKeyConstraint(['company_id'], ['coverage_companies.id'],
                            name=op.f('fk_research_models_company_id_coverage_companies'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_research_models_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_research_models')),
    sa.UniqueConstraint('company_id', name='uq_research_models_company_id')
    )
    op.create_index('ix_research_models_owner_id', 'research_models', ['owner_id'], unique=False)
    op.create_table('research_model_versions',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('model_id', sa.Uuid(), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('kind', sa.String(length=10), nullable=False),
    sa.Column('label', sa.String(length=120), nullable=True),
    sa.Column('structure', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('company', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('as_of', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('comparison', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind IN ('created', 'saved', 'auto', 'restored', 'rolled')",
                       name=op.f('ck_research_model_versions_kind')),
    sa.ForeignKeyConstraint(['model_id'], ['research_models.id'],
                            name=op.f('fk_research_model_versions_model_id_research_models'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_research_model_versions')),
    sa.UniqueConstraint('model_id', 'number', name='uq_research_model_versions_model_id_number')
    )
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_table('research_model_versions')
    op.drop_index('ix_research_models_owner_id', table_name='research_models')
    op.drop_table('research_models')
