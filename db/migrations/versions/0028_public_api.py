"""public_api: API keys, webhooks and their deliveries (PLAN.md 7.7)

Revision ID: 0028
Revises: 0027
Created: 2026-11-04

Three new tables and four more audit actions. The previous release never reads the tables or writes the actions,
so it keeps working while this one rolls out. The restricted app role gets row rights on the tables through
migration 0005's default privileges; ``audit_events`` stays append-only (the check constraint is replaced, not
the grants).

Downgrade drops the tables (every key stops, every webhook goes) and deletes the new actions' audit entries,
since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0028'
down_revision: Union[str, None] = '0027'
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
               "team_member_removed", "team_role_changed",
               "portfolio_added", "portfolio_changed", "portfolio_removed", "quarters_saved", "quarters_cleared",
               "reforecast_run")
NEW = ("api_key_created", "api_key_revoked", "webhook_added", "webhook_removed")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('api_keys',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('name', sa.String(length=60), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('hint', sa.String(length=4), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_api_keys_owner_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_api_keys')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_api_keys_token_hash'))
    )
    op.create_index('ix_api_keys_owner_id_created_at', 'api_keys', ['owner_id', 'created_at'], unique=False)
    op.create_table('webhooks',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('url', sa.String(length=500), nullable=False),
    sa.Column('secret', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_webhooks_owner_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_webhooks'))
    )
    op.create_index('ix_webhooks_owner_id_created_at', 'webhooks', ['owner_id', 'created_at'], unique=False)
    op.create_table('webhook_deliveries',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('webhook_id', sa.Uuid(), nullable=False),
    sa.Column('job_id', sa.Uuid(), nullable=True),
    sa.Column('event', sa.String(length=20), nullable=False),
    sa.Column('state', sa.String(length=10), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
    sa.Column('next_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_status', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("event IN ('job.finished', 'ping')", name=op.f('ck_webhook_deliveries_event')),
    sa.CheckConstraint("state IN ('waiting', 'due', 'delivered', 'failed')", name=op.f('ck_webhook_deliveries_state')),
    sa.ForeignKeyConstraint(['webhook_id'], ['webhooks.id'], name=op.f('fk_webhook_deliveries_webhook_id_webhooks'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_webhook_deliveries'))
    )
    op.create_index('ix_webhook_deliveries_job_id', 'webhook_deliveries', ['job_id'], unique=False)
    op.create_index('ix_webhook_deliveries_state_next_at', 'webhook_deliveries', ['state', 'next_at'], unique=False)
    op.create_index('ix_webhook_deliveries_webhook_id_created_at', 'webhook_deliveries', ['webhook_id', 'created_at'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_webhook_deliveries_webhook_id_created_at', table_name='webhook_deliveries')
    op.drop_index('ix_webhook_deliveries_state_next_at', table_name='webhook_deliveries')
    op.drop_index('ix_webhook_deliveries_job_id', table_name='webhook_deliveries')
    op.drop_table('webhook_deliveries')
    op.drop_index('ix_webhooks_owner_id_created_at', table_name='webhooks')
    op.drop_table('webhooks')
    op.drop_index('ix_api_keys_owner_id_created_at', table_name='api_keys')
    op.drop_table('api_keys')
