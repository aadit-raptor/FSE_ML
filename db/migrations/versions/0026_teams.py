"""teams: teams, invitations, shared deals, comments and mention emails (PLAN.md 7.2)

Revision ID: 0026
Revises: 0025
Created: 2026-10-30

Six new tables and eleven more audit actions. The previous release never reads
the tables or writes the new actions, so it keeps working while this one rolls
out (a deal shared under this release is simply its owner's alone to the old
one). The restricted app role gets row rights on the tables through migration
0005's default privileges; ``audit_events`` stays append-only (the check
constraint is replaced, not the grants).

Downgrade drops the tables (teams, shares and comments with them) and deletes
the new actions' audit entries, since the older constraint can't hold them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0026'
down_revision: Union[str, None] = '0025'
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
               "memo_drafted", "memo_written", "memo_deleted")
NEW = ("unshared", "commented", "comment_deleted",
       "team_created", "team_renamed", "team_deleted", "team_invited", "team_joined", "team_left",
       "team_member_removed", "team_role_changed")
NEW_ACTIONS = OLD_ACTIONS + NEW


def _actions(actions) -> str:
    return "action IN (" + ", ".join(f"'{a}'" for a in actions) + ")"


def upgrade() -> None:
    op.create_table('teams',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_teams'))
    )
    op.create_table('team_invites',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('team_id', sa.Uuid(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('role', sa.String(length=10), nullable=False),
    sa.Column('uses_left', sa.Integer(), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("role IN ('editor', 'viewer')", name=op.f('ck_team_invites_role')),
    sa.CheckConstraint('uses_left >= 0', name=op.f('ck_team_invites_uses_left')),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], name=op.f('fk_team_invites_team_id_teams'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_team_invites')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_team_invites_token_hash'))
    )
    op.create_index('ix_team_invites_team_id', 'team_invites', ['team_id'], unique=False)
    op.create_table('team_members',
    sa.Column('team_id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.BigInteger(), nullable=False),
    sa.Column('role', sa.String(length=10), nullable=False),
    sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("role IN ('owner', 'editor', 'viewer')", name=op.f('ck_team_members_role')),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], name=op.f('fk_team_members_team_id_teams'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_team_members_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('team_id', 'user_id', name=op.f('pk_team_members'))
    )
    op.create_index('ix_team_members_user_id', 'team_members', ['user_id'], unique=False)
    op.create_table('deal_comments',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('deal_id', sa.Uuid(), nullable=False),
    sa.Column('author_id', sa.BigInteger(), nullable=False),
    sa.Column('body', sa.Text(), nullable=False),
    sa.Column('mentions', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('char_length(body) BETWEEN 1 AND 2000', name=op.f('ck_deal_comments_body')),
    sa.ForeignKeyConstraint(['author_id'], ['users.id'], name=op.f('fk_deal_comments_author_id_users'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['deal_id'], ['deals.id'], name=op.f('fk_deal_comments_deal_id_deals'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_deal_comments'))
    )
    op.create_index('ix_deal_comments_deal_id_created_at', 'deal_comments', ['deal_id', 'created_at'], unique=False)
    op.create_table('deal_shares',
    sa.Column('deal_id', sa.Uuid(), nullable=False),
    sa.Column('team_id', sa.Uuid(), nullable=False),
    sa.Column('shared_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['deal_id'], ['deals.id'], name=op.f('fk_deal_shares_deal_id_deals'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], name=op.f('fk_deal_shares_team_id_teams'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('deal_id', 'team_id', name=op.f('pk_deal_shares'))
    )
    op.create_index('ix_deal_shares_team_id', 'deal_shares', ['team_id'], unique=False)
    op.create_table('mention_notices',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('comment_id', sa.Uuid(), nullable=False),
    sa.Column('recipient_id', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome', sa.String(length=12), nullable=True),
    sa.CheckConstraint("outcome IN ('sent', 'no_address', 'expired', 'failed')", name=op.f('ck_mention_notices_outcome')),
    sa.ForeignKeyConstraint(['comment_id'], ['deal_comments.id'], name=op.f('fk_mention_notices_comment_id_deal_comments'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['recipient_id'], ['users.id'], name=op.f('fk_mention_notices_recipient_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_mention_notices'))
    )
    op.create_index('ix_mention_notices_recipient_id_sent_at', 'mention_notices', ['recipient_id', 'sent_at'], unique=False)
    op.create_index('ix_mention_notices_sent_at', 'mention_notices', ['sent_at'], unique=False)
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(NEW_ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM audit_events WHERE action IN (" + ", ".join(f"'{a}'" for a in NEW) + ")")
    op.drop_constraint(op.f('ck_audit_events_action'), 'audit_events', type_='check')
    op.create_check_constraint(op.f('ck_audit_events_action'), 'audit_events', _actions(OLD_ACTIONS))
    op.drop_index('ix_mention_notices_sent_at', table_name='mention_notices')
    op.drop_index('ix_mention_notices_recipient_id_sent_at', table_name='mention_notices')
    op.drop_table('mention_notices')
    op.drop_index('ix_deal_shares_team_id', table_name='deal_shares')
    op.drop_table('deal_shares')
    op.drop_index('ix_deal_comments_deal_id_created_at', table_name='deal_comments')
    op.drop_table('deal_comments')
    op.drop_index('ix_team_members_user_id', table_name='team_members')
    op.drop_table('team_members')
    op.drop_index('ix_team_invites_team_id', table_name='team_invites')
    op.drop_table('team_invites')
    op.drop_table('teams')
