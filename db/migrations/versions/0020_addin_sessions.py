"""addin_sessions: the Excel add-in's sign-ins (PLAN.md R3)

Revision ID: 0020
Revises: 0019
Created: 2026-10-15

One new table: a sign-in of the Excel add-in, kept as the SHA-256 of its
pass and when it ends. The previous release never reads it, so it keeps
working while this one rolls out. The restricted app role gets row rights on
it through migration 0005's default privileges.

Downgrade drops the table, which signs every add-in out.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0020'
down_revision: Union[str, None] = '0019'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('addin_sessions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_addin_sessions_owner_id_users'),
                            ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_addin_sessions')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_addin_sessions_token_hash'))
    )
    op.create_index('ix_addin_sessions_owner_id_created_at', 'addin_sessions', ['owner_id', 'created_at'],
                    unique=False)
    op.create_index('ix_addin_sessions_expires_at', 'addin_sessions', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_addin_sessions_expires_at', table_name='addin_sessions')
    op.drop_index('ix_addin_sessions_owner_id_created_at', table_name='addin_sessions')
    op.drop_table('addin_sessions')
