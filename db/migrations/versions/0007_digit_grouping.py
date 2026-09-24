"""digit grouping: how the account groups long numbers (PLAN.md 2.3a)

Revision ID: 0007
Revises: 0006
Created: 2026-09-24

One column with a server default, so the previous release keeps working
while this one rolls out: it never names the column, and the rows it writes
get 'locale', which is what it showed anyway.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0007'
down_revision: Union[str, None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('digit_grouping', sa.String(length=10),
                                     server_default='locale', nullable=False))
    op.create_check_constraint(op.f('ck_users_digit_grouping'), 'users',
                               "digit_grouping IN ('locale', 'thousands', 'lakh')")


def downgrade() -> None:
    op.drop_constraint(op.f('ck_users_digit_grouping'), 'users', type_='check')
    op.drop_column('users', 'digit_grouping')
