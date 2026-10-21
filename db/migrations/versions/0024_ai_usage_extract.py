"""ai_usage: a third kind, a document read for a deal (PLAN.md 6.2)

Revision ID: 0024
Revises: 0023
Created: 2026-10-21

Widens the kind check to take ``extract``. The previous release never writes
it, so it keeps working while this one rolls out. Downgrade deletes the
``extract`` rows (their counts) before narrowing the check again.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '0024'
down_revision: Union[str, None] = '0023'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', type_='check')
    op.create_check_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', "kind IN ('read', 'write', 'extract')")


def downgrade() -> None:
    op.execute("DELETE FROM ai_usage WHERE kind = 'extract'")
    op.drop_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', type_='check')
    op.create_check_constraint(op.f('ck_ai_usage_kind'), 'ai_usage', "kind IN ('read', 'write')")
