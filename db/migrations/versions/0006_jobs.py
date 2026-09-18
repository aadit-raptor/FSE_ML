"""jobs: background jobs and the scheduled-run log (PLAN.md 1.9)

Revision ID: 0006
Revises: 0005
Created: 2026-09-18

New tables only, so the previous release keeps working while this one rolls
out (it never reads them). The restricted app role gets row rights on both
through the default privileges migration 0005 set up.

Both stay small: a job's inputs are cleared when it ends and its result a few
hours later, finished jobs are deleted after a week, and the scheduled-run log
keeps the newest runs per task (jobs/queue.py, jobs/scheduled.py).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0006'
down_revision: Union[str, None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('jobs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('owner', sa.String(length=255), nullable=False),
    sa.Column('kind', sa.String(length=40), nullable=False),
    sa.Column('status', sa.String(length=10), server_default='queued', nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('error', sa.String(length=500), nullable=True),
    sa.Column('error_status', sa.Integer(), nullable=True),
    sa.Column('progress', sa.Float(), server_default='0', nullable=False),
    sa.Column('stage', sa.String(length=80), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
    sa.Column('cancel_requested', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('worker', sa.String(length=80), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')", name=op.f('ck_jobs_status')),
    sa.CheckConstraint('progress >= 0 AND progress <= 1', name=op.f('ck_jobs_progress')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_jobs'))
    )
    op.create_index('ix_jobs_owner_created_at', 'jobs', ['owner', 'created_at'], unique=False)
    op.create_index('ix_jobs_status_created_at', 'jobs', ['status', 'created_at'], unique=False)
    op.create_table('scheduled_runs',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('task', sa.String(length=40), nullable=False),
    sa.Column('status', sa.String(length=10), nullable=False),
    sa.Column('trigger', sa.String(length=20), nullable=False),
    sa.Column('workflow', sa.String(length=80), nullable=False),
    sa.Column('github_run_id', sa.BigInteger(), nullable=True),
    sa.Column('summary', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('error', sa.String(length=500), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('succeeded', 'failed')", name=op.f('ck_scheduled_runs_status')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_scheduled_runs'))
    )
    op.create_index('ix_scheduled_runs_task_started_at', 'scheduled_runs', ['task', 'started_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_scheduled_runs_task_started_at', table_name='scheduled_runs')
    op.drop_table('scheduled_runs')
    op.drop_index('ix_jobs_status_created_at', table_name='jobs')
    op.drop_index('ix_jobs_owner_created_at', table_name='jobs')
    op.drop_table('jobs')
