"""audit events: an append-only log of what was done to deals and settings (PLAN.md 3.3)

Revision ID: 0010
Revises: 0009
Created: 2026-10-06

A new table, so the previous release keeps working while this one rolls out
(it never names it). **Append-only for the API**: the API's role
(``fse_app``) keeps SELECT and INSERT and loses the UPDATE and DELETE that
migration 0005's default privileges gave it, so no bug and no stolen API
connection can rewrite the history. TRUNCATE was never granted.

The one change allowed is compaction, ``audit_compact(cutoff)``: it merges
``edited`` entries older than ``cutoff`` into one per user, deal and UTC day
(``count`` and ``last_at`` keep how many and until when, ``detail.fields``
the union of the fields). It is ``SECURITY DEFINER``, so it runs as the
schema owner, and only ``fse_app`` may call it. It never touches the last
seven days whatever ``cutoff`` says (db/audit.py ``COMPACT_AFTER_DAYS``), so
even the API's own role can't blur recent history. Returns the rows it removed.

A restore leaves grants out (ops/backup.py); DEPLOY.md "Restoring" puts these
back, and ``/api/health/database`` reports ``audit_log_writable`` until it does.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = '0010'
down_revision: Union[str, None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ROLE = "fse_app"
ACTIONS = ("created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
           "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed", "shared")

COMPACT = """
CREATE FUNCTION audit_compact(cutoff timestamptz) RETURNS integer
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, public
AS $$
  WITH old AS (
    DELETE FROM public.audit_events e
    USING (
      SELECT user_id, deal_id, date_trunc('day', occurred_at AT TIME ZONE 'UTC') AS day
      FROM public.audit_events
      WHERE action = 'edited' AND occurred_at < LEAST(cutoff, now() - interval '7 days')
      GROUP BY 1, 2, 3 HAVING count(*) > 1
    ) g
    WHERE e.action = 'edited' AND e.occurred_at < LEAST(cutoff, now() - interval '7 days')
      AND e.user_id = g.user_id AND e.deal_id = g.deal_id
      AND date_trunc('day', e.occurred_at AT TIME ZONE 'UTC') = g.day
    RETURNING e.*
  ), merged AS (
    SELECT user_id, deal_id, date_trunc('day', occurred_at AT TIME ZONE 'UTC') AS day,
           min(occurred_at) AS first_at, max(coalesce(last_at, occurred_at)) AS until_at,
           sum(count)::integer AS n
    FROM old GROUP BY 1, 2, 3
  ), names AS (
    SELECT o.user_id, o.deal_id, date_trunc('day', o.occurred_at AT TIME ZONE 'UTC') AS day,
           jsonb_agg(DISTINCT f.name COLLATE "C" ORDER BY f.name COLLATE "C") AS fields
    FROM old o
    CROSS JOIN LATERAL jsonb_array_elements_text(coalesce(o.detail -> 'fields', '[]'::jsonb)) AS f(name)
    GROUP BY 1, 2, 3
  ), added AS (
    INSERT INTO public.audit_events (user_id, deal_id, action, detail, count, occurred_at, last_at)
    SELECT m.user_id, m.deal_id, 'edited',
           jsonb_build_object('fields', coalesce(n.fields, '[]'::jsonb)), m.n, m.first_at, m.until_at
    FROM merged m
    LEFT JOIN names n ON n.user_id = m.user_id AND n.deal_id = m.deal_id AND n.day = m.day
    RETURNING 1
  )
  SELECT ((SELECT count(*) FROM old) - (SELECT count(*) FROM added))::integer
$$
"""


def upgrade() -> None:
    op.create_table(
        'audit_events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('deal_id', sa.Uuid(), nullable=True),
        sa.Column('action', sa.String(length=20), nullable=False),
        sa.Column('detail', postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column('count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('occurred_at', sa.DateTime(timezone=True), server_default=sa.text('now()'),
                  nullable=False),
        sa.Column('last_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("action IN (" + ", ".join(f"'{a}'" for a in ACTIONS) + ")",
                           name=op.f('ck_audit_events_action')),
        sa.CheckConstraint('count >= 1', name=op.f('ck_audit_events_count')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_audit_events_user_id_users'),
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_events')),
    )
    op.create_index('ix_audit_events_deal_id_occurred_at', 'audit_events', ['deal_id', 'occurred_at'])
    op.create_index('ix_audit_events_user_id_occurred_at', 'audit_events', ['user_id', 'occurred_at'])
    # Append-only for the API's role (it got these through 0005's default privileges)
    op.execute(f"REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM {APP_ROLE}")
    op.execute(COMPACT)
    op.execute("REVOKE ALL ON FUNCTION audit_compact(timestamptz) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION audit_compact(timestamptz) TO {APP_ROLE}")


def downgrade() -> None:
    op.execute("DROP FUNCTION audit_compact(timestamptz)")
    op.drop_index('ix_audit_events_user_id_occurred_at', table_name='audit_events')
    op.drop_index('ix_audit_events_deal_id_occurred_at', table_name='audit_events')
    op.drop_table('audit_events')
