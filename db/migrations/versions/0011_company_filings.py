"""company filings: companies, their yearly figures, EDINET's report index (PLAN.md 4.1)

Revision ID: 0011
Revises: 0010
Created: 2026-10-06

New tables only, so the previous release keeps working while this one rolls
out (it never reads them). The restricted app role gets row rights on them
through the default privileges migration 0005 set up. Public filing data,
kept compact (db/companies.py): a summary per company and year, never a
filing.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0011'
down_revision: Union[str, None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('companies',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('source', sa.String(length=20), nullable=False),
    sa.Column('source_id', sa.String(length=20), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('local_name', sa.String(length=200), nullable=True),
    sa.Column('country', sa.String(length=2), nullable=True),
    sa.Column('lei', sa.String(length=20), nullable=True),
    sa.Column('identifiers', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('unit', sa.String(length=10), nullable=False),
    sa.Column('accounting_standard', sa.String(length=10), nullable=False),
    sa.Column('fiscal_year_end_month', sa.SmallInteger(), nullable=True),
    sa.Column('warnings', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('refreshed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source IN ('sec', 'esef', 'companies_house', 'edinet')", name=op.f('ck_companies_source')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_companies')),
    sa.UniqueConstraint('source', 'source_id', name='uq_companies_source_source_id')
    )
    op.create_index('ix_companies_last_used_at', 'companies', ['last_used_at'], unique=False)
    op.create_index('ix_companies_lei', 'companies', ['lei'], unique=False)
    op.create_index('ix_companies_refreshed_at', 'companies', ['refreshed_at'], unique=False)
    op.create_table('edinet_reports',
    sa.Column('edinet_code', sa.String(length=6), nullable=False),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.Column('doc_id', sa.String(length=8), nullable=False),
    sa.Column('period_start', sa.Date(), nullable=True),
    sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('edinet_code', 'period_end', name=op.f('pk_edinet_reports'))
    )
    op.create_table('source_cursors',
    sa.Column('source', sa.String(length=20), nullable=False),
    sa.Column('through', sa.Date(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('source', name=op.f('pk_source_cursors'))
    )
    op.create_table('company_years',
    sa.Column('company_id', sa.BigInteger(), nullable=False),
    sa.Column('fiscal_year', sa.SmallInteger(), nullable=False),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.Column('figures', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('filing_url', sa.String(length=400), nullable=False),
    sa.Column('filing_form', sa.String(length=30), nullable=False),
    sa.Column('filing_id', sa.String(length=80), nullable=False),
    sa.Column('filed_on', sa.Date(), nullable=True),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_company_years_company_id_companies'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('company_id', 'fiscal_year', name=op.f('pk_company_years'))
    )


def downgrade() -> None:
    op.drop_table('company_years')
    op.drop_table('source_cursors')
    op.drop_table('edinet_reports')
    op.drop_index('ix_companies_refreshed_at', table_name='companies')
    op.drop_index('ix_companies_lei', table_name='companies')
    op.drop_index('ix_companies_last_used_at', table_name='companies')
    op.drop_table('companies')
