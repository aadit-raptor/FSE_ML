"""Database layer (PLAN.md 1.3): Postgres on Neon, SQLAlchemy models, Alembic
migrations.

The API works without a database: nothing here connects until a request
needs it, and ``DATABASE_URL`` unset means "no database" (local runs, CI's
model tests). See CLAUDE.md "Adding a table".
"""
from db.engine import DatabaseUnavailable, connect, database_url, is_configured, transaction

__all__ = ["DatabaseUnavailable", "connect", "database_url", "is_configured", "transaction"]
