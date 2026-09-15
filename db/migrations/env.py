"""Alembic environment. Migrations always run through ``db.migrate``, which
hands over an open connection (direct, not pooled, with the migration lock)."""
from alembic import context
from sqlalchemy.types import TypeDecorator

from db.models import Base


def render_item(type_, obj, autogen_context):
    """Write our column types (UTCDateTime, MoneyAmount...) as their plain SQL
    type, so migration files don't import the models."""
    if type_ == "type" and isinstance(obj, TypeDecorator):
        return "sa." + repr(obj.impl)
    return False


connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("run migrations with `python -m db.migrate`, not the alembic command")

context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    compare_type=True,
    compare_server_default=True,
    render_item=render_item,
)
with context.begin_transaction():
    context.run_migrations()
