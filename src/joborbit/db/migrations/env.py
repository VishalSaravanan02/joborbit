"""Alembic's entry point: connects migrations to JobOrbit's models and database."""

from logging.config import fileConfig

from alembic import context

from joborbit.db.models import Base
from joborbit.db.session import get_engine
from joborbit.settings import get_settings

config = context.config

# Use alembic.ini's logging settings, without switching off JobOrbit's own loggers.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# The tables Alembic compares against when generating migrations.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Write the migration as SQL text instead of running it (rarely needed)."""
    context.configure(
        url=f"sqlite:///{get_settings().database_file}",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migration against the real database file from .env."""
    with get_engine().connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite can't alter tables directly; this works around it
            compare_type=True,  # also notice when a column's type changes
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
