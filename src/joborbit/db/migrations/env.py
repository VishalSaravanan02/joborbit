"""Alembic's entry point: connects migrations to JobOrbit's models and database."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

from joborbit.db.models import Base
from joborbit.db.session import create_sqlite_engine
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


def _migration_engine() -> Engine:
    """A connection to the real database, set up for safely changing its tables.

    Two differences from the app's normal connection:

    - Links between tables (foreign keys) are not checked while a migration runs.
      SQLite can't change most tables in place, so a migration rebuilds them: copy the
      table, drop the old one, rename the copy. Dropping a table that other tables link
      to is refused while links are checked. Every link is checked at the end instead
      (see _check_links), so nothing broken can be saved.
    - Each migration is one real transaction: if any part fails, the whole migration is
      undone. Python's sqlite3 module doesn't put table changes inside a transaction by
      default, so we switch its own handling off and send BEGIN ourselves.
    """
    db_file = get_settings().database_file
    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(f"sqlite:///{db_file}")

    @event.listens_for(engine, "connect")
    def _prepare(dbapi_connection, _connection_record) -> None:
        # Runs after create_sqlite_engine's own setup, which switched foreign keys on.
        dbapi_connection.execute("PRAGMA foreign_keys=OFF")  # only works outside a transaction
        dbapi_connection.isolation_level = None  # stop sqlite3 managing transactions itself

    @event.listens_for(engine, "begin")
    def _begin(connection: Connection) -> None:
        connection.exec_driver_sql("BEGIN")

    return engine


def _check_links(connection: Connection) -> None:
    """Refuse to finish a migration that left any link between tables broken."""
    broken = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise RuntimeError(f"Migration left {len(broken)} broken link(s) between tables, e.g. {broken[:3]}")


def run_migrations_online() -> None:
    """Run the migration against the real database file from .env."""
    engine = _migration_engine()
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,  # SQLite can't alter tables directly; this works around it
                compare_type=True,  # also notice when a column's type changes
                # Our BEGIN makes table changes undoable, so let Alembic run every migration
                # and the final link check inside that one transaction: all or nothing.
                transactional_ddl=True,
            )
            with context.begin_transaction():
                context.run_migrations()
                _check_links(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
