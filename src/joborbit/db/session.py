"""The database connection: SQLite, tuned so several processes can share it."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from joborbit.settings import get_settings


def create_sqlite_engine(url: str) -> Engine:
    """Create an SQLite connection with JobOrbit's settings switched on.

    Used for the real database and by the tests, so both follow the same rules
    (in particular, links between tables are enforced in both).
    """
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
        cursor.execute("PRAGMA foreign_keys=ON")  # enforce links between tables
        cursor.execute("PRAGMA busy_timeout=30000")  # wait up to 30 s if the file is busy
        cursor.close()

    return engine


@lru_cache
def get_engine() -> Engine:
    """Create the connection to the real database file once and reuse it."""
    db_file = get_settings().database_file
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_sqlite_engine(f"sqlite:///{db_file}")


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a session; save everything if all goes well, undo everything if not.

    Usage:
        with session_scope() as session:
            session.add(company)
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
