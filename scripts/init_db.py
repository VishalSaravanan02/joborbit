"""Create the database, or update it to the latest version of the tables.

Safe to run any time: it only applies changes that haven't been applied yet.

Usage:
    python scripts/init_db.py
"""

from alembic import command
from alembic.config import Config

from joborbit.settings import PROJECT_ROOT, get_settings


def main() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    print(f"Database is up to date: {get_settings().database_file}")


if __name__ == "__main__":
    main()
