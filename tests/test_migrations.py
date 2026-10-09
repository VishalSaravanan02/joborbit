"""Tests for the database migrations (Alembic), run against throwaway database files.

They check that the migrations build exactly the tables in models.py, can be undone,
keep existing data when tables are rebuilt, and that a failed migration changes nothing.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

from joborbit.db.models import Base
from joborbit.db.session import create_sqlite_engine
from joborbit.settings import PROJECT_ROOT, get_settings

MIGRATIONS = PROJECT_ROOT / "src" / "joborbit" / "db" / "migrations"
BEFORE_USERS = "2d223caf33cf"  # the version before the users tables were added
BEFORE_MATCHES = "3d87c1822ecf"  # the version before matches and jobs.became_new_at were added


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """Point the migrations at a fresh database file instead of the real one."""
    path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    get_settings.cache_clear()  # settings are remembered; make them re-read DATABASE_PATH
    yield path
    get_settings.cache_clear()  # back to the real settings for the other tests


def alembic_config(script_location: Path = MIGRATIONS) -> Config:
    config = Config()  # no alembic.ini: leaves the test run's logging alone
    config.set_main_option("script_location", str(script_location))
    return config


def tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        return [name for (name,) in rows if name != "alembic_version"]


def version(path: Path) -> str | None:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        return row[0] if row else None


def add_linked_rows(path: Path) -> None:
    """One company with a job and a fetch run pointing at it (as in the real database).

    Works at any version: jobs.became_new_at is filled in only once the column exists.
    """
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO companies (name, slug, size_category, countries, active, baseline_done, "
            "consecutive_failures, created_at) VALUES ('Acme', 'acme', 'startup', '[\"GB\"]', 1, 1, 0, '2026-10-01')"
        )
        job_columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]
        became_new = ("became_new_at, ", "'2026-10-01', ") if "became_new_at" in job_columns else ("", "")
        connection.execute(
            f"INSERT INTO jobs (company_id, ats_type, external_id, url, title, country_codes, first_seen_at, "
            f"{became_new[0]}last_seen_at, missing_count, is_baseline, prefilter_status, analysis_status) "
            f"VALUES (1, 'greenhouse', '1', 'https://e.com/1', 'Data Analyst', '[]', '2026-10-01', "
            f"{became_new[1]}'2026-10-01', 0, 1, 'skipped', 'pending')"
        )
        connection.execute("INSERT INTO fetch_runs (company_id, started_at, status) VALUES (1, '2026-10-01', 'ok')")


def test_migrations_build_exactly_the_tables_in_models(db_file):
    command.upgrade(alembic_config(), "head")
    engine = create_sqlite_engine(f"sqlite:///{db_file}")
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    engine.dispose()
    assert differences == [], "models.py and the migrations disagree: a migration is missing"


def test_every_migration_can_be_undone_and_redone(db_file):
    config = alembic_config()
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    assert tables(db_file) == []
    command.upgrade(config, "head")
    assert "users" in tables(db_file)


def test_rebuilding_a_linked_table_keeps_its_rows_and_links(db_file):
    """Adding the users link rebuilds `companies`, which jobs and fetch runs point at."""
    config = alembic_config()
    command.upgrade(config, BEFORE_USERS)
    add_linked_rows(db_file)

    command.upgrade(config, "head")

    with sqlite3.connect(db_file) as connection:
        counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("companies", "jobs", "fetch_runs")
        ]
        assert counts == [1, 1, 1]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        links = {(row[2], row[3]) for row in connection.execute("PRAGMA foreign_key_list(companies)")}
        assert ("users", "added_by_user_id") in links


def copy_with_extra_migration(tmp_path: Path, upgrade_body: str) -> Config:
    """A copy of our migrations folder plus one extra migration with the given upgrade code."""
    folder = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, folder, ignore=shutil.ignore_patterns("__pycache__"))
    current_head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    (folder / "versions" / "zz_extra.py").write_text(
        "from alembic import op\n"
        "import sqlalchemy as sa\n\n"
        "revision = 'zzextra'\n"
        f"down_revision = '{current_head}'\n"
        "branch_labels = None\n"
        "depends_on = None\n\n\n"
        "def upgrade():\n"
        "    op.create_table('extra_table', sa.Column('id', sa.Integer(), primary_key=True))\n"
        "    with op.batch_alter_table('companies') as batch_op:\n"
        "        batch_op.add_column(sa.Column('extra_column', sa.Integer()))\n"
        f"{upgrade_body}\n\n\n"
        "def downgrade():\n"
        "    pass\n"
    )
    return alembic_config(folder)


@pytest.mark.parametrize(
    ("upgrade_body", "error"),
    [
        ("    raise RuntimeError('something went wrong halfway')", "halfway"),
        ("    op.execute('UPDATE jobs SET company_id = 999')", "broken link"),
    ],
)
def test_a_failed_migration_changes_nothing(db_file, tmp_path, upgrade_body, error):
    """Whether it crashes halfway or leaves a broken link, the database must be exactly as before."""
    command.upgrade(alembic_config(), "head")
    add_linked_rows(db_file)
    before_tables, before_version = tables(db_file), version(db_file)

    with pytest.raises(RuntimeError, match=error):
        command.upgrade(copy_with_extra_migration(tmp_path, upgrade_body), "head")

    assert tables(db_file) == before_tables and version(db_file) == before_version
    with sqlite3.connect(db_file) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(companies)")]
        assert "extra_column" not in columns
        assert connection.execute("SELECT company_id FROM jobs").fetchall() == [(1,)]


def test_existing_jobs_became_new_when_they_were_first_seen(db_file):
    """Adding jobs.became_new_at fills it in for the jobs already there, then makes it required."""
    config = alembic_config()
    command.upgrade(config, BEFORE_MATCHES)
    add_linked_rows(db_file)
    with sqlite3.connect(db_file) as connection:
        connection.execute("UPDATE jobs SET first_seen_at = '2026-10-03 09:20:00'")

    command.upgrade(config, "head")

    with sqlite3.connect(db_file) as connection:
        assert connection.execute("SELECT became_new_at FROM jobs").fetchall() == [("2026-10-03 09:20:00",)]
        required = {row[1]: row[3] for row in connection.execute("PRAGMA table_info(jobs)")}
        assert required["became_new_at"] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
