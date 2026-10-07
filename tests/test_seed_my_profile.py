"""End-to-end tests for scripts/seed_my_profile.py: the real script, run on a throwaway database.

The example file is used as the input, so these tests also prove that
my_profile.example.yaml stays valid and that its companies are in the real company list.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sqlalchemy.orm import Session

from joborbit.companies import read_company_csv, upsert_companies
from joborbit.db.models import Base
from joborbit.db.session import create_sqlite_engine
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "seed_my_profile.py"
EXAMPLE = PROJECT_ROOT / "my_profile.example.yaml"
TELEGRAM_ID = "123456789"


@pytest.fixture
def db_file(tmp_path) -> Path:
    """A throwaway database holding the real company list."""
    path = tmp_path / "test.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        upsert_companies(session, read_company_csv(PROJECT_ROOT / "data" / "companies_seed.csv"))
        session.commit()
    engine.dispose()
    return path


def run(db_file: Path, profile_file: Path) -> subprocess.CompletedProcess:
    """Run the script as you would, but pointed at the throwaway database."""
    env = {**os.environ, "DATABASE_PATH": str(db_file), "ADMIN_TELEGRAM_ID": TELEGRAM_ID}
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(profile_file)], env=env, capture_output=True, text=True, check=False
    )


def query(db_file: Path, sql: str) -> list[tuple]:
    with sqlite3.connect(db_file) as connection:
        return connection.execute(sql).fetchall()


def edited_example(tmp_path: Path, change) -> Path:
    """A copy of the example file with `change` applied to its contents."""
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_example_file_creates_an_admin_user_with_its_profile(db_file):
    result = run(db_file, EXAMPLE)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Created Alex (admin)")
    assert query(db_file, "SELECT telegram_id, display_name, is_admin FROM users") == [(int(TELEGRAM_ID), "Alex", 1)]
    assert query(db_file, "SELECT countries, languages FROM user_profiles") == [('["GB"]', '["en", "es"]')]
    favourites = query(
        db_file, "SELECT c.name, p.never_miss FROM user_company_prefs p JOIN companies c ON c.id = p.company_id"
    )
    assert sorted(favourites) == [("Cloudflare", 0), ("Monzo", 1)]


def test_running_again_updates_the_same_user(db_file, tmp_path):
    run(db_file, EXAMPLE)

    def rename_and_drop_favourites(data):
        data["display_name"] = "Vishal"
        data["profile"]["favourite_companies"] = []

    result = run(db_file, edited_example(tmp_path, rename_and_drop_favourites))

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Updated Vishal (admin)")
    assert query(db_file, "SELECT display_name FROM users") == [("Vishal",)]
    assert query(db_file, "SELECT COUNT(*) FROM user_company_prefs") == [(0,)]


def test_mistakes_are_explained_and_nothing_is_saved(db_file, tmp_path):
    def add_mistakes(data):
        data["profile"]["preferred_industries"] = ["fintek"]
        data["profile"]["langauges"] = ["en"]

    result = run(db_file, edited_example(tmp_path, add_mistakes))

    assert result.returncode == 1
    assert "unknown industry 'fintek'" in result.stderr
    assert "profile > langauges: not a field JobOrbit knows (a typo?)" in result.stderr
    assert query(db_file, "SELECT COUNT(*) FROM users") == [(0,)]


def test_an_unknown_company_is_named_and_nothing_is_saved(db_file, tmp_path):
    def misspell(data):
        data["profile"]["favourite_companies"] = [{"name": "Cloudflair"}]

    result = run(db_file, edited_example(tmp_path, misspell))

    assert result.returncode == 1
    assert "Cloudflair" in result.stderr
    assert query(db_file, "SELECT COUNT(*) FROM users") == [(0,)]


def test_a_missing_file_says_how_to_start_one(db_file, tmp_path):
    result = run(db_file, tmp_path / "my_profile.yaml")
    assert result.returncode == 1
    assert "Copy my_profile.example.yaml" in result.stderr
