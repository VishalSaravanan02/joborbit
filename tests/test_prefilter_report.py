"""End-to-end tests for scripts/prefilter_report.py: the real script, on a throwaway database."""

import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, Job, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "prefilter_report.py"
JOBS = [
    ("Graduate Data Scientist", "London, UK"),
    ("Data Analyst", "Remote"),
    ("Senior Data Engineer", "London"),
    ("Lead Data Scientist", "London"),
    ("Software Engineer", "London"),
    ("Data Scientist", "Berlin, Germany"),
]


@pytest.fixture
def db_file(tmp_path) -> Path:
    """A throwaway database with one user (data roles), a few baseline jobs and one closed job."""
    path = tmp_path / "test.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Company(name="Acme", slug="acme", size_category="startup"))
        user = User(telegram_id=1, display_name="Vishal")
        user.profile = UserProfile(roles=["data_scientist", "data_analyst", "data_engineer"], countries=["GB"])
        session.add(user)
        for number, (title, location) in enumerate(JOBS):
            session.add(
                Job(
                    company_id=1, ats_type="greenhouse", external_id=str(number), url="https://e.com",
                    title=title, location_raw=location, is_baseline=True, prefilter_status="skipped",
                )
            )
        session.add(
            Job(
                company_id=1, ats_type="greenhouse", external_id="closed", url="https://e.com",
                title="Closed Data Scientist Job", location_raw="London", closed_at=datetime(2026, 10, 1),
            )
        )
        session.commit()
    engine.dispose()
    return path


def run(db_file: Path, *arguments: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_PATH": str(db_file)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], env=env, capture_output=True, text=True, check=False
    )


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_report_counts_and_shows_each_group(db_file):
    result = run(db_file)

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Checked 6 open jobs" in out and "3 default roles, 0 custom; internships off" in out
    assert "Would pass:       2  (33.3%)" in out
    assert "senior title               2" in out and "country                    1" in out
    assert "WOULD PASS (2 of 2)" in out and "Graduate Data Scientist" in out
    assert "REJECTED: no matching role (1 of 1)" in out and "Software Engineer" in out
    assert "Closed Data Scientist Job" not in out


def test_closed_jobs_are_included_only_when_asked(db_file):
    result = run(db_file, "--include-closed")
    assert "Checked 7 jobs" in result.stdout and "Would pass:       3" in result.stdout
    assert "Closed Data Scientist Job" in result.stdout


def test_the_report_changes_nothing(db_file):
    before = fingerprint(db_file)
    assert run(db_file).returncode == 0
    assert run(db_file, "--reason", "senior").returncode == 0
    assert fingerprint(db_file) == before


def test_listing_one_reason(db_file):
    result = run(db_file, "--reason", "senior title: lead")
    assert 'REJECTED, reason starting "senior title: lead": 1' in result.stdout
    assert "Lead Data Scientist" in result.stdout and "Senior Data Engineer" not in result.stdout


def test_samples_are_limited_but_the_same_every_run(db_file):
    first, second = run(db_file, "--samples", "1"), run(db_file, "--samples", "1")
    assert "WOULD PASS (1 of 2)" in first.stdout
    assert first.stdout == second.stdout
