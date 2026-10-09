"""Tests for what scripts/dry_run.py prints. Nothing here reads a real database or asks a real LLM."""

import importlib.util
import subprocess
import sys
from collections import Counter
from datetime import datetime

import pytest

from joborbit.pipeline.dry_run import DryResult, DryRun
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "dry_run.py"
spec = importlib.util.spec_from_file_location("dry_run_script", SCRIPT)
dry_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry_run)

SINCE, UNTIL = datetime(2026, 9, 25, 9, 0), datetime(2026, 10, 9, 9, 0)
POINTS = {"role_fit": 30, "entry_fit": 6.25, "skills": 13.33, "country": 15, "transfer": 0}


def result(title: str, tier: str | None, score: int | None = None, day: int = 8, user: str = "Vishal", **extra):
    return DryResult(
        user_name=user, job_id=1, company="Acme", title=title, url=f"https://e.com/{title.replace(' ', '-')}",
        posted_at=datetime(2026, 10, day, 12, 0), tier=tier, score=score,
        points=POINTS if tier else {}, **extra,
    )


def make_run(results=(), **counts) -> DryRun:
    rejected = Counter({"country": 90, "senior title": 20})
    run = DryRun(since=SINCE, until=UNTIL, jobs=120, no_date=4, passed=10, rejected=rejected)
    run.from_database, run.from_file, run.asked, run.cost_usd = 2, 5, 3, 0.0012
    for name, value in counts.items():
        setattr(run, name, value)
    run.results = list(results)
    return run


def report(capsys, run: DryRun, **flags) -> str:
    dry_run.print_report(run, **flags)
    return capsys.readouterr().out


# --- The summary ---------------------------------------------------------------------------------------


def test_the_summary_says_the_window_and_what_each_step_did(capsys):
    out = report(capsys, make_run())
    assert out.startswith("Dry run over the last 14 days (25 Sep to 09 Oct). Nothing is saved or sent.\n")
    assert "Jobs: 120 open jobs posted in the window (4 more have no posting date and are left out)." in out
    assert "Pre-filter: 10 passed, 110 rejected (90 country, 20 senior title)" in out
    assert "Analysis: 2 from the database, 5 saved, 3 asked (about $0.0012), 0 failed, 0 not analysed" in out
    assert "stopped" not in out and "not analysed are left out" not in out
    assert "No analysed jobs to match." in out


def test_jobs_not_analysed_and_a_stop_are_explained(capsys):
    out = report(capsys, make_run(not_analysed=4, stopped="Could not reach OpenAI."))
    assert "Analysis stopped: Could not reach OpenAI.\n" in out
    assert "Jobs not analysed are left out below: run again (or with a higher --limit) to include them." in out


# --- Each user ----------------------------------------------------------------------------------------------


def test_each_user_gets_counts_a_rate_per_day_and_the_busiest_digest_day(capsys):
    results = [
        result("Graduate Analyst", "instant", 82),
        result("Data Analyst", "digest", 61, day=8),
        result("BI Analyst", "digest", 58, day=8),
        result("Junior Analyst", "digest", 55, day=6),
        result("Data Analyst II", "silent", 40),
        result("Senior Analyst", None, dropped="level: senior"),
        result("Analista de Datos", None, dropped="language: fr required"),
        result("Lead Analyst", None, dropped="level: 3+ years required"),
    ]
    out = report(capsys, make_run(results))
    assert "Vishal: 1 instant, 3 digest, 1 silent, 3 dropped (2 level, 1 language)" in out
    assert "  Per day: 0.1 instant, 0.2 digest (busiest day: 2 digest jobs)" in out


def test_instant_then_digest_jobs_are_listed_best_first_with_their_points(capsys):
    results = [  # alphabetical order would put BI Analyst first: the score must decide
        result("BI Analyst", "digest", 58),
        result("Graduate Analyst", "instant", 82, reasons=("Role: Data Analyst", "Graduate role"),
               notes=("Originally posted Jan 2022",)),
        result("Data Analyst", "digest", 61),
        result("Data Analyst II", "silent", 40),
    ]
    out = report(capsys, make_run(results))
    tags = ["[instant 82]", "[digest 61]", "[digest 58]"]
    assert [out.index(tag) for tag in tags] == sorted(out.index(tag) for tag in tags)
    assert (
        "\n  Instant:\n  [instant 82] Acme: Graduate Analyst (posted 08 Oct)\n"
        "    role 30 | level 6.25 | skills 13.33 | country 15 | transfer 0\n"
        "    Role: Data Analyst | Graduate role\n"
        "    ! Originally posted Jan 2022\n"
        "    https://e.com/Graduate-Analyst\n"
    ) in out
    assert "Silent" not in out and "Data Analyst II" not in out


def test_silent_and_dropped_jobs_are_listed_only_when_asked(capsys):
    results = [result("Data Analyst II", "silent", 40), result("Senior Analyst", None, dropped="level: senior")]
    quiet = report(capsys, make_run(results))
    assert "Data Analyst II" not in quiet and "Senior Analyst" not in quiet

    out = report(capsys, make_run(results), show_silent=True, show_dropped=True)
    assert "\n  Silent:\n  [silent 40] Acme: Data Analyst II (posted 08 Oct)\n" in out
    assert (
        "\n  Dropped by the hard filters:\n  [dropped: level: senior] Acme: Senior Analyst (posted 08 Oct)\n"
        "    https://e.com/Senior-Analyst\n"
    ) in out


def test_bonuses_are_named_in_the_points_line():
    points = {**POINTS, "favourite_company": 15, "preferred_industry": 5}
    assert dry_run.points_line(points).endswith("transfer 0 | favourite 15 | industry 5")


def test_each_user_has_their_own_section(capsys):
    results = [
        result("Data Analyst", "digest", 61),
        result("Data Analyst", None, user="Ana", dropped="role: not one of yours"),
    ]
    out = report(capsys, make_run(results))
    vishal = out.index("\nVishal: 0 instant, 1 digest")
    assert vishal < out.index("\nAna: 0 instant, 0 digest, 0 silent, 1 dropped (1 role)")


# --- The whole script ---------------------------------------------------------------------------------------


def test_main_runs_the_dry_run_with_the_options_given(monkeypatch, capsys):
    seen = {}

    def fake_run(session, now, days, limit, client_factory):
        seen.update(days=days, limit=limit, factory=client_factory)
        return make_run()

    def fake_write(run, path):
        seen.update(sheet=path)
        return 18, 5

    monkeypatch.setattr(dry_run, "run_dry_run", fake_run)
    monkeypatch.setattr(dry_run, "write_review", fake_write)  # never touch the real sheet: it holds answers
    monkeypatch.setattr(dry_run, "setup_logging", lambda *_: None)
    monkeypatch.setattr(dry_run, "get_engine", lambda: None)  # never open the real database in a test
    monkeypatch.setattr(sys, "argv", ["dry_run.py", "--days", "7", "--limit", "20"])
    dry_run.main()
    assert seen == {"days": 7, "limit": 20, "factory": dry_run.make_client, "sheet": dry_run.DEFAULT_REVIEW_FILE}
    out = capsys.readouterr().out
    assert "Dry run over the last 14 days" in out
    assert (
        "\nReview sheet: var/dry_run/review.csv (18 rows, 5 answers kept from the last sheet). "
        "Fill in the want column: yes, maybe or no.\n"
    ) in out

    monkeypatch.setattr(sys, "argv", ["dry_run.py", "--no-analysis"])
    dry_run.main()
    assert seen["days"] == 14 and seen["limit"] == 100 and seen["factory"] is None


@pytest.mark.parametrize(
    ("option", "message"), [("--days=0", "--days must be at least 1"), ("--limit=-1", "can't be negative")]
)
def test_bad_options_are_refused_before_anything_runs(option, message):
    done = subprocess.run([sys.executable, str(SCRIPT), option], capture_output=True, text=True, check=False)
    assert done.returncode == 2 and message in done.stderr
