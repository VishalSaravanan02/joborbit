"""Tests for what scripts/run_cycle.py prints. Nothing here fetches from a real job site or asks a real LLM.

The script's parts are loaded straight from the file and run on a throwaway database.
"""

import importlib.util
import subprocess
import sys
from collections import Counter

import pytest
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, Job, JobAnalysis
from joborbit.db.session import create_sqlite_engine
from joborbit.pipeline.analyse import AnalysisRun
from joborbit.pipeline.cycle import Processing
from joborbit.pipeline.prefilter import PrefilterRun
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "run_cycle.py"
spec = importlib.util.spec_from_file_location("run_cycle", SCRIPT)
run_cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_cycle)


def analysis(**fields) -> JobAnalysis:
    data = {
        "countries": ["GB"], "seniority": "graduate", "experience_years": None, "experience_mandatory": False,
        "role_families": ["data_analyst"], "matched_custom_roles": [],
    }
    return JobAnalysis(**{**data, **fields})


@pytest.fixture
def session(tmp_path):
    """Acme with four jobs: 1 analysed, 2 rejected, 3 failed, 4 waiting."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Company(name="Acme", slug="acme", size_category="startup"))
        jobs = [
            ("Graduate Data Analyst", "passed", None, "done"),
            ("Senior Data Analyst", "rejected", "senior title: senior", "pending"),
            ("Data Scientist", "passed", None, "failed"),
            ("Junior Data Analyst", "passed", None, "pending"),
        ]
        for number, (title, prefilter, reason, status) in enumerate(jobs, start=1):
            session.add(
                Job(
                    company_id=1, ats_type="greenhouse", external_id=str(number), url=f"https://e.com/{number}",
                    title=title, location_raw="London", prefilter_status=prefilter, prefilter_reason=reason,
                    analysis_status=status,
                )
            )
        session.flush()
        session.add(
            JobAnalysis(
                job_id=1, countries=["GB"], cities=["London"], work_mode="hybrid", seniority="graduate",
                is_graduate_scheme=False, experience_years=None, experience_mandatory=False,
                required_languages=[], role_families=["data_analyst"], matched_custom_roles=["Insights Analyst"],
                skills=["SQL"], min_degree=None, deadline=None, summary="A graduate job.",
                model="gpt-6-luna", prompt_version="2", input_tokens=3000, output_tokens=500,
            )
        )
        session.commit()
        yield session
    engine.dispose()


def processing(analysis_run: AnalysisRun | None, waiting: int = 1, retried: int = 0) -> Processing:
    prefilter = PrefilterRun(passed_job_ids=[1, 3, 4], rejected=Counter({"senior title": 1}))
    return Processing(prefilter, retried, analysis_run, waiting)


# --- The parts of a line ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("years", "mandatory", "expected"),
    [
        (None, False, "no experience asked"),
        (None, True, "experience required"),
        (1, False, "1+ year preferred"),
        (3, True, "3+ years required"),
        (0, False, "0+ years preferred"),
    ],
)
def test_the_experience_is_said_in_a_few_words(years, mandatory, expected):
    assert run_cycle.experience(analysis(experience_years=years, experience_mandatory=mandatory)) == expected


def test_an_analysis_is_described_on_one_line():
    described = run_cycle.describe_analysis(analysis(matched_custom_roles=["Insights Analyst"]))
    assert described == "GB | graduate | no experience asked | data_analyst, Insights Analyst (custom)"


def test_missing_values_are_said_plainly():
    described = run_cycle.describe_analysis(analysis(countries=[], seniority=None, role_families=[]))
    assert described == "no country found | seniority unclear | no experience asked | no role"


# --- What is printed ----------------------------------------------------------------------------------


def test_each_new_job_shows_the_prefilters_verdict_when_asked(session, capsys):
    run_cycle.print_new_jobs(session, [1, 2], show_rejected=True)
    out = capsys.readouterr().out
    assert "New jobs:\n" in out
    assert "Acme: Graduate Data Analyst (London)  [passed]\n    https://e.com/1" in out
    assert "Acme: Senior Data Analyst (London)  [rejected: senior title: senior]" in out
    assert "not listed" not in out


def test_by_default_only_new_jobs_that_passed_are_listed(session, capsys):
    run_cycle.print_new_jobs(session, [1, 2])
    out = capsys.readouterr().out
    assert "New jobs that passed the pre-filter:\n  Acme: Graduate Data Analyst (London)  [passed]" in out
    assert "Senior Data Analyst" not in out
    assert "1 other new job not listed (--show-rejected lists them)." in out


def test_when_no_new_job_passed_only_the_count_is_shown(session, capsys):
    run_cycle.print_new_jobs(session, [2])
    out = capsys.readouterr().out
    assert "New jobs" not in out
    assert "1 other new job not listed" in out


def test_no_new_jobs_prints_nothing(session, capsys):
    run_cycle.print_new_jobs(session, [])
    assert capsys.readouterr().out == ""


def test_the_analysed_and_failed_jobs_are_listed(session, capsys):
    run = AnalysisRun(done=[1], failed=[3], cost_usd=0.0004)
    run_cycle.print_processing(session, processing(run))
    out = capsys.readouterr().out
    assert "Pre-filter: 4 checked, 3 passed, 1 rejected (1 senior title)" in out
    assert "Analysis: 1 done, 1 failed, 1 waiting, about $0.0004" in out
    assert "Acme: Graduate Data Analyst\n    GB | graduate | no experience asked | " in out
    assert "data_analyst, Insights Analyst (custom)" in out
    assert "No usable analysis" in out and "  Acme: Data Scientist" in out
    assert "stopped" not in out


def test_a_stopped_analysis_says_why_and_that_the_jobs_wait(session, capsys):
    run_cycle.print_processing(session, processing(AnalysisRun(stopped="Could not reach OpenAI: Timed out.")))
    out = capsys.readouterr().out
    assert "Analysis stopped: Could not reach OpenAI: Timed out. The waiting jobs are kept for a later run." in out
    assert "Analysed jobs" not in out and "No usable analysis" not in out


def test_analysis_switched_off_says_how_many_wait(session, capsys):
    run_cycle.print_processing(session, processing(None, waiting=2))
    out = capsys.readouterr().out
    assert "Analysis switched off; 2 jobs waiting for it." in out
    assert "Analysis:" not in out


def test_retried_jobs_are_counted(session, capsys):
    run_cycle.print_processing(session, processing(AnalysisRun(), retried=1))
    assert "1 failed job sent back for another try." in capsys.readouterr().out


def test_a_limit_below_one_is_refused_before_anything_is_fetched():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--limit", "0"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2 and "--limit must be at least 1" in result.stderr
