"""Tests for the dry run (joborbit/pipeline/dry_run.py). Nothing here contacts a real LLM."""

import codecs
import csv
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from joborbit.config import load_app_settings
from joborbit.db.models import Base, Company, Job, JobAnalysis, LlmUsage, Match, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.llm.client import LlmClient, LlmUnavailable, Provider, Reply
from joborbit.llm.prompts import PROMPT_VERSION
from joborbit.pipeline.dry_run import (
    REVIEW_COLUMNS,
    AnswerStore,
    DryResult,
    DryRun,
    Unmatched,
    experience_words,
    review_rows,
    run_dry_run,
    write_review,
)

NOW = datetime(2026, 10, 9, 9, 0)
ANSWER = {
    "countries": ["GB"], "cities": ["London"], "work_mode": "hybrid", "seniority": "graduate",
    "is_graduate_scheme": False, "experience_years": None, "experience_mandatory": False,
    "required_languages": [], "role_families": ["data_analyst"], "matched_custom_roles": [],
    "skills": ["SQL", "Python"], "min_degree": None, "deadline": None,
    "summary": "A graduate data analyst job in London.",
}
GOOD = json.dumps(ANSWER)


class FakeProvider(Provider):
    """Answers with the prepared replies in order and keeps every request."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[str] = []

    def send(self, instructions, messages, schema_name, schema):
        self.requests.append(messages[0].content)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply("done", reply, "gpt-6-luna-2026-05-18", 3000, 2000, 500)


@pytest.fixture
def sessions(tmp_path):
    """Acme, a UK fintech, and Vishal, who wants data analyst jobs in the UK and knows SQL."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        session.add(Company(name="Acme", slug="acme", size_category="startup", industry="fintech", countries=["GB"]))
        add_user(session, "Vishal", 1)
    return factory


@pytest.fixture
def store(tmp_path):
    return AnswerStore(tmp_path / "dry_run" / "answers.json")


def add_user(session, name: str, telegram_id: int, **profile) -> None:
    user = User(telegram_id=telegram_id, display_name=name)
    data = {"roles": ["data_analyst"], "countries": ["GB"], "languages": ["en"], "skills": ["SQL"]}
    user.profile = UserProfile(**{**data, **profile})
    session.add(user)


def add_job(sessions, title: str = "Graduate Data Analyst", days_ago: float = 1, **fields) -> int:
    """A baseline job at Acme in London, posted `days_ago` days before NOW."""
    data = {
        "company_id": 1, "ats_type": "greenhouse", "external_id": f"{title}-{days_ago}", "url": "https://e.com/1",
        "title": title, "location_raw": "London, UK", "description_text": "Join our data team.",
        "posted_at": NOW - timedelta(days=days_ago), "first_seen_at": NOW - timedelta(days=20),
        "became_new_at": NOW - timedelta(days=20), "is_baseline": True, "prefilter_status": "skipped",
        "analysis_status": "pending", "content_hash": "hash-1",
    }
    with sessions() as session, session.begin():
        job = Job(**{**data, **fields})
        session.add(job)
        session.flush()
        return job.id


def add_analysis(sessions, job_id: int, prompt_version: str = PROMPT_VERSION, **fields) -> None:
    with sessions() as session, session.begin():
        made_by = {"model": "gpt-6-luna", "prompt_version": prompt_version, "input_tokens": 1, "output_tokens": 1}
        session.add(JobAnalysis(job_id=job_id, **made_by, **{**ANSWER, **fields}))


def client_with(sessions, *replies) -> LlmClient:
    return LlmClient(FakeProvider(*replies), load_app_settings().llm, sessions, now=lambda: NOW)


def dry_run(sessions, store, client=None, **options):
    with sessions() as session:
        return run_dry_run(session, NOW, client=client, store=store, **options)


# --- Which jobs ------------------------------------------------------------------------------------


def test_open_jobs_posted_in_the_window_are_taken_baseline_included(sessions, store):
    inside = add_job(sessions, "Graduate Data Analyst", days_ago=13)
    add_job(sessions, "Data Analyst", days_ago=15)  # before the window
    add_job(sessions, "Junior Data Analyst", days_ago=2, closed_at=NOW)  # closed
    add_job(sessions, "Data Analyst II", days_ago=1, posted_at=None)  # no date: counted, left out
    client = client_with(sessions, GOOD)

    run = dry_run(sessions, store, client, days=14)

    assert (run.jobs, run.no_date, run.passed, run.asked) == (1, 1, 1, 1)
    assert [result.job_id for result in run.results] == [inside]
    assert run.days == 14 and run.until == NOW


def test_jobs_rejected_by_the_prefilter_are_counted_and_never_asked_about(sessions, store):
    add_job(sessions, "Senior Data Analyst")
    add_job(sessions, "Marketing Manager", days_ago=2)
    client = client_with(sessions)
    run = dry_run(sessions, store, client)
    assert run.passed == 0 and run.rejected == {"senior title": 2}
    assert client.provider.requests == [] and run.results == []


def test_newest_jobs_are_asked_about_first(sessions, store):
    add_job(sessions, "Data Analyst", days_ago=5)
    newest = add_job(sessions, "Graduate Data Analyst", days_ago=1)
    client = client_with(sessions, GOOD)
    run = dry_run(sessions, store, client, limit=1)
    assert [result.job_id for result in run.results] == [newest]
    assert "Title: Graduate Data Analyst" in client.provider.requests[0]
    assert (run.asked, run.not_analysed) == (1, 1)


# --- Analyses ---------------------------------------------------------------------------------------


def test_a_current_analysis_in_the_database_is_reused(sessions, store):
    job_id = add_job(sessions)
    add_analysis(sessions, job_id)
    client = client_with(sessions)
    run = dry_run(sessions, store, client)
    assert (run.from_database, run.asked) == (1, 0) and client.provider.requests == []
    assert run.results[0].tier == "instant"


def test_an_analysis_made_with_an_older_prompt_is_asked_again(sessions, store):
    job_id = add_job(sessions)
    add_analysis(sessions, job_id, prompt_version="1")
    run = dry_run(sessions, store, client_with(sessions, GOOD))
    assert (run.from_database, run.asked) == (0, 1)


def test_a_new_answer_is_saved_and_reused_by_the_next_run(sessions, store, tmp_path):
    add_job(sessions)
    first = dry_run(sessions, store, client_with(sessions, GOOD))
    assert first.asked == 1 and round(first.cost_usd, 6) > 0

    again = AnswerStore(tmp_path / "dry_run" / "answers.json")  # read back from the file
    second_client = client_with(sessions)
    second = dry_run(sessions, again, second_client)

    assert (second.from_file, second.asked) == (1, 0) and second_client.provider.requests == []
    assert second.results == first.results
    assert not (tmp_path / "dry_run" / "answers.tmp").exists()


@pytest.mark.parametrize("change", ["edited ad", "new custom role"])
def test_a_saved_answer_is_not_reused_once_the_job_or_the_custom_roles_change(sessions, store, change):
    job_id = add_job(sessions)
    dry_run(sessions, store, client_with(sessions, GOOD))
    with sessions() as session, session.begin():
        if change == "edited ad":
            session.get_one(Job, job_id).content_hash = "hash-2"
        else:
            session.get_one(User, 1).profile.custom_roles = ["Insights Analyst"]

    run = dry_run(sessions, store, client_with(sessions, GOOD))
    assert (run.from_file, run.asked) == (0, 1)


def test_without_a_client_only_saved_answers_are_used(sessions, store):
    add_job(sessions)
    run = dry_run(sessions, store)
    assert (run.asked, run.not_analysed, run.results) == (0, 1, [])


def test_the_client_is_made_only_when_an_answer_is_needed(sessions, store):
    job_id = add_job(sessions)
    add_analysis(sessions, job_id)
    made = []

    def factory():
        made.append(1)
        return client_with(sessions)

    with sessions() as session:
        run_dry_run(session, NOW, client_factory=factory, store=store)
    assert made == []


def test_a_client_that_cant_be_made_stops_the_analysis_but_not_the_run(sessions, store):
    add_job(sessions)
    add_job(sessions, "Data Analyst", days_ago=2)

    def no_key():
        raise LlmUnavailable("OPENAI_API_KEY is not set in .env")

    with sessions() as session:
        run = run_dry_run(session, NOW, client_factory=no_key, store=store)
    assert run.stopped == "OPENAI_API_KEY is not set in .env"
    assert (run.passed, run.not_analysed, run.asked) == (2, 2, 0)


def test_when_the_llm_stops_the_remaining_jobs_are_not_asked_about(sessions, store):
    add_job(sessions, days_ago=1)
    add_job(sessions, "Data Analyst", days_ago=2)
    client = client_with(sessions, LlmUnavailable("Could not reach OpenAI"))
    run = dry_run(sessions, store, client)
    assert run.stopped == "Could not reach OpenAI"
    assert (run.asked, run.not_analysed) == (0, 2) and len(client.provider.requests) == 1


def test_an_unusable_answer_is_counted_as_failed(sessions, store):
    add_job(sessions)
    run = dry_run(sessions, store, client_with(sessions, "not json", "still not json"))
    assert (run.failed, run.asked, run.not_analysed, run.results) == (1, 0, 1, [])
    assert len(store) == 0


def test_failed_jobs_count_towards_the_limit(sessions, store):
    add_job(sessions, days_ago=1)
    add_job(sessions, "Data Analyst", days_ago=2)
    client = client_with(sessions, "not json", "still not json")
    run = dry_run(sessions, store, client, limit=1)
    assert (run.failed, run.asked, run.not_analysed) == (1, 0, 2)


# --- Matching -------------------------------------------------------------------------------------------


def test_every_active_user_gets_a_tier_or_the_reason_it_was_dropped(sessions, store):
    with sessions() as session, session.begin():
        add_user(session, "Ana", 2, roles=["data_scientist"])
    job_id = add_job(sessions)

    run = dry_run(sessions, store, client_with(sessions, GOOD))

    vishal, ana = run.results
    assert (vishal.user_name, vishal.tier, vishal.score, vishal.dropped) == ("Vishal", "instant", 80, None)
    assert vishal.points == {"role_fit": 30, "entry_fit": 25, "skills": 10, "country": 15, "transfer": 0}
    assert vishal.reasons == ("Role: Data Analyst", "Graduate role", "London") and vishal.notes == ()
    assert (vishal.job_id, vishal.company, vishal.url) == (job_id, "Acme", "https://e.com/1")
    assert vishal.posted_at == NOW - timedelta(days=1)
    assert (ana.user_name, ana.tier, ana.score, ana.dropped) == ("Ana", None, None, "role: not one of yours")


# --- Nothing changes --------------------------------------------------------------------------------------


def test_nothing_in_the_database_changes_except_the_llm_usage(sessions, store):
    job_id = add_job(sessions)
    dry_run(sessions, store, client_with(sessions, GOOD))

    with sessions() as session:
        job = session.get_one(Job, job_id)
        assert (job.prefilter_status, job.analysis_status, job.matched_at) == ("skipped", "pending", None)
        assert session.scalar(select(func.count()).select_from(JobAnalysis)) == 0
        assert session.scalar(select(func.count()).select_from(Match)) == 0
        assert session.scalar(select(LlmUsage.calls)) == 1  # the money spent is still recorded


def test_each_users_alert_style_decides_the_tier(sessions, store):
    with sessions() as session, session.begin():
        add_user(session, "Ben", 2, alert_style="fewer")
    add_job(sessions)
    entry = json.dumps({**ANSWER, "seniority": "entry"})  # 30 + 22.5 + 10 + 15 = 77.5 -> 78
    run = dry_run(sessions, store, client_with(sessions, entry))
    # Vishal (balanced) gets instant alerts from 75; Ben (fewer) only from 80.
    assert [(result.user_name, result.score, result.tier) for result in run.results] == [
        ("Vishal", 78, "instant"), ("Ben", 78, "digest"),
    ]


def test_the_dry_run_leaves_its_session_untouched_and_never_commits(sessions, store, monkeypatch):
    """Even objects it builds for matching are never added to the session, and nothing is ever saved."""
    job_id = add_job(sessions)
    add_job(sessions, "Data Analyst", days_ago=2)
    add_analysis(sessions, job_id)
    with sessions() as session:
        state = {}

        def refuse_commit():
            raise AssertionError("the dry run must never commit")

        def record_rollback():
            state.update(new=list(session.new), dirty=list(session.dirty))

        monkeypatch.setattr(session, "commit", refuse_commit)
        monkeypatch.setattr(session, "rollback", record_rollback)
        run_dry_run(session, NOW, client=client_with(sessions, GOOD), store=store)

    assert state == {"new": [], "dirty": []}


def test_a_mid_level_job_requiring_experience_is_kept_but_silent(sessions, store):
    add_job(sessions)
    mid = json.dumps({**ANSWER, "seniority": "mid", "experience_mandatory": True})  # 30 + 6.25 + 10 + 15 = 61
    [result] = dry_run(sessions, store, client_with(sessions, mid)).results
    assert (result.score, result.tier) == (61, "silent")  # a digest score, held back by the cap
    assert result.notes == ("Experience required (no number given)",)


# --- The review sheet -----------------------------------------------------------------------------------


def test_only_jobs_with_no_matching_role_are_kept_from_the_prefilter(sessions, store):
    software = add_job(sessions, "Software Engineer", location_raw="London")
    add_job(sessions, "Data Analyst", days_ago=2, location_raw="New York")  # country
    add_job(sessions, "Senior Data Analyst", days_ago=3)  # senior title
    run = dry_run(sessions, store)
    assert run.rejected == {"no matching role": 1, "country": 1, "senior title": 1}
    [job] = run.unmatched
    assert (job.job_id, job.company, job.title, job.location, job.url) == (
        software, "Acme", "Software Engineer", "London", "https://e.com/1"
    )
    assert job.posted_at == NOW - timedelta(days=1)


def test_results_carry_the_location_level_and_experience(sessions, store):
    add_job(sessions)
    answer = json.dumps({**ANSWER, "seniority": "entry", "experience_years": 0, "experience_mandatory": True})
    [result] = dry_run(sessions, store, client_with(sessions, answer)).results
    assert (result.location, result.seniority, result.experience_years, result.experience_mandatory) == (
        "London, UK", "entry", 0, True
    )


@pytest.mark.parametrize(
    ("mandatory", "years", "words"),
    [(True, None, "required, no number"), (False, None, "none asked"), (True, 2, "2+ years required"),
     (False, 1, "1+ years preferred"), (True, 0, "0+ years required")],
)
def test_experience_is_said_in_a_few_words(mandatory, years, words):
    assert experience_words(mandatory, years) == words


def sheet_run() -> DryRun:
    """Two users' results and one pre-filter job, in no particular order."""
    posted = NOW - timedelta(days=1)
    common = {"company": "Acme", "url": "https://e.com/1", "posted_at": posted, "location": "London"}
    run = DryRun(since=NOW - timedelta(days=14), until=NOW)
    run.results = [
        DryResult(user_name="Vishal", job_id=2, title="Data Analyst", tier="silent", score=61,
                  points={"role_fit": 30, "entry_fit": 6.25}, reasons=("Role: Data Analyst",),
                  notes=("Experience required (no number given)",), seniority="mid", experience_mandatory=True,
                  **common),
        DryResult(user_name="Ana", job_id=2, title="Data Analyst", tier=None, score=None,
                  dropped="role: not one of yours", seniority="mid", experience_mandatory=True, **common),
        DryResult(user_name="Vishal", job_id=1, title="Graduate Analyst", tier="instant", score=80,
                  points={"role_fit": 30}, reasons=("Graduate role", "London"), seniority="graduate", **common),
    ]
    run.unmatched = [Unmatched(3, "Acme", "Insights Specialist", None, "https://e.com/3", posted)]
    return run


def test_the_sheet_lists_every_stage_best_first_with_the_pre_filter_last():
    rows = review_rows(sheet_run())
    assert [(row["stage"], row["user"], row["job id"]) for row in rows] == [
        ("instant", "Vishal", "1"), ("silent", "Vishal", "2"), ("dropped", "Ana", "2"), ("prefilter", "", "3"),
    ]
    instant, silent, dropped, prefilter = rows
    assert instant == {
        "want": "", "stage": "instant", "score": "80", "user": "Vishal", "company": "Acme",
        "title": "Graduate Analyst", "location": "London", "posted": "2026-10-08", "level": "graduate",
        "experience": "none asked", "points": "role 30", "why": "Graduate role; London",
        "url": "https://e.com/1", "job id": "1",
    }
    assert silent["why"] == "Role: Data Analyst; Experience required (no number given)"
    assert (silent["experience"], silent["points"]) == ("required, no number", "role 30 | level 6.25")
    assert (dropped["score"], dropped["points"], dropped["why"]) == ("", "", "role: not one of yours")
    assert (prefilter["level"], prefilter["location"], prefilter["why"]) == ("", "", "pre-filter: no matching role")


def read_sheet(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_the_sheet_is_written_for_excel_with_every_column(tmp_path):
    path = tmp_path / "new" / "review.csv"
    assert write_review(sheet_run(), path) == (4, 0)
    assert path.read_bytes().startswith(codecs.BOM_UTF8)  # so Excel reads accents correctly
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == ",".join(REVIEW_COLUMNS)
    assert [row["title"] for row in read_sheet(path)] == [
        "Graduate Analyst", "Data Analyst", "Data Analyst", "Insights Specialist",
    ]
    assert not path.with_suffix(".tmp").exists()


def test_answers_in_the_last_sheet_are_kept_by_user_and_job(tmp_path):
    path = tmp_path / "review.csv"
    write_review(sheet_run(), path)
    rows = read_sheet(path)
    answers = {("Vishal", "1"): "yes", ("Ana", "2"): " no ", ("", "3"): "maybe"}
    for row in rows:
        row["want"] = answers.get((row["user"], row["job id"]), "")
    rows.append({**rows[0], "job id": "99", "want": "yes"})  # a job no longer in the run: its answer goes
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    assert write_review(sheet_run(), path) == (4, 3)
    assert [(row["user"], row["job id"], row["want"]) for row in read_sheet(path)] == [
        ("Vishal", "1", "yes"), ("Vishal", "2", ""), ("Ana", "2", "no"), ("", "3", "maybe"),
    ]


def test_within_a_stage_the_highest_score_comes_first():
    run = DryRun(since=NOW - timedelta(days=14), until=NOW)
    common = {"user_name": "Vishal", "company": "Acme", "url": "https://e.com/1", "posted_at": NOW, "tier": "digest"}
    run.results = [
        DryResult(job_id=1, title="A Analyst", score=52, **common),  # first by name, last by score
        DryResult(job_id=2, title="B Analyst", score=70, **common),
    ]
    assert [row["title"] for row in review_rows(run)] == ["B Analyst", "A Analyst"]
