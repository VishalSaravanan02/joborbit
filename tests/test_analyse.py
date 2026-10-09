"""Tests for the LLM step (joborbit/pipeline/analyse.py). Nothing here contacts a real LLM."""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from joborbit.config import load_app_settings
from joborbit.db.models import Base, Company, Job, JobAnalysis, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.llm.client import LlmClient, LlmUnavailable, Provider, Reply
from joborbit.llm.prompts import PROMPT_VERSION, SCHEMA_NAME, instructions
from joborbit.llm.schemas import answer_schema
from joborbit.pipeline.analyse import active_custom_roles, pending_job_ids, retry_failed, run_analysis

START = datetime(2026, 10, 9, 8, 0)
ANSWER = {
    "countries": ["GB"], "cities": ["London"], "work_mode": "hybrid", "seniority": "graduate",
    "is_graduate_scheme": False, "experience_years": None, "experience_mandatory": False,
    "required_languages": [], "role_families": ["data_analyst"], "matched_custom_roles": [],
    "skills": ["SQL"], "min_degree": "bachelor", "deadline": "2026-11-30",
    "summary": "A graduate data analyst job in London.",
}


class FakeProvider(Provider):
    """Answers with the prepared replies in order, keeps every request, and can run a check mid-call."""

    def __init__(self, *replies: str | Exception, during_call=None) -> None:
        self.replies = list(replies)
        self.requests: list[tuple] = []
        self.during_call = during_call

    def send(self, instructions, messages, schema_name, schema):
        self.requests.append((instructions, list(messages), schema_name, schema))
        if self.during_call:
            self.during_call()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply("done", reply, "gpt-6-luna-2026-05-18", 3000, 2000, 500)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sessions(db_path):
    """A database file with one company and one active user who has a custom role."""
    engine = create_sqlite_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        session.add(Company(name="Acme", slug="acme", size_category="startup"))
        user = User(telegram_id=1, display_name="Vishal")
        user.profile = UserProfile(roles=["data_analyst"], custom_roles=["Insights Analyst"], countries=["GB"])
        session.add(user)
    return factory


def add_job(sessions, title: str = "Graduate Data Analyst", minutes: int = 0, **fields) -> int:
    data = {
        "company_id": 1, "ats_type": "greenhouse", "external_id": f"{title}-{minutes}", "url": "https://e.com",
        "title": title, "location_raw": "London", "description_text": "Join our data team.",
        "first_seen_at": START + timedelta(minutes=minutes), "prefilter_status": "passed",
    }
    with sessions() as session, session.begin():
        job = Job(**{**data, **fields})
        session.add(job)
        session.flush()
        return job.id


def client_with(sessions, *replies, during_call=None) -> LlmClient:
    provider = FakeProvider(*replies, during_call=during_call)
    return LlmClient(provider, load_app_settings().llm, sessions, now=lambda: START)


def job_and_analysis(sessions, job_id: int) -> tuple[Job, JobAnalysis | None]:
    with sessions() as session:
        job = session.get_one(Job, job_id)
        return job, session.get(JobAnalysis, job_id)


GOOD = json.dumps(ANSWER)
BAD = json.dumps({**ANSWER, "countries": ["US"]})


# --- Which jobs are analysed ---------------------------------------------------------------------


def test_only_open_jobs_that_passed_and_wait_for_analysis_are_picked_newest_first(sessions):
    older = add_job(sessions, minutes=0)
    newer = add_job(sessions, minutes=30)
    add_job(sessions, "Rejected", prefilter_status="rejected")
    add_job(sessions, "Baseline", prefilter_status="skipped")
    add_job(sessions, "Analysed", analysis_status="done")
    add_job(sessions, "Failed", analysis_status="failed")
    add_job(sessions, "Closed", closed_at=START)
    with sessions() as session:
        assert pending_job_ids(session) == [newer, older]


def test_custom_roles_come_from_active_users_paused_ones_included(sessions):
    with sessions() as session, session.begin():
        paused = User(telegram_id=2, display_name="Paused", paused=True)
        paused.profile = UserProfile(custom_roles=["Pricing Analyst", "insights analyst"], countries=["GB"])
        gone = User(telegram_id=3, display_name="Gone", is_active=False)
        gone.profile = UserProfile(custom_roles=["Growth Hacker"], countries=["GB"])
        session.add_all([paused, gone])
    with sessions() as session:
        assert active_custom_roles(session) == ["Insights Analyst", "Pricing Analyst"]


# --- A good answer ------------------------------------------------------------------------------------


def test_a_good_answer_is_saved_and_the_job_marked_done(sessions):
    job_id = add_job(sessions)
    run = run_analysis(client_with(sessions, GOOD), sessions)
    assert (run.done, run.failed, run.stopped) == ([job_id], [], None)
    assert run.cost_usd > 0
    job, analysis = job_and_analysis(sessions, job_id)
    assert job.analysis_status == "done"
    assert (analysis.countries, analysis.seniority, analysis.role_families) == (["GB"], "graduate", ["data_analyst"])
    assert analysis.deadline.isoformat() == "2026-11-30" and analysis.summary == ANSWER["summary"]
    assert (analysis.model, analysis.prompt_version) == ("gpt-6-luna-2026-05-18", PROMPT_VERSION)
    assert (analysis.input_tokens, analysis.output_tokens) == (3000, 500)
    assert analysis.analysed_at is not None


def test_the_llm_is_asked_with_the_prompt_the_job_and_the_custom_roles(sessions):
    add_job(sessions, posted_at=datetime(2026, 10, 1, 12, 0))
    client = client_with(sessions, GOOD)
    run_analysis(client, sessions)
    [(sent_instructions, messages, schema_name, schema)] = client.provider.requests
    assert sent_instructions == instructions()
    assert (schema_name, schema) == (SCHEMA_NAME, answer_schema(["Insights Analyst"]))
    assert messages[0].content == (
        "Company: Acme\nTitle: Graduate Data Analyst\nLocation: London\nPosted: 2026-10-01\n"
        "Custom roles to check: Insights Analyst\nPosting:\nJoin our data team."
    )


def test_a_job_without_a_posted_date_uses_the_day_it_was_first_seen(sessions):
    add_job(sessions)
    client = client_with(sessions, GOOD)
    run_analysis(client, sessions)
    assert "Posted: 2026-10-09\n" in client.provider.requests[0][1][0].content


def test_a_matched_custom_role_is_saved_as_the_user_wrote_it(sessions):
    job_id = add_job(sessions, "Junior Insights Analyst")
    run_analysis(client_with(sessions, json.dumps({**ANSWER, "matched_custom_roles": ["insights analyst"]})), sessions)
    assert job_and_analysis(sessions, job_id)[1].matched_custom_roles == ["Insights Analyst"]


def test_a_reopened_job_has_its_analysis_updated_not_duplicated(sessions):
    job_id = add_job(sessions)
    run_analysis(client_with(sessions, GOOD), sessions)
    with sessions() as session, session.begin():
        session.get_one(Job, job_id).analysis_status = "pending"  # as when a job comes back after 7+ days
    run_analysis(client_with(sessions, json.dumps({**ANSWER, "seniority": "entry"})), sessions)
    with sessions() as session:
        [analysis] = session.scalars(select(JobAnalysis)).all()
        assert analysis.seniority == "entry"


def test_at_most_limit_jobs_are_analysed(sessions):
    jobs = [add_job(sessions, minutes=n) for n in range(3)]
    run = run_analysis(client_with(sessions, GOOD, GOOD), sessions, limit=2)
    assert run.done == [jobs[2], jobs[1]]
    assert job_and_analysis(sessions, jobs[0])[0].analysis_status == "pending"


# --- When things go wrong ----------------------------------------------------------------------------


def test_a_job_without_a_usable_answer_is_failed_and_the_run_goes_on(sessions):
    first = add_job(sessions, minutes=10)
    second = add_job(sessions, minutes=0)
    run = run_analysis(client_with(sessions, BAD, BAD, GOOD), sessions)
    assert (run.done, run.failed, run.stopped) == ([second], [first], None)
    job, analysis = job_and_analysis(sessions, first)
    assert job.analysis_status == "failed" and analysis is None


def test_when_the_llm_is_unavailable_the_run_stops_and_the_rest_wait(sessions):
    first = add_job(sessions, minutes=20)
    second = add_job(sessions, minutes=10)
    third = add_job(sessions, minutes=0)
    run = run_analysis(client_with(sessions, GOOD, LlmUnavailable("Could not reach OpenAI")), sessions)
    assert (run.done, run.failed, run.stopped) == ([first], [], "Could not reach OpenAI")
    for job_id in (second, third):
        assert job_and_analysis(sessions, job_id)[0].analysis_status == "pending"


def test_the_budget_stops_the_run_too(sessions):
    add_job(sessions, minutes=0)
    add_job(sessions, minutes=5)
    config = load_app_settings().llm.model_copy(update={"daily_call_cap": 1})
    client = LlmClient(FakeProvider(GOOD, GOOD), config, sessions, now=lambda: START)
    run = run_analysis(client, sessions)
    assert len(run.done) == 1 and run.stopped.startswith("Daily cap reached")


def test_no_database_write_is_held_while_the_llm_is_asked(sessions, db_path):
    """SQLite has one writer at a time: holding a write during a slow LLM call would block everyone else."""
    add_job(sessions, minutes=10)
    add_job(sessions, minutes=0)

    def another_process_writes():
        connection = sqlite3.connect(db_path, timeout=0.1)  # fails at once if someone holds the write lock
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("ROLLBACK")
        finally:
            connection.close()

    run = run_analysis(client_with(sessions, GOOD, GOOD, during_call=another_process_writes), sessions)
    assert len(run.done) == 2


def test_a_job_analysed_meanwhile_is_not_asked_about_again(sessions):
    first = add_job(sessions, minutes=10)
    second = add_job(sessions, minutes=0)

    def another_run_finishes_the_second_job():
        with sessions() as session, session.begin():
            session.get_one(Job, second).analysis_status = "done"

    client = client_with(sessions, GOOD, during_call=another_run_finishes_the_second_job)
    run = run_analysis(client, sessions)
    assert run.done == [first] and len(client.provider.requests) == 1


# --- Retrying failed jobs -----------------------------------------------------------------------------


def test_failed_jobs_can_be_sent_back_for_another_try(sessions):
    failed = add_job(sessions, analysis_status="failed")
    done = add_job(sessions, "Done", analysis_status="done")
    with sessions() as session, session.begin():
        assert retry_failed(session) == 1
    assert job_and_analysis(sessions, failed)[0].analysis_status == "pending"
    assert job_and_analysis(sessions, done)[0].analysis_status == "done"
