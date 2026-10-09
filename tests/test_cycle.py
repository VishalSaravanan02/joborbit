"""Tests for what happens after each fetch (joborbit/pipeline/cycle.py). Nothing here contacts a real LLM."""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from joborbit.config import load_app_settings
from joborbit.db.models import Base, Company, Job, JobAnalysis, Match, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.llm.client import LlmClient, LlmUnavailable, Provider, Reply
from joborbit.pipeline import cycle
from joborbit.pipeline.cycle import process_new_jobs

START = datetime(2026, 10, 9, 8, 0)
GOOD = json.dumps(
    {
        "countries": ["GB"], "cities": ["London"], "work_mode": "hybrid", "seniority": "graduate",
        "is_graduate_scheme": False, "experience_years": None, "experience_mandatory": False,
        "required_languages": [], "role_families": ["data_analyst"], "matched_custom_roles": [],
        "skills": ["SQL"], "min_degree": "bachelor", "deadline": None,
        "summary": "A graduate data analyst job in London.",
    }
)


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
    """A database file with one company and one active user who wants data analyst jobs in the UK."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        session.add(Company(name="Acme", slug="acme", size_category="startup"))
        user = User(telegram_id=1, display_name="Vishal")
        user.profile = UserProfile(roles=["data_analyst"], countries=["GB"])
        session.add(user)
    return factory


def add_job(sessions, title: str, minutes: int = 0, **fields) -> int:
    data = {
        "company_id": 1, "ats_type": "greenhouse", "external_id": f"{title}-{minutes}", "url": "https://e.com",
        "title": title, "location_raw": "London, UK", "description_text": "Join our data team.",
        "first_seen_at": START + timedelta(minutes=minutes),
    }
    with sessions() as session, session.begin():
        job = Job(**{**data, **fields})
        session.add(job)
        session.flush()
        return job.id


def client_with(sessions, *replies) -> LlmClient:
    return LlmClient(FakeProvider(*replies), load_app_settings().llm, sessions, now=lambda: START)


def statuses(sessions, job_id: int) -> tuple[str, str]:
    with sessions() as session:
        job = session.get_one(Job, job_id)
        return job.prefilter_status, job.analysis_status


def test_the_prefilter_decides_first_and_only_jobs_that_pass_reach_the_llm(sessions):
    wanted = add_job(sessions, "Graduate Data Analyst", minutes=10)
    senior = add_job(sessions, "Senior Data Analyst", minutes=5)
    client = client_with(sessions, GOOD)
    result = process_new_jobs(session_factory=sessions, client=client)
    assert result.prefilter.passed_job_ids == [wanted] and result.prefilter.rejected == {"senior title": 1}
    assert result.analysis.done == [wanted] and result.waiting == 0 and result.retried == 0
    assert len(client.provider.requests) == 1 and "Title: Graduate Data Analyst" in client.provider.requests[0]
    assert statuses(sessions, wanted) == ("passed", "done")
    assert statuses(sessions, senior)[0] == "rejected"


def test_jobs_left_from_an_earlier_run_are_picked_up_too(sessions):
    """A run that stopped between steps leaves jobs waiting: either step takes them next time."""
    not_yet_filtered = add_job(sessions, "Data Analyst", minutes=0)
    not_yet_analysed = add_job(sessions, "Junior Data Analyst", minutes=-60, prefilter_status="passed")
    result = process_new_jobs(session_factory=sessions, client=client_with(sessions, GOOD, GOOD))
    assert result.analysis.done == [not_yet_filtered, not_yet_analysed]


def test_with_analysis_switched_off_nothing_is_asked(sessions):
    job_id = add_job(sessions, "Graduate Data Analyst")
    client = client_with(sessions)
    result = process_new_jobs(analyse=False, session_factory=sessions, client=client)
    assert result.analysis is None and result.waiting == 1
    assert client.provider.requests == []
    assert statuses(sessions, job_id) == ("passed", "pending")


def test_when_the_llm_is_unreachable_the_verdicts_are_kept_and_the_jobs_wait(sessions):
    job_id = add_job(sessions, "Graduate Data Analyst")
    client = client_with(sessions, LlmUnavailable("Could not reach OpenAI"))
    result = process_new_jobs(session_factory=sessions, client=client)
    assert result.analysis.stopped == "Could not reach OpenAI" and result.waiting == 1
    assert statuses(sessions, job_id) == ("passed", "pending")


def test_a_missing_key_stops_the_analysis_but_not_the_cycle(sessions, monkeypatch):
    def no_key(session_factory=None):
        raise LlmUnavailable("OPENAI_API_KEY is not set in .env")

    monkeypatch.setattr(cycle, "make_client", no_key)
    job_id = add_job(sessions, "Graduate Data Analyst")
    result = process_new_jobs(session_factory=sessions)
    assert result.analysis.stopped == "OPENAI_API_KEY is not set in .env"
    assert result.analysis.done == [] and result.waiting == 1
    assert statuses(sessions, job_id) == ("passed", "pending")


def test_at_most_limit_jobs_are_analysed(sessions):
    newer = add_job(sessions, "Graduate Data Analyst", minutes=10)
    older = add_job(sessions, "Data Analyst", minutes=0)
    result = process_new_jobs(limit=1, session_factory=sessions, client=client_with(sessions, GOOD))
    assert result.analysis.done == [newer] and result.waiting == 1
    assert statuses(sessions, older) == ("passed", "pending")


def test_failed_jobs_are_only_tried_again_when_asked(sessions):
    job_id = add_job(sessions, "Graduate Data Analyst", prefilter_status="passed", analysis_status="failed")
    left = process_new_jobs(session_factory=sessions, client=client_with(sessions))
    assert (left.retried, left.analysis.done) == (0, [])
    retried = process_new_jobs(retry=True, session_factory=sessions, client=client_with(sessions, GOOD))
    assert (retried.retried, retried.analysis.done) == (1, [job_id])
    assert statuses(sessions, job_id) == ("passed", "done")


def test_jobs_analysed_in_a_cycle_are_matched_in_the_same_cycle(sessions):
    job_id = add_job(sessions, "Graduate Data Analyst")
    result = process_new_jobs(session_factory=sessions, client=client_with(sessions, GOOD))

    [new] = result.matching.matches
    assert (new.job_id, new.user_name, new.tier) == (job_id, "Vishal", "instant")
    assert result.matching.jobs == 1
    with sessions() as session:
        assert session.get_one(Job, job_id).matched_at is not None
        assert session.scalars(select(Match.job_id)).all() == [job_id]


def test_jobs_analysed_earlier_are_matched_even_with_analysis_switched_off(sessions):
    """Matching costs nothing, so a run without analysis still matches what earlier runs analysed."""
    job_id = add_job(sessions, "Graduate Data Analyst", prefilter_status="passed", analysis_status="done")
    with sessions() as session, session.begin():
        fields = json.loads(GOOD)
        fields["deadline"] = None
        made_by = {"model": "gpt-6-luna", "prompt_version": "2", "input_tokens": 1, "output_tokens": 1}
        session.add(JobAnalysis(job_id=job_id, **made_by, **fields))
    client = client_with(sessions)

    result = process_new_jobs(analyse=False, session_factory=sessions, client=client)

    assert result.analysis is None and client.provider.requests == []
    assert [new.job_id for new in result.matching.matches] == [job_id]
