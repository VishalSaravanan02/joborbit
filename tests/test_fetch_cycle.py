"""Tests for a full fetch cycle: several companies fetched at once, each saved and recorded.

The three job sites are faked with respx, using our real saved replies.
"""

import copy
import json
from pathlib import Path

import pytest
import respx
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from joborbit.db.models import Base, Company, FetchRun, Job
from joborbit.db.session import create_sqlite_engine
from joborbit.fetchers.ashby import API_URL as ASHBY_API
from joborbit.fetchers.greenhouse import API_URL as GREENHOUSE_API
from joborbit.fetchers.greenhouse import GreenhouseFetcher
from joborbit.fetchers.lever import api_url_for
from joborbit.pipeline import ingest
from joborbit.pipeline.ingest import run_fetch_cycle
from joborbit.utils.http import PoliteClient

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures"
GREENHOUSE = json.loads((FIXTURES / "greenhouse.json").read_text())
LEVER = json.loads((FIXTURES / "lever.json").read_text())
ASHBY = json.loads((FIXTURES / "ashby.json").read_text())


@pytest.fixture
def sessions(tmp_path):
    """A fresh database file with our three real companies, and a way to open sessions on it."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        session.add_all(
            [
                Company(name="Monzo", slug="monzo", size_category="medium", ats_type="greenhouse", ats_token="monzo"),
                Company(name="Palantir", slug="palantir", size_category="mnc", ats_type="lever", ats_token="palantir"),
                Company(name="ElevenLabs", slug="elevenlabs", size_category="startup", ats_type="ashby",
                        ats_token="elevenlabs"),
            ]
        )
    return factory


def fake_sites(greenhouse=GREENHOUSE, lever=LEVER, ashby=ASHBY, greenhouse_status=200):
    """Make the three job sites answer with these replies (call inside @respx.mock)."""
    respx.get(GREENHOUSE_API.format(token="monzo"), params={"content": "true"}).respond(
        greenhouse_status, json=greenhouse
    )
    respx.get(api_url_for("palantir"), params={"mode": "json"}).respond(json=lever)
    respx.get(ASHBY_API.format(token="elevenlabs")).respond(json=ashby)


def fake_sites_except_monzo():
    """Fake only Palantir's and ElevenLabs' sites, for tests where Monzo is never contacted."""
    respx.get(api_url_for("palantir"), params={"mode": "json"}).respond(json=LEVER)
    respx.get(ASHBY_API.format(token="elevenlabs")).respond(json=ASHBY)


async def run_cycle(sessions):
    async with PoliteClient(per_host_delay=0, retry_wait_seconds=0, block_private_addresses=False) as client:
        return await run_fetch_cycle(sessions, client)


def company(sessions, name: str) -> Company:
    with sessions() as session:
        return session.scalars(select(Company).where(Company.name == name)).one()


def runs_for(sessions, name: str) -> list[FetchRun]:
    with sessions() as session:
        company_id = session.scalars(select(Company.id).where(Company.name == name)).one()
        return list(session.scalars(select(FetchRun).where(FetchRun.company_id == company_id).order_by(FetchRun.id)))


def jobs_for(sessions, name: str) -> list[Job]:
    with sessions() as session:
        company_id = session.scalars(select(Company.id).where(Company.name == name)).one()
        return list(session.scalars(select(Job).where(Job.company_id == company_id)))


# --- A normal cycle ------------------------------------------------------------------


@respx.mock
async def test_first_cycle_records_a_baseline_for_every_company(sessions):
    fake_sites()
    result = await run_cycle(sessions)

    assert [outcome.company_name for outcome in result.outcomes] == ["Monzo", "Palantir", "ElevenLabs"]
    assert result.failed == []
    assert result.new_job_ids == []  # first fetch of each company: all baseline
    for name in ("Monzo", "Palantir", "ElevenLabs"):
        assert company(sessions, name).baseline_done is True
        assert company(sessions, name).last_fetch_ok_at is not None
        assert len(jobs_for(sessions, name)) == 3
        [run] = runs_for(sessions, name)
        assert (run.status, run.jobs_returned, run.new_jobs, run.error) == ("ok", 3, 0, None)
        assert run.started_at <= run.finished_at


@respx.mock
async def test_second_cycle_reports_only_genuinely_new_jobs(sessions):
    fake_sites()
    await run_cycle(sessions)

    with_new_job = copy.deepcopy(GREENHOUSE)
    new_job = copy.deepcopy(GREENHOUSE["jobs"][0])
    new_job["id"], new_job["title"] = 99999, "Graduate Data Scientist"
    with_new_job["jobs"].append(new_job)
    respx.clear()
    fake_sites(greenhouse=with_new_job)
    result = await run_cycle(sessions)

    [new_id] = result.new_job_ids
    with sessions() as session:
        assert session.get(Job, new_id).title == "Graduate Data Scientist"
    assert runs_for(sessions, "Monzo")[-1].new_jobs == 1
    assert runs_for(sessions, "Palantir")[-1].new_jobs == 0


# --- Failures ---------------------------------------------------------------------------


@respx.mock
async def test_one_failing_company_does_not_stop_the_others(sessions):
    fake_sites(greenhouse_status=404)
    result = await run_cycle(sessions)

    [failure] = result.failed
    assert failure.company_name == "Monzo"
    assert failure.error.startswith("HTTP 404 from ")  # a site problem, not labelled as a bug in our code
    [run] = runs_for(sessions, "Monzo")
    assert run.status == "error" and "404" in run.error and run.jobs_returned is None
    assert company(sessions, "Monzo").consecutive_failures == 1
    assert company(sessions, "Monzo").baseline_done is False  # no baseline until a fetch works
    assert len(jobs_for(sessions, "Palantir")) == 3
    assert len(jobs_for(sessions, "ElevenLabs")) == 3


@respx.mock
async def test_a_failed_fetch_never_closes_or_counts_missing_jobs(sessions):
    fake_sites()
    await run_cycle(sessions)
    for _ in range(3):
        respx.clear()
        fake_sites(greenhouse_status=500)
        await run_cycle(sessions)

    assert company(sessions, "Monzo").consecutive_failures == 3
    for job in jobs_for(sessions, "Monzo"):
        assert job.missing_count == 0 and job.closed_at is None


@respx.mock
async def test_failures_reset_after_a_successful_fetch(sessions):
    fake_sites(greenhouse_status=404)
    await run_cycle(sessions)
    await run_cycle(sessions)
    assert company(sessions, "Monzo").consecutive_failures == 2

    respx.clear()
    fake_sites()
    await run_cycle(sessions)
    assert company(sessions, "Monzo").consecutive_failures == 0
    assert [run.status for run in runs_for(sessions, "Monzo")] == ["error", "error", "ok"]


@respx.mock
async def test_a_bug_while_fetching_is_recorded_and_the_cycle_continues(sessions, monkeypatch):
    async def broken_fetch(self, token, client):
        raise ValueError("something unexpected")

    monkeypatch.setattr(GreenhouseFetcher, "fetch", broken_fetch)
    fake_sites_except_monzo()
    result = await run_cycle(sessions)

    [failure] = result.failed
    assert failure.error == "Unexpected error: ValueError: something unexpected"
    assert runs_for(sessions, "Monzo")[0].error == failure.error
    assert len(jobs_for(sessions, "Palantir")) == 3


@respx.mock
async def test_a_bug_while_saving_undoes_that_company_only(sessions, monkeypatch):
    real_save = ingest.save_fetched_jobs

    def save_that_breaks_for_monzo(session, company, fetched, now=None):
        result = real_save(session, company, fetched, now)
        if company.name == "Monzo":
            raise RuntimeError("disk on fire")
        return result

    monkeypatch.setattr(ingest, "save_fetched_jobs", save_that_breaks_for_monzo)
    fake_sites()
    result = await run_cycle(sessions)

    [failure] = result.failed
    assert "RuntimeError: disk on fire" in failure.error
    assert jobs_for(sessions, "Monzo") == []  # Monzo's half-saved jobs were undone
    assert company(sessions, "Monzo").baseline_done is False
    assert company(sessions, "Monzo").consecutive_failures == 1
    assert runs_for(sessions, "Monzo")[0].status == "error"
    assert len(jobs_for(sessions, "Palantir")) == 3


@respx.mock
async def test_very_long_errors_are_shortened(sessions, monkeypatch):
    async def wordy_fetch(self, token, client):
        raise ValueError("x" * 5000)

    monkeypatch.setattr(GreenhouseFetcher, "fetch", wordy_fetch)
    fake_sites_except_monzo()
    await run_cycle(sessions)
    assert len(runs_for(sessions, "Monzo")[0].error) == ingest.MAX_ERROR_CHARS


# --- Which companies are fetched --------------------------------------------------------


@respx.mock
async def test_inactive_companies_are_not_fetched(sessions):
    with sessions() as session, session.begin():
        session.scalars(select(Company).where(Company.name == "Monzo")).one().active = False
    fake_sites_except_monzo()

    result = await run_cycle(sessions)

    assert [outcome.company_name for outcome in result.outcomes] == ["Palantir", "ElevenLabs"]
    assert runs_for(sessions, "Monzo") == []


@respx.mock
async def test_an_active_company_without_a_fetcher_is_recorded_as_an_error(sessions):
    with sessions() as session, session.begin():
        session.scalars(select(Company).where(Company.name == "Monzo")).one().ats_type = "workday"
    fake_sites_except_monzo()

    result = await run_cycle(sessions)

    [failure] = result.failed
    assert failure.company_name == "Monzo" and "workday" in failure.error


async def test_a_cycle_with_no_active_companies_does_nothing(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    Base.metadata.create_all(engine)
    result = await run_cycle(sessionmaker(bind=engine, expire_on_commit=False))
    assert result.outcomes == [] and result.new_job_ids == []
