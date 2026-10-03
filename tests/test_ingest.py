"""Tests for saving fetched jobs: baseline, new jobs, edits, duplicates, closing and reopening."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, Job
from joborbit.db.session import create_sqlite_engine
from joborbit.fetchers.base import NormalisedJob
from joborbit.fetchers.greenhouse import GreenhouseFetcher
from joborbit.pipeline.ingest import CLOSE_AFTER_MISSING, SKIPPED, save_fetched_jobs

START = datetime(2026, 10, 1, 9, 0)
CYCLE = timedelta(minutes=20)
MONZO_JOBS = GreenhouseFetcher().parse_jobs(
    json.loads((Path(__file__).parent / "fixtures" / "greenhouse.json").read_text())["jobs"], "monzo"
)


@pytest.fixture
def session():
    """A brand-new, empty database in memory for each test, with the real app's settings."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def company(session):
    """A company that has already had its first (baseline) fetch."""
    company = Company(name="Acme", slug="acme", size_category="startup", ats_type="greenhouse", ats_token="acme")
    company.baseline_done = True
    session.add(company)
    session.flush()
    return company


def make_job(external_id: str, title: str = "Graduate Data Analyst", location: str = "London, UK", **extra):
    return NormalisedJob(
        external_id=external_id,
        title=title,
        url=f"https://example.com/jobs/{external_id}",
        location_raw=location,
        description_html=extra.pop("description_html", "<p>Join our data team.</p>"),
        **extra,
    )


def save(session, company, jobs, now):
    result = save_fetched_jobs(session, company, jobs, now=now)
    session.commit()
    return result


def job_by_id(session, external_id: str) -> Job:
    return session.scalars(select(Job).where(Job.external_id == external_id)).one()


# --- Baseline ------------------------------------------------------------------


def test_first_fetch_is_saved_as_baseline_and_never_alerted(session):
    monzo = Company(name="Monzo", slug="monzo", size_category="medium", ats_type="greenhouse", ats_token="monzo")
    session.add(monzo)
    session.flush()

    result = save(session, monzo, MONZO_JOBS, START)

    assert result.jobs_returned == 3
    assert result.baseline == 3
    assert result.new_job_ids == []
    assert monzo.baseline_done is True
    for job in session.scalars(select(Job)):
        assert job.is_baseline is True
        assert job.prefilter_status == SKIPPED
        assert job.prefilter_reason == "baseline"


def test_real_jobs_are_saved_with_clean_text_countries_and_hashes(session):
    monzo = Company(name="Monzo", slug="monzo", size_category="medium", ats_type="greenhouse", ats_token="monzo")
    session.add(monzo)
    session.flush()
    save(session, monzo, MONZO_JOBS, START)

    job = job_by_id(session, "8143930")
    assert job.title == "Anaplan Support Analyst"
    assert job.ats_type == "greenhouse"
    assert job.location_raw == "Cardiff, London or Remote (UK)"
    assert job.country_codes == ["GB"]
    assert job.description_text and "<" not in job.description_text
    assert job.content_hash and job.fingerprint
    assert job.first_seen_at == job.last_seen_at == START
    assert job.posted_at == datetime(2026, 8, 24, 13, 4, 42)


def test_second_fetch_of_the_same_jobs_finds_nothing_new(session):
    monzo = Company(name="Monzo", slug="monzo", size_category="medium", ats_type="greenhouse", ats_token="monzo")
    session.add(monzo)
    session.flush()
    save(session, monzo, MONZO_JOBS, START)

    result = save(session, monzo, MONZO_JOBS, START + CYCLE)

    assert result.new_job_ids == [] and result.baseline == 0 and result.edited == 0
    assert len(session.scalars(select(Job)).all()) == 3
    assert job_by_id(session, "8143930").last_seen_at == START + CYCLE
    assert job_by_id(session, "8143930").first_seen_at == START


# --- New and edited jobs -------------------------------------------------------


def test_a_new_job_after_the_baseline_goes_to_the_prefilter(session, company):
    save(session, company, [make_job("1")], START)  # company already had its baseline: this one is new
    result = save(session, company, [make_job("1"), make_job("2", "Junior Data Scientist")], START + CYCLE)

    new_job = job_by_id(session, "2")
    assert result.new_job_ids == [new_job.id]
    assert new_job.is_baseline is False
    assert new_job.prefilter_status == "pending"
    assert new_job.first_seen_at == START + CYCLE


def test_an_edited_ad_is_updated_but_not_alerted_again(session, company):
    save(session, company, [make_job("1")], START)
    edited = make_job("1", description_html="<p>Join our data team. Python and SQL required.</p>")

    result = save(session, company, [edited], START + CYCLE)

    job = job_by_id(session, "1")
    assert result.edited == 1
    assert result.new_job_ids == []
    assert "Python and SQL required" in job.description_text
    assert job.first_seen_at == START


def test_a_moved_apply_link_is_updated(session, company):
    save(session, company, [make_job("1")], START)
    moved = make_job("1").model_copy(update={"url": "https://example.com/new-link/1"})
    save(session, company, [moved], START + CYCLE)
    assert job_by_id(session, "1").url == "https://example.com/new-link/1"


# --- Duplicates -------------------------------------------------------------------


def test_a_repost_of_an_open_job_is_linked_and_not_alerted(session, company):
    save(session, company, [make_job("1")], START)
    original = job_by_id(session, "1")

    repost = make_job("2", title="  graduate data ANALYST ", location="London UK")  # same job, new ID
    result = save(session, company, [make_job("1"), repost], START + CYCLE)

    job = job_by_id(session, "2")
    assert result.duplicates == 1 and result.new_job_ids == []
    assert job.duplicate_of_id == original.id
    assert job.prefilter_status == SKIPPED
    assert job.prefilter_reason == f"duplicate of job {original.id}"


def test_two_identical_new_jobs_in_one_fetch_alert_only_once(session, company):
    result = save(session, company, [make_job("1"), make_job("2")], START)
    first, second = job_by_id(session, "1"), job_by_id(session, "2")
    assert result.new_job_ids == [first.id]
    assert second.duplicate_of_id == first.id


def test_a_job_matching_only_a_closed_job_is_new(session, company):
    save(session, company, [make_job("1")], START)
    for cycle in range(1, CLOSE_AFTER_MISSING + 1):
        save(session, company, [make_job("other", "Data Engineer")], START + cycle * CYCLE)
    assert job_by_id(session, "1").closed_at is not None

    result = save(session, company, [make_job("other", "Data Engineer"), make_job("2")], START + 10 * CYCLE)

    assert result.new_job_ids == [job_by_id(session, "2").id]
    assert job_by_id(session, "2").duplicate_of_id is None


def test_baseline_jobs_are_linked_to_matching_open_jobs_after_a_source_change(session, company):
    """When a company moves ATS, its re-listed jobs are linked to the old ones (but still baseline)."""
    save(session, company, [make_job("old-1")], START)
    company.ats_type, company.ats_token, company.baseline_done = "ashby", "acme", False

    result = save(session, company, [make_job("new-1")], START + CYCLE)

    new = job_by_id(session, "new-1")
    assert result.baseline == 1 and result.new_job_ids == []
    assert new.ats_type == "ashby"
    assert new.duplicate_of_id == job_by_id(session, "old-1").id
    assert new.prefilter_reason == "baseline"


def test_the_same_id_twice_in_one_fetch_is_saved_once(session, company):
    result = save(session, company, [make_job("1"), make_job("1")], START)
    assert result.jobs_returned == 1
    assert len(session.scalars(select(Job)).all()) == 1


# --- Closing --------------------------------------------------------------------


def test_a_missing_job_closes_only_after_two_fetches_without_it(session, company):
    save(session, company, [make_job("1"), make_job("2", "Data Engineer")], START)

    first = save(session, company, [make_job("2", "Data Engineer")], START + CYCLE)
    assert first.closed == 0
    assert job_by_id(session, "1").missing_count == 1
    assert job_by_id(session, "1").closed_at is None

    second = save(session, company, [make_job("2", "Data Engineer")], START + 2 * CYCLE)
    assert second.closed == 1
    assert job_by_id(session, "1").closed_at == START + 2 * CYCLE


def test_a_job_that_reappears_before_closing_starts_counting_again(session, company):
    save(session, company, [make_job("1"), make_job("2", "Data Engineer")], START)
    save(session, company, [make_job("2", "Data Engineer")], START + CYCLE)  # missing once
    save(session, company, [make_job("1"), make_job("2", "Data Engineer")], START + 2 * CYCLE)  # back
    save(session, company, [make_job("2", "Data Engineer")], START + 3 * CYCLE)  # missing once again
    assert job_by_id(session, "1").missing_count == 1
    assert job_by_id(session, "1").closed_at is None


def test_other_companies_jobs_are_never_touched(session, company):
    other = Company(name="Other", slug="other", size_category="startup", ats_type="lever", ats_token="other")
    other.baseline_done = True
    session.add(other)
    session.flush()
    save(session, other, [make_job("1")], START)

    for cycle in range(1, 4):
        save(session, company, [make_job("x", "Data Engineer")], START + cycle * CYCLE)

    other_job = session.scalars(select(Job).where(Job.company_id == other.id)).one()
    assert other_job.missing_count == 0 and other_job.closed_at is None


# --- Empty replies ------------------------------------------------------------------


def test_an_empty_reply_closes_nothing_at_first(session, company):
    save(session, company, [make_job("1")], START)
    for cycle in range(1, 5):
        result = save(session, company, [], START + cycle * CYCLE)
        assert result.empty_reply_ignored is True
    job = job_by_id(session, "1")
    assert job.missing_count == 0 and job.closed_at is None


def test_jobs_close_after_a_full_day_of_empty_replies(session, company):
    save(session, company, [make_job("1")], START)
    day_later = START + timedelta(hours=24)

    first = save(session, company, [], day_later)
    second = save(session, company, [], day_later + CYCLE)

    assert first.empty_reply_ignored is False and first.closed == 0
    assert second.closed == 1
    assert job_by_id(session, "1").closed_at == day_later + CYCLE


def test_an_empty_reply_from_a_company_with_no_open_jobs_is_fine(session, company):
    result = save(session, company, [], START)
    assert result.jobs_returned == 0 and result.closed == 0 and result.empty_reply_ignored is False


# --- Reopening ------------------------------------------------------------------


def close_job_1(session, company):
    """Save job 1, then let it go missing until it is closed. Returns when it was closed."""
    save(session, company, [make_job("1"), make_job("2", "Data Engineer")], START)
    for cycle in range(1, CLOSE_AFTER_MISSING + 1):
        save(session, company, [make_job("2", "Data Engineer")], START + cycle * CYCLE)
    closed_at = job_by_id(session, "1").closed_at
    assert closed_at is not None
    return closed_at


def test_a_job_back_after_a_short_gap_is_reopened_quietly(session, company):
    closed_at = close_job_1(session, company)
    result = save(session, company, [make_job("1"), make_job("2", "Data Engineer")], closed_at + timedelta(days=2))

    job = job_by_id(session, "1")
    assert result.reopened == 1 and result.new_job_ids == []
    assert job.closed_at is None and job.missing_count == 0


def test_a_job_back_after_a_long_gap_is_treated_as_a_fresh_posting(session, company):
    closed_at = close_job_1(session, company)
    job = job_by_id(session, "1")
    job.prefilter_status, job.analysis_status = "passed", "done"  # as if it went through Phase 2 the first time
    session.commit()

    result = save(session, company, [make_job("1"), make_job("2", "Data Engineer")], closed_at + timedelta(days=8))

    assert result.reopened == 1 and result.new_job_ids == [job.id]
    assert job.closed_at is None
    assert job.prefilter_status == "pending" and job.analysis_status == "pending"


def test_a_baseline_job_back_after_a_long_gap_is_no_longer_baseline(session):
    acme = Company(name="Acme", slug="acme", size_category="startup", ats_type="greenhouse", ats_token="acme")
    session.add(acme)
    session.flush()
    save(session, acme, [make_job("1"), make_job("2", "Data Engineer")], START)  # baseline run
    for cycle in range(1, CLOSE_AFTER_MISSING + 1):
        save(session, acme, [make_job("2", "Data Engineer")], START + cycle * CYCLE)
    closed_at = job_by_id(session, "1").closed_at

    result = save(session, acme, [make_job("1"), make_job("2", "Data Engineer")], closed_at + timedelta(days=8))

    job = job_by_id(session, "1")
    assert result.new_job_ids == [job.id]
    assert job.is_baseline is False
    assert job.prefilter_status == "pending" and job.prefilter_reason is None


def test_a_long_gap_return_that_matches_an_open_job_is_a_duplicate(session, company):
    closed_at = close_job_1(session, company)
    save(session, company, [make_job("2", "Data Engineer"), make_job("3")], closed_at + timedelta(days=1))  # re-post
    repost = job_by_id(session, "3")

    result = save(
        session,
        company,
        [make_job("1"), make_job("2", "Data Engineer"), make_job("3")],
        closed_at + timedelta(days=8),
    )

    job = job_by_id(session, "1")
    assert result.new_job_ids == [] and result.duplicates == 1
    assert job.duplicate_of_id == repost.id
    assert job.prefilter_status == SKIPPED
