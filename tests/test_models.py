"""Checks that the database tables work together as intended."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, FetchRun, Job


@pytest.fixture
def session():
    """A brand-new, empty database in memory for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_company(**overrides) -> Company:
    data = {"name": "Example Ltd", "slug": "example-ltd", "size_category": "startup", "countries": ["GB"]}
    data.update(overrides)
    return Company(**data)


def make_job(company: Company, external_id: str = "123") -> Job:
    return Job(
        company_id=company.id,
        ats_type="greenhouse",
        external_id=external_id,
        url=f"https://example.com/jobs/{external_id}",
        title="Graduate Data Scientist",
        location_raw="London, UK",
    )


def test_company_and_job_save_and_load_with_defaults(session):
    company = make_company(ats_type="greenhouse", ats_token="example")
    session.add(company)
    session.flush()  # gives the company its id
    session.add(make_job(company))
    session.commit()

    job = session.scalars(select(Job)).one()
    assert job.company.name == "Example Ltd"
    assert job.prefilter_status == "pending"
    assert job.analysis_status == "pending"
    assert job.is_baseline is False
    assert job.country_codes == []
    assert job.missing_count == 0
    assert job.first_seen_at is not None
    assert company.active is True
    assert company.baseline_done is False
    assert company.consecutive_failures == 0


def test_same_job_twice_for_one_company_is_rejected(session):
    company = make_company()
    session.add(company)
    session.flush()
    session.add(make_job(company, "123"))
    session.add(make_job(company, "123"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_same_job_id_at_two_companies_is_allowed(session):
    first = make_company()
    second = make_company(name="Other Ltd", slug="other-ltd")
    session.add_all([first, second])
    session.flush()
    session.add_all([make_job(first, "123"), make_job(second, "123")])
    session.commit()
    assert len(session.scalars(select(Job)).all()) == 2


def test_company_slug_must_be_unique(session):
    session.add_all([make_company(), make_company(name="Example Again")])
    with pytest.raises(IntegrityError):
        session.commit()


def test_fetch_run_starts_as_running(session):
    company = make_company()
    session.add(company)
    session.flush()
    run = FetchRun(company_id=company.id)
    session.add(run)
    session.commit()
    assert run.status == "running"
    assert run.started_at is not None
    assert run.finished_at is None
