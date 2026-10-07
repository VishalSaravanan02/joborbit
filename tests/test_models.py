"""Checks that the database tables work together as intended."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, FetchRun, Job, User, UserCompanyPref, UserProfile
from joborbit.db.session import create_sqlite_engine


@pytest.fixture
def session():
    """A brand-new, empty database in memory for each test, with the real app's settings."""
    engine = create_sqlite_engine("sqlite:///:memory:")
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


def test_links_between_tables_are_enforced(session):
    """A job pointing at a company that doesn't exist must be refused, as in the real database."""
    session.add(Job(company_id=999, ats_type="greenhouse", external_id="1", url="https://e.com/1", title="Analyst"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_a_duplicate_job_links_to_the_original(session):
    company = make_company()
    session.add(company)
    session.flush()
    original = make_job(company, "100")
    session.add(original)
    session.flush()
    repost = make_job(company, "200")
    repost.duplicate_of_id = original.id
    session.add(repost)
    session.commit()

    assert original.duplicate_of_id is None
    assert repost.duplicate_of_id == original.id


def test_deleting_the_original_keeps_the_duplicate_but_clears_the_link(session):
    """The daily clean-up may delete old closed jobs; their re-posts must survive."""
    company = make_company()
    session.add(company)
    session.flush()
    original = make_job(company, "100")
    session.add(original)
    session.flush()
    repost = make_job(company, "200")
    repost.duplicate_of_id = original.id
    session.add(repost)
    session.commit()

    session.delete(original)
    session.commit()
    session.expire_all()  # read the job again from the database

    [remaining] = session.scalars(select(Job)).all()
    assert remaining.external_id == "200"
    assert remaining.duplicate_of_id is None


# --- Users, profiles and company choices ---------------------------------------------


def make_user(telegram_id: int = 123456789, **overrides) -> User:
    return User(telegram_id=telegram_id, display_name="Vishal", **overrides)


def test_a_user_and_profile_save_with_sensible_defaults(session):
    user = make_user()
    user.profile = UserProfile(roles=["data_scientist"], countries=["GB"], languages=["en", "es"])
    session.add(user)
    session.commit()

    assert (user.is_admin, user.is_active, user.paused) == (False, True, False)
    profile = session.get(UserProfile, user.id)
    assert profile.roles == ["data_scientist"] and profile.custom_roles == []
    assert profile.alert_style == "balanced"
    assert profile.include_internships is False and profile.transfer_boost is True
    assert profile.weights is None  # empty means "use the current default weights"
    assert profile.updated_at is not None


def test_telegram_ids_are_unique(session):
    session.add_all([make_user(1), make_user(1)])
    with pytest.raises(IntegrityError):
        session.commit()


def test_large_telegram_ids_are_stored_exactly(session):
    session.add(make_user(9_876_543_210_123))  # bigger than a 32-bit number
    session.commit()
    session.expire_all()
    assert session.scalars(select(User.telegram_id)).one() == 9_876_543_210_123


def test_a_user_has_at_most_one_profile(session):
    user = make_user()
    session.add(user)
    session.flush()
    session.add_all([UserProfile(user_id=user.id), UserProfile(user_id=user.id)])
    with pytest.raises(IntegrityError):
        session.flush()


def test_a_user_can_mark_a_company_only_once(session):
    user, company = make_user(), make_company()
    session.add_all([user, company])
    session.flush()
    session.add(UserCompanyPref(user_id=user.id, company_id=company.id, kind="favourite"))
    session.flush()
    session.add(UserCompanyPref(user_id=user.id, company_id=company.id, kind="excluded"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_user_deletes_their_profile_and_company_choices_only(session):
    user, other, company = make_user(1), make_user(2), make_company()
    user.profile = UserProfile(countries=["GB"])
    session.add_all([user, other, company])
    session.flush()
    session.add_all(
        [
            UserCompanyPref(user_id=user.id, company_id=company.id, kind="favourite", never_miss=True),
            UserCompanyPref(user_id=other.id, company_id=company.id, kind="excluded"),
        ]
    )
    session.commit()

    session.delete(user)
    session.commit()

    assert session.scalars(select(UserProfile)).all() == []
    [remaining] = session.scalars(select(UserCompanyPref)).all()
    assert remaining.user_id == other.id
    assert session.scalars(select(Company)).one().name == "Example Ltd"  # companies are shared: kept


def test_deleting_a_company_removes_users_choices_about_it(session):
    """Done by the database itself (ON DELETE CASCADE), e.g. if a company row is ever removed."""
    user, company = make_user(), make_company()
    session.add_all([user, company])
    session.flush()
    session.add(UserCompanyPref(user_id=user.id, company_id=company.id, kind="excluded"))
    session.commit()

    session.execute(Company.__table__.delete())
    session.commit()
    assert session.scalars(select(UserCompanyPref)).all() == []


def test_a_company_added_by_a_user_survives_the_user_being_deleted(session):
    user = make_user()
    session.add(user)
    session.flush()
    company = make_company(added_by_user_id=user.id)
    session.add(company)
    session.commit()

    session.execute(User.__table__.delete())  # straight in the database: tests ON DELETE SET NULL
    session.commit()
    session.expire_all()
    assert session.scalars(select(Company)).one().added_by_user_id is None


def test_added_by_must_be_a_real_user(session):
    session.add(make_company(added_by_user_id=999))
    with pytest.raises(IntegrityError):
        session.commit()
