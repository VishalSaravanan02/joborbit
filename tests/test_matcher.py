"""Tests for matching jobs against users and saving the matches (joborbit/matching/matcher.py)."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.db.models import Base, Company, Job, JobAnalysis, Match, User, UserCompanyPref, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.matching.matcher import run_matching, waiting_job_ids

START = datetime(2026, 10, 9, 9, 0)


@pytest.fixture
def session():
    """Acme, a UK fintech, and Vishal, who wants data analyst jobs in the UK."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Company(name="Acme", slug="acme", size_category="startup", industry="fintech", countries=["GB"]))
        add_user(session, "Vishal", telegram_id=1)
        session.commit()
        yield session


def add_user(session, name: str, telegram_id: int, active: bool = True, paused: bool = False, **profile) -> User:
    user = User(telegram_id=telegram_id, display_name=name, is_active=active, paused=paused)
    data = {"roles": ["data_analyst"], "countries": ["GB"], "languages": ["en"], "skills": ["SQL"]}
    user.profile = UserProfile(**{**data, **profile})
    session.add(user)
    session.flush()
    return user


def add_job(session, title: str = "Graduate Data Analyst", minutes: int = 0, analysed: bool = True, **fields) -> Job:
    """An open job at Acme; analysed as a graduate data analyst job in London unless told otherwise."""
    analysis = fields.pop("analysis", {})
    job = Job(
        company_id=1, ats_type="greenhouse", external_id=f"{title}-{minutes}", url=f"https://e.com/{minutes}",
        title=title, location_raw="London, UK", first_seen_at=START + timedelta(minutes=minutes),
        became_new_at=START + timedelta(minutes=minutes), prefilter_status="passed",
        analysis_status="done" if analysed else "pending", **fields,
    )
    session.add(job)
    session.flush()
    if analysed:
        data = {
            "countries": ["GB"], "cities": ["London"], "seniority": "graduate", "is_graduate_scheme": False,
            "experience_years": None, "experience_mandatory": False, "required_languages": [],
            "role_families": ["data_analyst"], "matched_custom_roles": [], "skills": ["SQL", "Python"],
            "min_degree": None, "summary": "A graduate data analyst job.", "model": "gpt-6-luna",
            "prompt_version": "2", "input_tokens": 3000, "output_tokens": 500,
        }
        session.add(JobAnalysis(job_id=job.id, **{**data, **analysis}))
        session.flush()
    return job


def matches(session) -> list[Match]:
    return list(session.scalars(select(Match).order_by(Match.user_id, Match.job_id)))


# --- Which jobs wait for matching -------------------------------------------------------------------


def test_only_open_analysed_unmatched_new_jobs_wait_newest_first(session):
    older = add_job(session, "Data Analyst A", minutes=1)
    newer = add_job(session, "Data Analyst B", minutes=2)
    add_job(session, "Not analysed", minutes=3, analysed=False)
    add_job(session, "Already matched", minutes=4, matched_at=START)
    add_job(session, "Closed", minutes=5, closed_at=START)
    add_job(session, "Baseline", minutes=6, is_baseline=True)
    add_job(session, "Failed", minutes=7, analysed=False).analysis_status = "failed"
    assert waiting_job_ids(session) == [newer.id, older.id]


def test_jobs_wait_by_when_they_last_became_new():
    """A job back after a long gap became new again, so it counts as newer than its first sighting."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Company(name="Acme", slug="acme", size_category="startup", countries=["GB"]))
        back = add_job(session, "Came back", minutes=1)
        fresh = add_job(session, "Fresh", minutes=2)
        back.became_new_at = START + timedelta(days=9)
        session.flush()
        assert waiting_job_ids(session) == [back.id, fresh.id]


# --- Making matches ------------------------------------------------------------------------------------


def test_a_fitting_job_is_saved_as_a_scored_routed_match(session):
    job = add_job(session)
    run = run_matching(session)

    [match] = matches(session)
    assert (match.job_id, match.score, match.tier) == (job.id, 80, "instant")  # 30 + 25 + 10 (1 of 2 skills) + 15
    assert match.components == {"role_fit": 30, "entry_fit": 25, "skills": 10, "country": 15, "transfer": 0}
    assert match.reasons == ["Role: Data Analyst", "Graduate role", "London"]
    assert match.created_at == match.scored_at and job.matched_at == match.created_at

    assert run.jobs == 1 and not run.dropped
    [new] = run.matches
    assert (new.match_id, new.user_name, new.company, new.title, new.url) == (
        match.id, "Vishal", "Acme", "Graduate Data Analyst", "https://e.com/0"
    )
    assert (new.score, new.tier, new.notes) == (80, "instant", ())


def test_notes_are_saved_after_the_reasons(session):
    add_job(session, analysis={"experience_mandatory": True, "seniority": "entry"})
    run_matching(session)
    [match] = matches(session)
    assert match.reasons[-1] == "Experience required (no number given)"
    assert match.reasons[:-1] == ["Role: Data Analyst", "London", "1 of your skills: SQL"]


def test_a_job_that_breaks_a_hard_filter_makes_no_match_but_is_still_done(session):
    job = add_job(session, analysis={"seniority": "senior"})
    run = run_matching(session)
    assert matches(session) == []
    assert run.dropped == {"level": 1} and run.jobs == 1
    assert job.matched_at is not None


def test_each_user_is_matched_on_their_own_profile(session):
    add_user(session, "Ana", telegram_id=2, roles=["data_scientist"])
    add_user(session, "Ben", telegram_id=3, alert_style="fewer")
    add_job(session)

    run = run_matching(session)

    assert [(match.user_id, match.tier) for match in matches(session)] == [(1, "instant"), (3, "instant")]
    assert run.dropped == {"role": 1}
    assert sorted(new.user_name for new in run.matches) == ["Ben", "Vishal"]


def test_each_users_alert_style_decides_the_tier(session):
    add_user(session, "Ben", telegram_id=2, alert_style="fewer")
    add_job(session, analysis={"seniority": "entry"})  # 30 + 22.5 + 10 + 15 = 77.5 -> 78
    run_matching(session)
    # Vishal (balanced) gets instant alerts from 75; Ben (fewer) only from 80.
    assert [(match.score, match.tier) for match in matches(session)] == [(78, "instant"), (78, "digest")]


def test_company_choices_count(session):
    vishal = session.get_one(User, 1)
    session.add(UserCompanyPref(user_id=vishal.id, company_id=1, kind="favourite", never_miss=True))
    add_job(session, title="Software Engineer", analysis={"role_families": ["data_engineer"]})
    run_matching(session)
    [match] = matches(session)
    assert match.tier == "instant" and "Favourite company" in match.reasons


def test_an_excluded_company_is_never_matched(session):
    session.add(UserCompanyPref(user_id=1, company_id=1, kind="excluded"))
    add_job(session)
    assert run_matching(session).dropped == {"excluded company": 1}
    assert matches(session) == []


def test_paused_users_are_matched_but_inactive_ones_and_users_without_a_profile_are_not(session):
    add_user(session, "Paused", telegram_id=2, paused=True)
    add_user(session, "Gone", telegram_id=3, active=False)
    session.add(User(telegram_id=4, display_name="No profile"))
    add_job(session)
    run_matching(session)
    assert [match.user_id for match in matches(session)] == [1, 2]


def test_with_no_users_jobs_keep_waiting(session):
    session.get_one(User, 1).is_active = False
    job = add_job(session)
    run = run_matching(session)
    assert run.jobs == 0 and job.matched_at is None


# --- Matching again -----------------------------------------------------------------------------------------


def test_a_job_is_matched_only_once(session):
    add_job(session)
    run_matching(session)
    again = run_matching(session)
    assert again.jobs == 0 and again.matches == [] and len(matches(session)) == 1


def test_a_job_matched_again_updates_its_match(session):
    """A job back after a long gap: ingest empties matched_at, and the job may now score differently."""
    job = add_job(session)
    run_matching(session)
    [match] = matches(session)
    first_time = match.created_at

    job.matched_at = None
    job.analysis.seniority = "mid"
    session.flush()
    run = run_matching(session)

    assert matches(session) == [match]  # the same row, not a second one
    assert (match.score, match.tier) == (61, "digest")  # 30 + 6.25 + 10 + 15
    assert match.created_at == first_time and match.scored_at >= first_time
    assert [new.match_id for new in run.matches] == [match.id]


def test_matching_never_commits(session):
    add_job(session)
    session.commit()
    run_matching(session)
    session.rollback()
    assert matches(session) == []
    assert session.scalars(select(Job.matched_at)).one() is None
