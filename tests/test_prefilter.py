"""Tests for the pre-filter rules (joborbit/pipeline/prefilter.py).

Most use the real config files (roles.yaml, seniority.yaml, countries.yaml), so they also
check that our keyword lists behave as intended on realistic titles.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.config import load_roles
from joborbit.db.models import Base, Company, Job, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.fetchers.ashby import AshbyFetcher
from joborbit.fetchers.greenhouse import GreenhouseFetcher
from joborbit.fetchers.lever import LeverFetcher
from joborbit.pipeline.prefilter import (
    MAX_REASON_CHARS,
    PrefilterRules,
    RoleMatch,
    check_job,
    contains,
    fuzzy_contains,
    reason_kind,
    rules_for_active_users,
    run_prefilter,
    words,
)

ALL_ROLES = {role.slug for role in load_roles()}
RULES = PrefilterRules.build(ALL_ROLES, set(), allow_internships=False)
LONDON = "London, United Kingdom"


def reason(title: str, location: str | None = LONDON, rules: PrefilterRules = RULES) -> str:
    return check_job(title, location, rules).reason


# --- Words and matching ------------------------------------------------------------------------


def test_words_drop_capitals_and_punctuation():
    assert words("Sr. Data-Scientist (UK) / AI&ML") == ("sr", "data", "scientist", "uk", "ai", "ml")
    assert words(None) == () and words("") == ()


def test_exact_matching_needs_whole_words_in_order():
    title = words("Graduate Data Scientist")
    assert contains(title, ("data", "scientist"))
    assert not contains(title, ("scientist", "data"))
    assert not contains(words("Leading Data Teams"), ("lead",))


@pytest.mark.parametrize(
    ("title", "keyword", "expected"),
    [
        ("Data Scientst", "data scientist", True),  # typo
        ("Data Analysts", "data analyst", True),  # plural
        ("Graduate Program", "graduate programme", True),  # US spelling
        ("QA Engineer", "ai engineer", False),  # would pass if whole phrases were compared
        ("UI Developer", "ai developer", False),
        ("HR Analyst", "bi analyst", False),
        ("Data Analytics", "data analyst", False),  # different words, not a typo
        ("ML Engineering", "ml engineer", False),  # "engineering" scores 84: another word form, not a typo
    ],
)
def test_fuzzy_matching_compares_word_by_word(title, keyword, expected):
    assert fuzzy_contains(words(title), words(keyword)) is expected


# --- Titles that should pass, with the roles they match -----------------------------------------


@pytest.mark.parametrize(
    ("title", "roles"),
    [
        ("Graduate Data Scientist", "roles: data_scientist"),
        ("Junior Data Analyst - Insights", "roles: data_analyst"),
        ("Machine Learning Engineer, Early Careers", "roles: ml_engineer"),
        ("AI/ML Engineer", "roles: ml_engineer"),
        ("Associate Business Intelligence Analyst", "roles: bi_analyst"),
        ("Credit Risk Analyst - Graduate Programme 2026", "roles: credit_risk_analyst, tech_graduate_scheme"),
        ("Technology Graduate Scheme", "roles: tech_graduate_scheme"),
        ("Analytics Engineer (dbt)", "roles: analytics_engineer"),
        ("Data Engineer II", "roles: data_engineer"),  # level II is left for the LLM
        ("Research Scientist, Machine Learning", "roles: ml_engineer, applied_scientist"),
        ("Quantitative Researcher - New Grad", "roles: quant_analyst"),
        ("Data Scientst", "roles: data_scientist (fuzzy)"),
    ],
)
def test_entry_level_data_titles_pass(title, roles):
    decision = check_job(title, LONDON, RULES)
    assert decision.passed
    assert decision.reason == roles


def test_exact_and_fuzzy_matches_are_told_apart():
    assert check_job("Data Scientist", LONDON, RULES).roles == (RoleMatch("data_scientist", exact=True),)
    assert check_job("Data Scientst", LONDON, RULES).roles == (RoleMatch("data_scientist", exact=False),)


# --- Titles that should be rejected, with the reason ----------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Data Scientist", "senior title: senior"),
        ("Sr. Data Analyst", "senior title: sr"),
        ("Lead Machine Learning Engineer", "senior title: lead"),
        ("Staff Data Engineer", "senior title: staff"),
        ("Head of Data Science", "senior title: head of"),
        ("Data Science Manager", "senior title: manager"),
        ("Data Scientist III", "senior title: iii"),
        ("Data Science Intern", "internship: intern"),
        ("Summer Analyst - Quantitative Analyst Programme", "internship: summer analyst"),
        ("Software Engineer", "no matching role"),
        ("QA Engineer", "no matching role"),
        ("Sales Graduate Programme", "no matching role"),  # excluded from the graduate-scheme role
        ("Research Scientist, Medicinal Chemistry", "no matching role"),  # lab science, not ML
    ],
)
def test_wrong_titles_are_rejected_with_a_reason(title, expected):
    decision = check_job(title, LONDON, RULES)
    assert not decision.passed
    assert decision.reason == expected


def test_words_inside_other_words_do_not_drop_a_title():
    """'Leading', 'Staffing' and 'Internal' contain drop or internship words, but aren't them."""
    assert check_job("Data Analyst, Staffing and Internal Leading Indicators", LONDON, RULES).passed


# --- Countries --------------------------------------------------------------------------------------


@pytest.mark.parametrize("location", [LONDON, "Remote (UK)", "Remote", "EMEA", "", None, "London / New York"])
def test_uk_and_vague_locations_pass(location):
    assert check_job("Data Scientist", location, RULES).passed


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("Bengaluru, India", "country: IN (not switched on)"),
        ("Singapore", "country: SG (not switched on)"),
        ("New York, NY", "country: not one we cover"),
        ("Berlin, Germany", "country: not one we cover"),
    ],
)
def test_other_countries_are_rejected(location, expected):
    assert reason("Data Scientist", location) == expected


def test_the_country_is_checked_before_the_title():
    """A senior job abroad is reported as a country problem: one clear reason per job."""
    assert reason("Senior Data Scientist", "Berlin, Germany") == "country: not one we cover"


# --- Which roles are looked for -----------------------------------------------------------------


def test_only_roles_someone_has_are_looked_for():
    analysts_only = PrefilterRules.build({"data_analyst"}, set(), allow_internships=False)
    assert reason("Data Scientist", rules=analysts_only) == "no matching role"
    assert reason("Data Analyst", rules=analysts_only) == "roles: data_analyst"


def test_custom_roles_are_matched_by_their_name():
    custom = PrefilterRules.build(set(), {"Pricing Analyst"}, allow_internships=False)
    assert reason("Junior Pricing Analyst", rules=custom) == "roles: Pricing Analyst"
    assert reason("Pricing Analysts", rules=custom) == "roles: Pricing Analyst (fuzzy)"


def test_unknown_role_slugs_are_ignored():
    rules = PrefilterRules.build({"data_analyst", "retired_role"}, set(), allow_internships=False)
    assert [rule.role for rule in rules.roles] == ["data_analyst"]


def test_with_no_users_nothing_passes():
    nobody = PrefilterRules.build(set(), set(), allow_internships=False)
    assert reason("Data Scientist", rules=nobody) == "no active users"


def test_internships_pass_when_someone_wants_them():
    with_interns = PrefilterRules.build(ALL_ROLES, set(), allow_internships=True)
    assert check_job("Data Science Intern", LONDON, with_interns).passed
    assert reason("Senior Data Science Intern", rules=with_interns) == "senior title: senior"


# --- Real jobs from our saved replies -------------------------------------------------------------


def test_the_real_saved_jobs_get_the_right_verdicts():
    """None of the nine saved jobs is a data role; the non-UK ones are rejected for country first."""
    fixtures = Path(__file__).parent / "fixtures"
    jobs = [
        *GreenhouseFetcher().parse_jobs(json.loads((fixtures / "greenhouse.json").read_text())["jobs"], "monzo"),
        *LeverFetcher().parse_jobs(json.loads((fixtures / "lever.json").read_text()), "palantir"),
        *AshbyFetcher().parse_jobs(json.loads((fixtures / "ashby.json").read_text())["jobs"], "elevenlabs"),
    ]
    verdicts = [reason(job.title, job.location_raw) for job in jobs]
    assert verdicts == [
        "no matching role",  # Anaplan Support Analyst, Cardiff/London/Remote UK
        "no matching role",  # Android Engineer, UK
        "country: ES (not switched on)",  # Android Engineer, Barcelona
        "country: SG (not switched on)",  # Administrative Business Partner, Singapore
        "no matching role",  # Administrative Business Partner, London
        "country: not one we cover",  # Washington, D.C.
        "country: IN (not switched on)",  # Account Manager - India
        "country: not one we cover",  # United States / New York / San Francisco
        "country: not one we cover",  # San Francisco / United States / New York
    ]


def test_custom_roles_are_marked_as_custom():
    rules = PrefilterRules.build({"data_analyst"}, {"Pricing Analyst"}, allow_internships=False)
    assert [(rule.role, rule.custom) for rule in rules.roles] == [("data_analyst", False), ("Pricing Analyst", True)]


def test_reason_kinds_are_the_part_before_the_colon():
    assert reason_kind("senior title: lead") == "senior title"
    assert reason_kind("no matching role") == "no matching role"


# --- Running on the database ---------------------------------------------------------------------

START = datetime(2026, 10, 1, 9, 0)


@pytest.fixture
def session():
    """An empty in-memory database with one company."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(Company(name="Acme", slug="acme", size_category="startup"))
        s.commit()
        yield s


def add_user(session, telegram_id: int, roles: list[str], custom: list[str] = (), **user_fields) -> None:
    user = User(telegram_id=telegram_id, display_name="Someone", **user_fields)
    user.profile = UserProfile(roles=list(roles), custom_roles=list(custom), countries=["GB"])
    session.add(user)
    session.commit()


def add_job(session, title: str, location: str = LONDON, minutes: int = 0, **job_fields) -> Job:
    job = Job(
        company_id=1,
        ats_type="greenhouse",
        external_id=title + str(minutes),
        url="https://example.com",
        title=title,
        location_raw=location,
        first_seen_at=START + timedelta(minutes=minutes),
        **job_fields,
    )
    session.add(job)
    session.commit()
    return job


def test_pending_jobs_get_a_verdict_and_a_reason(session):
    add_user(session, 1, ["data_scientist"])
    good = add_job(session, "Graduate Data Scientist")
    senior = add_job(session, "Senior Data Scientist")
    abroad = add_job(session, "Data Scientist", "Berlin, Germany")

    run = run_prefilter(session)
    session.commit()

    assert (good.prefilter_status, good.prefilter_reason) == ("passed", "roles: data_scientist")
    assert (senior.prefilter_status, senior.prefilter_reason) == ("rejected", "senior title: senior")
    assert (abroad.prefilter_status, abroad.prefilter_reason) == ("rejected", "country: not one we cover")
    assert run.passed_job_ids == [good.id]
    assert run.rejected == {"senior title": 1, "country": 1}
    assert run.checked == 3


def test_skipped_decided_and_closed_jobs_are_left_alone(session):
    add_user(session, 1, ["data_scientist"])
    baseline = add_job(session, "Data Scientist", prefilter_status="skipped", prefilter_reason="baseline")
    decided = add_job(session, "Data Scientist", minutes=1, prefilter_status="rejected", prefilter_reason="old")
    closed = add_job(session, "Data Scientist", minutes=2, closed_at=START)

    run = run_prefilter(session)

    assert run.checked == 0
    assert (baseline.prefilter_status, baseline.prefilter_reason) == ("skipped", "baseline")
    assert (decided.prefilter_status, decided.prefilter_reason) == ("rejected", "old")
    assert closed.prefilter_status == "pending"


def test_newest_jobs_are_decided_first(session):
    add_user(session, 1, ["data_scientist"])
    older = add_job(session, "Data Scientist", minutes=0)
    newer = add_job(session, "Data Scientist", minutes=30)
    assert run_prefilter(session).passed_job_ids == [newer.id, older.id]


def test_every_active_users_roles_count_including_paused_users(session):
    add_user(session, 1, ["data_scientist"])
    add_user(session, 2, [], custom=["Pricing Analyst"], paused=True)
    add_user(session, 3, ["data_engineer"], is_active=False)  # inactive: their roles don't count

    rules = rules_for_active_users(session)

    assert sorted(rule.role for rule in rules.roles) == ["Pricing Analyst", "data_scientist"]


def test_internships_are_allowed_if_any_active_user_wants_them(session):
    add_user(session, 1, ["data_scientist"])
    assert rules_for_active_users(session).allow_internships is False
    session.scalars(select(UserProfile).where(UserProfile.user_id == 1)).one().include_internships = True
    add_user(session, 2, ["data_scientist"])  # added after: one user wanting them is enough, in any order
    assert rules_for_active_users(session).allow_internships is True


def test_with_no_users_every_pending_job_is_rejected(session):
    job = add_job(session, "Data Scientist")
    run_prefilter(session)
    assert (job.prefilter_status, job.prefilter_reason) == ("rejected", "no active users")


def test_very_long_reasons_are_shortened_to_fit(session):
    many = [f"Data Role {number}" for number in range(40)]
    add_user(session, 1, [], custom=many)
    job = add_job(session, " ".join(many))  # matches all 40 custom roles
    run_prefilter(session)
    session.commit()
    assert job.prefilter_status == "passed"
    assert len(job.prefilter_reason) == MAX_REASON_CHARS
