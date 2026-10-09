"""Tests for building matching's facts from database rows (joborbit/matching/facts.py)."""

from datetime import datetime

from matching_samples import ACME, make_job, make_user

from joborbit.db.models import Company, Job, JobAnalysis, UserCompanyPref, UserProfile
from joborbit.matching.facts import JobFacts, UserFacts


def test_job_facts_are_built_from_the_job_its_analysis_and_its_company():
    company = Company(id=ACME, name="Acme", slug="acme", industry="ai", countries=["GB", "IN"])
    job = Job(
        id=3, company_id=ACME, title="AI Engineer", location_raw="Remote",
        posted_at=datetime(2026, 9, 1), became_new_at=datetime(2026, 10, 9, 9, 0),
    )
    analysis = JobAnalysis(
        job_id=3, countries=[], cities=[], seniority="entry", is_graduate_scheme=False, experience_years=None,
        experience_mandatory=True, required_languages=["en"], role_families=["ai_engineer"],
        matched_custom_roles=["Insights Analyst"], skills=["Python", "LLMs"], min_degree="bachelor",
    )

    assert JobFacts.from_rows(job, analysis, company) == make_job(
        job_id=3, title="AI Engineer", company_industry="ai", company_countries=frozenset({"GB", "IN"}),
        location_vague=True, countries=frozenset(), cities=(), seniority="entry", experience_mandatory=True,
        required_languages=frozenset({"en"}), role_families=("ai_engineer",),
        matched_custom_roles=frozenset({"insights analyst"}), skills=("Python", "LLMs"), min_degree="bachelor",
        posted_at=datetime(2026, 9, 1),
    )


def test_a_named_location_is_not_vague():
    company = Company(id=ACME, name="Acme", industry="fintech", countries=["GB"])
    job = Job(id=1, company_id=ACME, title="Graduate Data Scientist", location_raw="London, UK",
              became_new_at=datetime(2026, 10, 9, 9, 0))
    analysis = JobAnalysis(
        job_id=1, countries=["GB"], cities=["London"], seniority="graduate", is_graduate_scheme=False,
        experience_years=None, experience_mandatory=False, required_languages=[], role_families=["data_scientist"],
        matched_custom_roles=[], skills=["Python", "SQL"], min_degree=None,
    )
    assert JobFacts.from_rows(job, analysis, company) == make_job()


def test_user_facts_are_built_from_the_profile_and_company_choices():
    profile = UserProfile(
        user_id=1, roles=["data_scientist", "data_analyst"], custom_roles=["Insights Analyst"], countries=["GB"],
        languages=["en", "es"], preferred_industries=["ai"], excluded_industries=["gambling"], skills=["Python"],
        highest_degree="master", include_internships=False, transfer_boost=False, weights={"skills": 30},
    )
    prefs = [
        UserCompanyPref(company_id=ACME, kind="favourite", never_miss=True),
        UserCompanyPref(company_id=8, kind="favourite", never_miss=False),
        UserCompanyPref(company_id=9, kind="excluded", never_miss=False),
    ]

    assert UserFacts.from_rows(profile, prefs) == make_user(
        custom_roles=("Insights Analyst",), preferred_industries=frozenset({"ai"}),
        excluded_industries=frozenset({"gambling"}), transfer_boost=False, weights={"skills": 30},
        favourite_company_ids=frozenset({ACME, 8}), never_miss_company_ids=frozenset({ACME}),
        excluded_company_ids=frozenset({9}),
    )


def test_empty_personal_weights_mean_the_defaults():
    profile = UserProfile(
        user_id=1, roles=["data_scientist", "data_analyst"], custom_roles=[], countries=["GB"],
        languages=["en", "es"], preferred_industries=[], excluded_industries=[], skills=["Python"],
        highest_degree="master", include_internships=False, transfer_boost=True, weights={},
    )
    assert UserFacts.from_rows(profile, []).weights is None
