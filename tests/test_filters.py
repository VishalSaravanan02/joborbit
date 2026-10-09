"""Tests for the hard filters (joborbit/matching/filters.py).

Every test starts from a job and a user that match (tests/matching_samples.py), then changes
one thing, so each test fails only for the reason it is about. Rules are tested in both directions.
"""

import pytest
from matching_samples import ACME, make_job, make_user

from joborbit.matching.facts import JobFacts, UserFacts
from joborbit.matching.filters import REQUIRED_YEARS_LIMIT, check_job

def reason(job: JobFacts | None = None, user: UserFacts | None = None) -> str:
    return check_job(job or make_job(), user or make_user()).reason


def test_a_job_that_fits_passes_with_no_flags():
    verdict = check_job(make_job(), make_user())
    assert verdict.passed and verdict.reason == "passed"
    assert verdict.country_unclear is False and verdict.never_miss is False


# --- 1. Country ------------------------------------------------------------------------------


def test_a_job_in_another_country_is_dropped():
    assert reason(make_job(countries=frozenset({"IN"}))) == "country: IN (not yours)"


def test_a_job_in_several_countries_passes_if_one_is_yours():
    assert reason(make_job(countries=frozenset({"IN", "GB"}))) == "passed"


def test_any_of_the_users_countries_counts():
    user = make_user(countries=("IN", "GB"))
    assert reason(make_job(countries=frozenset({"GB"})), user) == "passed"


def test_no_country_with_a_vague_location_at_a_company_in_your_country_is_unclear():
    job = make_job(countries=frozenset(), location_vague=True)
    verdict = check_job(job, make_user())
    assert verdict.passed and verdict.country_unclear is True


def test_no_country_with_a_specific_location_is_dropped():
    """The location named a place and the LLM found none of ours there: trust it."""
    job = make_job(countries=frozenset(), location_vague=False)
    assert reason(job) == "country: none of ours"


def test_no_country_at_a_company_outside_your_countries_is_dropped():
    job = make_job(countries=frozenset(), location_vague=True, company_countries=frozenset({"IN"}))
    assert reason(job) == "country: not stated, and the company isn't in your countries"


def test_a_job_with_a_stated_country_is_never_unclear():
    verdict = check_job(make_job(location_vague=True), make_user())
    assert verdict.passed and verdict.country_unclear is False


# --- 2. Level ---------------------------------------------------------------------------------


def test_a_senior_job_is_dropped():
    assert reason(make_job(seniority="senior")) == "level: senior"


@pytest.mark.parametrize("seniority", ["graduate", "entry", "mid", None])
def test_other_levels_pass(seniority):
    assert reason(make_job(seniority=seniority)) == "passed"


def test_required_experience_with_years_is_dropped():
    job = make_job(experience_mandatory=True, experience_years=REQUIRED_YEARS_LIMIT)
    assert reason(job) == f"level: {REQUIRED_YEARS_LIMIT}+ years required"


def test_required_experience_below_the_limit_passes():
    job = make_job(experience_mandatory=True, experience_years=REQUIRED_YEARS_LIMIT - 1)
    assert reason(job) == "passed"


def test_required_experience_without_a_number_passes():
    """Scoring lowers these instead of hiding them."""
    assert reason(make_job(experience_mandatory=True, experience_years=None)) == "passed"


def test_preferred_experience_never_drops_a_job():
    assert reason(make_job(experience_mandatory=False, experience_years=5)) == "passed"


# --- 3. Internship ----------------------------------------------------------------------------


def test_an_internship_is_dropped_for_a_user_who_doesnt_want_them():
    assert reason(make_job(seniority="intern")) == "internship"


def test_an_internship_passes_for_a_user_who_wants_them():
    assert reason(make_job(seniority="intern"), make_user(include_internships=True)) == "passed"


# --- 4. Languages -----------------------------------------------------------------------------


def test_a_required_language_the_user_speaks_passes():
    assert reason(make_job(required_languages=frozenset({"en", "es"}))) == "passed"


def test_a_required_language_the_user_doesnt_speak_is_dropped():
    job = make_job(required_languages=frozenset({"en", "fr", "de"}))
    assert reason(job) == "language: de, fr required"


def test_a_required_language_we_have_no_code_for_is_dropped():
    assert reason(make_job(required_languages=frozenset({"other"}))) == "language: other required"


# --- 5. Excluded company or industry --------------------------------------------------------------


def test_an_excluded_company_is_dropped():
    assert reason(user=make_user(excluded_company_ids=frozenset({ACME}))) == "excluded company"


def test_excluding_another_company_changes_nothing():
    assert reason(user=make_user(excluded_company_ids=frozenset({ACME + 1}))) == "passed"


def test_an_excluded_industry_is_dropped():
    assert reason(user=make_user(excluded_industries=frozenset({"fintech"}))) == "excluded industry: fintech"


def test_excluding_another_industry_changes_nothing():
    assert reason(user=make_user(excluded_industries=frozenset({"gambling"}))) == "passed"


def test_a_company_with_no_industry_is_never_excluded_by_industry():
    user = make_user(excluded_industries=frozenset({"gambling"}))
    assert reason(make_job(company_industry=None), user) == "passed"


# --- 6. Degree -----------------------------------------------------------------------------------


def test_a_phd_job_is_dropped_for_a_user_with_a_lower_degree():
    assert reason(make_job(min_degree="phd")) == "degree: PhD required"


@pytest.mark.parametrize("degree", ["phd", None])
def test_a_phd_job_passes_for_a_phd_or_a_user_who_didnt_say(degree):
    assert reason(make_job(min_degree="phd"), make_user(highest_degree=degree)) == "passed"


@pytest.mark.parametrize("required", ["bachelor", "master", "none"])
def test_other_degree_requirements_never_drop_a_job(required):
    assert reason(make_job(min_degree=required), make_user(highest_degree="bachelor")) == "passed"


# --- 7. Role ---------------------------------------------------------------------------------------


def test_a_job_with_none_of_the_users_roles_is_dropped():
    assert reason(make_job(role_families=("data_engineer",))) == "role: not one of yours"


def test_a_job_with_no_role_at_all_is_dropped():
    assert reason(make_job(role_families=())) == "role: not one of yours"


def test_any_of_the_jobs_roles_counts():
    assert reason(make_job(role_families=("data_engineer", "data_analyst"))) == "passed"


def test_a_matched_custom_role_counts_whatever_the_capitals():
    user = make_user(custom_roles=("Insights Analyst",))
    job = make_job(role_families=(), matched_custom_roles=frozenset({"insights analyst"}))
    assert reason(job, user) == "passed"


def test_another_users_custom_role_doesnt_count():
    user = make_user(custom_roles=("Insights Analyst",))
    job = make_job(role_families=(), matched_custom_roles=frozenset({"pricing analyst"}))
    assert reason(job, user) == "role: not one of yours"


def test_a_never_miss_favourite_skips_the_role_check():
    user = make_user(never_miss_company_ids=frozenset({ACME}))
    verdict = check_job(make_job(role_families=("data_engineer",)), user)
    assert verdict.passed and verdict.never_miss is True


def test_a_never_miss_favourite_is_flagged_even_when_the_role_fits():
    verdict = check_job(make_job(), make_user(never_miss_company_ids=frozenset({ACME})))
    assert verdict.passed and verdict.never_miss is True


def test_never_miss_at_another_company_changes_nothing():
    user = make_user(never_miss_company_ids=frozenset({ACME + 1}))
    assert reason(make_job(role_families=("data_engineer",)), user) == "role: not one of yours"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"countries": frozenset({"IN"})}, "country: IN (not yours)"),
        ({"seniority": "senior"}, "level: senior"),
        ({"required_languages": frozenset({"fr"})}, "language: fr required"),
    ],
)
def test_never_miss_still_needs_the_right_country_level_and_language(change, expected):
    user = make_user(never_miss_company_ids=frozenset({ACME}))
    assert reason(make_job(role_families=(), **change), user) == expected


# --- Order -------------------------------------------------------------------------------------------


def test_the_first_rule_broken_is_the_one_reported():
    job = make_job(countries=frozenset({"IN"}), seniority="senior", role_families=())
    assert reason(job) == "country: IN (not yours)"
