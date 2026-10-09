"""A sample job and user for the matching tests: they fit each other, so every rule passes.

Each test changes only what it is about, e.g. make_job(seniority="senior"), so it can fail
only for that reason. The job is a graduate data science role in London at a UK fintech
company; the user wants data roles in the UK and speaks English and Spanish.
"""

from dataclasses import replace
from datetime import datetime

from joborbit.matching.facts import JobFacts, UserFacts

ACME = 7  # the company's id


def make_job(**changes) -> JobFacts:
    """A graduate data science job in London at a UK company: matches the default user."""
    job = JobFacts(
        job_id=1,
        title="Graduate Data Scientist",
        company_id=ACME,
        company_name="Acme",
        company_industry="fintech",
        company_countries=frozenset({"GB"}),
        location_vague=False,
        countries=frozenset({"GB"}),
        cities=("London",),
        seniority="graduate",
        is_graduate_scheme=False,
        experience_years=None,
        experience_mandatory=False,
        required_languages=frozenset(),
        role_families=("data_scientist",),
        matched_custom_roles=frozenset(),
        skills=("Python", "SQL"),
        min_degree=None,
        posted_at=None,
        became_new_at=datetime(2026, 10, 9, 9, 0),
    )
    return replace(job, **changes)


def make_user(**changes) -> UserFacts:
    user = UserFacts(
        user_id=1,
        roles=frozenset({"data_scientist", "data_analyst"}),
        custom_roles=(),
        countries=("GB",),
        languages=frozenset({"en", "es"}),
        preferred_industries=frozenset(),
        excluded_industries=frozenset(),
        skills=("Python",),
        highest_degree="master",
        include_internships=False,
        transfer_boost=True,
        weights=None,
        favourite_company_ids=frozenset(),
        never_miss_company_ids=frozenset(),
        excluded_company_ids=frozenset(),
    )
    return replace(user, **changes)
