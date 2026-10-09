"""What matching knows about one job and one user, gathered once from the database rows.

The hard filters (filters.py), the scoring (scoring.py) and the routing (router.py) all work
on these two small objects instead of on database rows. That keeps them pure: easy to test,
and reusable by the dry run on any job. from_rows builds each one; nothing here queries the
database.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from joborbit.db.models import Company, Job, JobAnalysis, UserCompanyPref, UserProfile
from joborbit.pipeline.location import parse_location


@dataclass(frozen=True)
class JobFacts:
    """One analysed job, with the bits of its company that matching needs."""

    job_id: int
    title: str
    company_id: int
    company_name: str
    company_industry: str | None
    company_countries: frozenset[str]
    location_vague: bool  # the location text named no place ("Remote", "EMEA", blank)
    countries: frozenset[str]  # from the analysis: where the job can be done
    cities: tuple[str, ...]
    seniority: str | None
    is_graduate_scheme: bool
    experience_years: int | None
    experience_mandatory: bool
    required_languages: frozenset[str]
    role_families: tuple[str, ...]  # best first
    matched_custom_roles: frozenset[str]  # lower case, for comparing
    skills: tuple[str, ...]  # as written in the analysis
    min_degree: str | None
    posted_at: datetime | None  # when the company first published it, if the ATS says
    became_new_at: datetime  # when it last became new to us

    @classmethod
    def from_rows(cls, job: Job, analysis: JobAnalysis, company: Company) -> "JobFacts":
        return cls(
            job_id=job.id,
            title=job.title,
            company_id=company.id,
            company_name=company.name,
            company_industry=company.industry,
            company_countries=frozenset(company.countries),
            location_vague=parse_location(job.location_raw).ambiguous,
            countries=frozenset(analysis.countries),
            cities=tuple(analysis.cities),
            seniority=analysis.seniority,
            is_graduate_scheme=analysis.is_graduate_scheme,
            experience_years=analysis.experience_years,
            experience_mandatory=analysis.experience_mandatory,
            required_languages=frozenset(analysis.required_languages),
            role_families=tuple(analysis.role_families),
            matched_custom_roles=frozenset(name.lower() for name in analysis.matched_custom_roles),
            skills=tuple(analysis.skills),
            min_degree=analysis.min_degree,
            posted_at=job.posted_at,
            became_new_at=job.became_new_at,
        )


@dataclass(frozen=True)
class UserFacts:
    """One user's profile and company choices."""

    user_id: int
    roles: frozenset[str]  # default role slugs
    custom_roles: tuple[str, ...]  # as the user wrote them
    countries: tuple[str, ...]  # ranked: first = most wanted
    languages: frozenset[str]
    preferred_industries: frozenset[str]
    excluded_industries: frozenset[str]
    skills: tuple[str, ...]  # as the user wrote them
    highest_degree: str | None
    include_internships: bool
    transfer_boost: bool
    alert_style: str  # "fewer", "balanced" or "more"
    weights: Mapping[str, float] | None  # personal scoring weights; None = the defaults
    favourite_company_ids: frozenset[int]
    never_miss_company_ids: frozenset[int]  # favourites with "never miss" on
    excluded_company_ids: frozenset[int]

    @classmethod
    def from_rows(cls, profile: UserProfile, company_prefs: Iterable[UserCompanyPref]) -> "UserFacts":
        prefs = list(company_prefs)
        favourites = [pref for pref in prefs if pref.kind == "favourite"]
        return cls(
            user_id=profile.user_id,
            roles=frozenset(profile.roles),
            custom_roles=tuple(profile.custom_roles),
            countries=tuple(profile.countries),
            languages=frozenset(profile.languages),
            preferred_industries=frozenset(profile.preferred_industries),
            excluded_industries=frozenset(profile.excluded_industries),
            skills=tuple(profile.skills),
            highest_degree=profile.highest_degree,
            include_internships=profile.include_internships,
            transfer_boost=profile.transfer_boost,
            alert_style=profile.alert_style,
            weights=dict(profile.weights) if profile.weights else None,
            favourite_company_ids=frozenset(pref.company_id for pref in favourites),
            never_miss_company_ids=frozenset(pref.company_id for pref in favourites if pref.never_miss),
            excluded_company_ids=frozenset(pref.company_id for pref in prefs if pref.kind == "excluded"),
        )
