"""The hard filters: the rules that decide whether a job can be shown to one user at all.

A job reaches matching only after the LLM has analysed it. For each user, check_job runs these
rules in order (cheapest and clearest first) and stops at the first one the job breaks:

1. Country: one of the job's countries is one of the user's. A job whose posting names no
   country can still pass as "country unclear" (see below).
2. Level: not senior, and no required experience of REQUIRED_YEARS_LIMIT years or more.
   Required experience without a number passes (scoring lowers it instead).
3. Internship: dropped unless the user wants internships.
4. Languages: every language the job requires is one the user speaks.
5. Excluded: the company, or its industry, is one the user excluded.
6. Degree: a PhD is required and the user has a lower degree.
7. Role: the LLM gave the job one of the user's roles, or matched one of their custom roles.
   A favourite company with "never miss" on skips this rule: every job there that passes
   rules 1 to 6 is the user's.

Country unclear. The LLM gives an empty country list both when the posting doesn't say where
the job is and when it says the job is somewhere that isn't ours (US only, say). The two can't
be told apart from the analysis alone, so an empty list passes only when the location text was
vague too ("Remote", or blank) and the company works in one of the user's countries. Scoring
values such a job lower, and the alert asks the user to check the location.

Which jobs are matched at all (open, analysed, not baseline) is the matcher's choice, not a
rule here: these rules only compare one job with one user, so the dry run can reuse them on
any job. Nothing here touches the database.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from joborbit.db.models import Company, Job, JobAnalysis, UserCompanyPref, UserProfile
from joborbit.pipeline.location import parse_location

REQUIRED_YEARS_LIMIT = 1  # required experience of this many years or more drops the job


@dataclass(frozen=True)
class JobFacts:
    """What the rules need to know about one analysed job."""

    job_id: int
    company_id: int
    company_industry: str | None
    company_countries: frozenset[str]
    location_vague: bool  # the location text named no place ("Remote", "EMEA", blank)
    countries: frozenset[str]  # from the analysis: where the job can be done
    seniority: str | None
    experience_years: int | None
    experience_mandatory: bool
    required_languages: frozenset[str]
    role_families: tuple[str, ...]
    matched_custom_roles: frozenset[str]  # lower case, for comparing
    min_degree: str | None

    @classmethod
    def from_rows(cls, job: Job, analysis: JobAnalysis, company: Company) -> "JobFacts":
        return cls(
            job_id=job.id,
            company_id=company.id,
            company_industry=company.industry,
            company_countries=frozenset(company.countries),
            location_vague=parse_location(job.location_raw).ambiguous,
            countries=frozenset(analysis.countries),
            seniority=analysis.seniority,
            experience_years=analysis.experience_years,
            experience_mandatory=analysis.experience_mandatory,
            required_languages=frozenset(analysis.required_languages),
            role_families=tuple(analysis.role_families),
            matched_custom_roles=frozenset(name.lower() for name in analysis.matched_custom_roles),
            min_degree=analysis.min_degree,
        )


@dataclass(frozen=True)
class UserFacts:
    """What the rules need to know about one user."""

    user_id: int
    roles: frozenset[str]  # default role slugs
    custom_roles: frozenset[str]  # lower case, for comparing
    countries: tuple[str, ...]  # ranked: first = most wanted
    languages: frozenset[str]
    excluded_industries: frozenset[str]
    highest_degree: str | None
    include_internships: bool
    excluded_company_ids: frozenset[int]
    never_miss_company_ids: frozenset[int]

    @classmethod
    def from_rows(cls, profile: UserProfile, company_prefs: Iterable[UserCompanyPref]) -> "UserFacts":
        prefs = list(company_prefs)
        return cls(
            user_id=profile.user_id,
            roles=frozenset(profile.roles),
            custom_roles=frozenset(name.lower() for name in profile.custom_roles),
            countries=tuple(profile.countries),
            languages=frozenset(profile.languages),
            excluded_industries=frozenset(profile.excluded_industries),
            highest_degree=profile.highest_degree,
            include_internships=profile.include_internships,
            excluded_company_ids=frozenset(pref.company_id for pref in prefs if pref.kind == "excluded"),
            never_miss_company_ids=frozenset(
                pref.company_id for pref in prefs if pref.kind == "favourite" and pref.never_miss
            ),
        )


@dataclass(frozen=True)
class Verdict:
    """The hard filters' answer for one job and one user."""

    passed: bool
    reason: str  # the rule it broke, in plain words; "passed" if none
    country_unclear: bool = False  # passed only as "country unclear": scored lower, flagged in the alert
    never_miss: bool = False  # a "never miss" favourite: always sent at once


def _dropped(reason: str) -> Verdict:
    return Verdict(passed=False, reason=reason)


def check_job(job: JobFacts, user: UserFacts) -> Verdict:
    """Run every hard filter on one job for one user, stopping at the first one it breaks."""
    # 1. Country.
    country_unclear = False
    if not job.countries & set(user.countries):
        if job.countries:
            return _dropped(f"country: {', '.join(sorted(job.countries))} (not yours)")
        if not job.location_vague:
            return _dropped("country: none of ours")
        if not job.company_countries & set(user.countries):
            return _dropped("country: not stated, and the company isn't in your countries")
        country_unclear = True

    # 2. Level.
    if job.seniority == "senior":
        return _dropped("level: senior")
    if job.experience_mandatory and (job.experience_years or 0) >= REQUIRED_YEARS_LIMIT:
        return _dropped(f"level: {job.experience_years}+ years required")

    # 3. Internship.
    if job.seniority == "intern" and not user.include_internships:
        return _dropped("internship")

    # 4. Languages.
    missing = job.required_languages - user.languages
    if missing:
        return _dropped(f"language: {', '.join(sorted(missing))} required")

    # 5. Excluded company or industry.
    if job.company_id in user.excluded_company_ids:
        return _dropped("excluded company")
    if job.company_industry in user.excluded_industries:
        return _dropped(f"excluded industry: {job.company_industry}")

    # 6. Degree.
    if job.min_degree == "phd" and user.highest_degree not in (None, "phd"):
        return _dropped("degree: PhD required")

    # 7. Role (skipped for a "never miss" favourite).
    never_miss = job.company_id in user.never_miss_company_ids
    has_role = bool(set(job.role_families) & user.roles) or bool(job.matched_custom_roles & user.custom_roles)
    if not has_role and not never_miss:
        return _dropped("role: not one of yours")

    return Verdict(passed=True, reason="passed", country_unclear=country_unclear, never_miss=never_miss)
