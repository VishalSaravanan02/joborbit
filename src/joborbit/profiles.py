"""The rules for a valid user profile, in one place.

Today my_profile.yaml is checked with these rules (scripts/seed_my_profile.py); in Phase 4
the dashboard's profile form uses exactly the same rules, so the two can never disagree.

ProfileInput only checks what was entered: known roles, countries, languages and
industries, the limits in config/settings.yaml, and choices that contradict each other.
Companies are given by name here; save_profile looks them up in the database and stores
the profile, so the form and the script also save in exactly the same way.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.companies import slugify
from joborbit.config import load_app_settings, load_countries, load_industries, load_languages, load_roles
from joborbit.db.models import Company, User, UserCompanyPref, UserProfile


def _tidy(value: object) -> str:
    """Trim and squash repeated spaces: "  Data   Analyst " -> "Data Analyst"."""
    return " ".join(str(value).split())


def _no_repeats(values: list[str], what: str) -> list[str]:
    """Refuse a list that names the same thing twice (capitals ignored)."""
    seen: set[str] = set()
    for value in values:
        if value.lower() in seen:
            raise ValueError(f"{what} {value!r} is listed twice")
        seen.add(value.lower())
    return values


def _free_text(values: list[str], what: str, max_chars: int) -> list[str]:
    """Tidy free-text entries (custom roles, skills): not empty, not too long, no repeats."""
    tidied = [_tidy(value) for value in values]
    for value in tidied:
        if not re.search(r"[^\W_]", value):
            raise ValueError(f"each {what} needs at least one letter or digit, got {value!r}")
        if len(value) > max_chars:
            raise ValueError(f"{what} {value!r} is longer than {max_chars} characters")
    return _no_repeats(tidied, what)


def _known(values: list[str], allowed: list[str], what: str) -> list[str]:
    """Refuse any value that isn't in the allowed list (and say which ones are allowed)."""
    for value in values:
        if value not in allowed:
            raise ValueError(f"unknown {what} {value!r}; choose from: {', '.join(allowed)}")
    return _no_repeats(values, what)


class FavouriteCompany(BaseModel):
    """A favourite company. "Never miss" means every suitable job there is sent at once."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    never_miss: bool = False

    @field_validator("name", mode="before")
    @classmethod
    def _tidy_name(cls, value: object) -> str:
        return _tidy(value)


class ProfileInput(BaseModel):
    """Everything one person can set in their profile, checked.

    Only one role and one country are needed; everything else has a sensible default.
    Leaving `roles` out means "all the default roles", as on the form.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelt field name is an error, not ignored

    roles: list[str] = Field(default_factory=lambda: [role.slug for role in load_roles()])
    custom_roles: list[str] = []
    countries: list[str]  # ranked: first = most wanted
    languages: list[str] = ["en"]
    preferred_industries: list[str] = []
    excluded_industries: list[str] = []
    skills: list[str] = []
    highest_degree: Literal["bachelor", "master", "phd"] | None = None
    include_internships: bool = False
    alert_style: Literal["fewer", "balanced", "more"] = "balanced"
    transfer_boost: bool = True
    favourite_companies: list[FavouriteCompany] = []
    excluded_companies: list[str] = []

    # --- One field at a time ---

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, values: list[str]) -> list[str]:
        slugs = [role.slug for role in load_roles()]
        return _known([value.strip().lower() for value in values], slugs, "role")

    @field_validator("custom_roles")
    @classmethod
    def _check_custom_roles(cls, values: list[str]) -> list[str]:
        return _free_text(values, "custom role", load_app_settings().limits.max_role_name_chars)

    @field_validator("countries")
    @classmethod
    def _check_countries(cls, values: list[str]) -> list[str]:
        codes = [value.strip().upper() for value in values]
        if not codes:
            raise ValueError("choose at least one country")
        everything = {country.code for country in load_countries()}
        switched_on = [country.code for country in load_countries() if country.enabled]
        for code in codes:
            if code in everything and code not in switched_on:
                raise ValueError(f"{code} is not switched on yet; available now: {', '.join(switched_on)}")
        return _known(codes, switched_on, "country")

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, values: list[str]) -> list[str]:
        codes = [value.strip().lower() for value in values]
        if not codes:
            raise ValueError("choose at least one language")
        return _known(codes, [language.code for language in load_languages()], "language")

    @field_validator("preferred_industries", "excluded_industries")
    @classmethod
    def _check_industries(cls, values: list[str]) -> list[str]:
        slugs = [industry.slug for industry in load_industries()]
        return _known([value.strip().lower() for value in values], slugs, "industry")

    @field_validator("skills")
    @classmethod
    def _check_skills(cls, values: list[str]) -> list[str]:
        limits = load_app_settings().limits
        if len(values) > limits.max_skills_per_user:
            raise ValueError(f"at most {limits.max_skills_per_user} skills, got {len(values)}")
        return _free_text(values, "skill", limits.max_skill_chars)

    @field_validator("favourite_companies")
    @classmethod
    def _check_favourites(cls, values: list[FavouriteCompany]) -> list[FavouriteCompany]:
        limit = load_app_settings().limits.max_favourites_per_user
        if len(values) > limit:
            raise ValueError(f"at most {limit} favourite companies, got {len(values)}")
        _no_repeats([slugify(company.name) for company in values], "favourite company")
        return values

    @field_validator("excluded_companies")
    @classmethod
    def _check_excluded(cls, values: list[str]) -> list[str]:
        names = [_tidy(value) for value in values]
        if not all(names):
            raise ValueError("an excluded company name is empty")
        _no_repeats([slugify(name) for name in names], "excluded company")
        return names

    # --- Choices that must agree with each other ---

    @model_validator(mode="after")
    def _check_role_total(self) -> "ProfileInput":
        total, limit = len(self.roles) + len(self.custom_roles), load_app_settings().limits.max_roles_per_user
        if total == 0:
            raise ValueError("choose at least one role (a default role or a custom one)")
        if total > limit:
            raise ValueError(f"at most {limit} roles in total, got {total}")
        default_names = {role.name.lower(): role.name for role in load_roles()}
        for custom in self.custom_roles:
            if custom.lower() in default_names:
                raise ValueError(f"custom role {custom!r} is already a default role ({default_names[custom.lower()]})")
        return self

    @model_validator(mode="after")
    def _check_industries_differ(self) -> "ProfileInput":
        both = sorted(set(self.preferred_industries) & set(self.excluded_industries))
        if both:
            raise ValueError(f"an industry can't be both preferred and excluded: {', '.join(both)}")
        return self

    @model_validator(mode="after")
    def _check_companies_differ(self) -> "ProfileInput":
        favourites = {slugify(company.name): company.name for company in self.favourite_companies}
        both = sorted(favourites[slugify(name)] for name in self.excluded_companies if slugify(name) in favourites)
        if both:
            raise ValueError(f"a company can't be both a favourite and excluded: {', '.join(both)}")
        return self


# --- Saving -----------------------------------------------------------------------------


class UnknownCompaniesError(ValueError):
    """Some company names in a profile don't match any company we track."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(f"not tracked (check the spelling, or add them to the company list first): {', '.join(names)}")
        self.names = names


def save_profile(session: Session, user: User, profile: ProfileInput) -> None:
    """Make `user`'s stored profile and company choices match `profile` exactly.

    Companies are matched by slug, so capitals and punctuation don't matter ("checkout com"
    finds Checkout.com). If any name is unknown, UnknownCompaniesError is raised before
    anything is changed. Saving never commits: the caller decides when to save.
    Personal scoring weights are left alone: only the weekly tuning sets them.
    """
    companies = {company.slug: company for company in session.scalars(select(Company))}
    wanted = [(company.name, "favourite", company.never_miss) for company in profile.favourite_companies]
    wanted += [(name, "excluded", False) for name in profile.excluded_companies]
    unknown = [name for name, _, _ in wanted if slugify(name) not in companies]
    if unknown:
        raise UnknownCompaniesError(unknown)

    fields = profile.model_dump(exclude={"favourite_companies", "excluded_companies"})
    if user.profile is None:
        user.profile = UserProfile(**fields)
    else:
        for name, value in fields.items():
            setattr(user.profile, name, value)

    # Update choices that are still wanted, add new ones, and drop the rest.
    existing = {pref.company_id: pref for pref in user.company_prefs}
    choices = []
    for name, kind, never_miss in wanted:
        company = companies[slugify(name)]
        pref = existing.get(company.id) or UserCompanyPref(company=company)
        pref.kind, pref.never_miss = kind, never_miss
        choices.append(pref)
    user.company_prefs = choices
