"""Loads JobOrbit's YAML configuration files from the config/ folder.

Each file is checked against a Pydantic model when it's loaded, so a typo
in a config file gives a clear error straight away instead of odd behaviour later.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from joborbit.settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"

# A matching term: words of letters or digits separated by single spaces, e.g. "data scientist".
_TERM = re.compile(r"[^\W_]+(?: [^\W_]+)*")
# A slug (a fixed ID): lower-case words joined by underscores, e.g. "data_scientist".
_SLUG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
# A language code: two or three lower-case letters (ISO 639), e.g. "en" or "yue".
_LANGUAGE_CODE = re.compile(r"[a-z]{2,3}")


def _tidy_terms(values: list[str]) -> list[str]:
    """Lower-case each term and squash extra spaces. Refuse empty terms, punctuation and repeats.

    Titles are compared without capitals or punctuation, so a term containing punctuation
    (e.g. "jr.") could never match: it is refused here instead of silently doing nothing.
    """
    tidied = []
    for value in values:
        term = " ".join(str(value).lower().split())
        if not _TERM.fullmatch(term):
            raise ValueError(f"{value!r} must be letters, digits and spaces only")
        if term in tidied:
            raise ValueError(f"{term!r} is listed twice")
        tidied.append(term)
    return tidied


def _check_slug(value: str) -> str:
    if not _SLUG.fullmatch(value):
        raise ValueError(f"slug {value!r} must be lower-case words joined by _, e.g. data_scientist")
    return value


def _find_repeats(items: list[BaseModel], fields: tuple[str, ...]) -> None:
    """Refuse two items sharing a value in any of `fields` (capitals ignored)."""
    for field in fields:
        values = [getattr(item, field).lower() for item in items]
        repeated = sorted({value for value in values if values.count(value) > 1})
        if repeated:
            raise ValueError(f"each entry needs its own {field}; repeated: {repeated}")


# --- countries.yaml -----------------------------------------------------------


class CountryConfig(BaseModel):
    code: str  # ISO code, e.g. "GB"
    name: str
    enabled: bool = False
    timezone: str
    local_languages: list[str] = []
    aliases: list[str] = []
    cities: list[str] = []


class CountriesFile(BaseModel):
    countries: list[CountryConfig]


# --- roles.yaml -----------------------------------------------------------------


class RoleConfig(BaseModel):
    """One default role: an ID, a display name, and the title words that identify it."""

    model_config = ConfigDict(extra="forbid")  # a misspelt field name is an error, not ignored

    slug: str
    name: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)  # at least one, or the role could never match
    exclude: list[str] = []

    @field_validator("slug")
    @classmethod
    def _valid_slug(cls, value: str) -> str:
        return _check_slug(value)

    @field_validator("keywords", "exclude")
    @classmethod
    def _check_terms(cls, values: list[str]) -> list[str]:
        return _tidy_terms(values)

    @model_validator(mode="after")
    def _keywords_and_excludes_differ(self) -> "RoleConfig":
        both = set(self.keywords) & set(self.exclude)
        if both:
            raise ValueError(f"role {self.slug!r} lists {sorted(both)} as both a keyword and an exclude")
        return self


class RolesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: list[RoleConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_repeats(self) -> "RolesFile":
        _find_repeats(self.roles, ("slug", "name"))
        return self


# --- seniority.yaml -------------------------------------------------------------


class SeniorityConfig(BaseModel):
    """Title words that mark a job as too senior, or as an internship."""

    model_config = ConfigDict(extra="forbid")

    drop_words: list[str] = Field(min_length=1)
    internship_words: list[str] = Field(min_length=1)

    @field_validator("drop_words", "internship_words")
    @classmethod
    def _check_terms(cls, values: list[str]) -> list[str]:
        return _tidy_terms(values)


# --- industries.yaml ------------------------------------------------------------


class IndustryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str = Field(min_length=1)

    @field_validator("slug")
    @classmethod
    def _valid_slug(cls, value: str) -> str:
        return _check_slug(value)


class IndustriesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industries: list[IndustryConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_repeats(self) -> "IndustriesFile":
        _find_repeats(self.industries, ("slug", "name"))
        return self


# --- languages.yaml -------------------------------------------------------------


class LanguageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    name: str = Field(min_length=1)

    @field_validator("code")
    @classmethod
    def _valid_code(cls, value: str) -> str:
        if not _LANGUAGE_CODE.fullmatch(value):
            raise ValueError(f"language code {value!r} must be 2 or 3 lower-case letters, e.g. en or yue")
        return value


class LanguagesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    languages: list[LanguageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_repeats(self) -> "LanguagesFile":
        _find_repeats(self.languages, ("code", "name"))
        return self


# --- settings.yaml --------------------------------------------------------------


class LimitsConfig(BaseModel):
    """Limits on what one person can enter, and on the number of people."""

    model_config = ConfigDict(extra="forbid")

    max_roles_per_user: int = Field(gt=0)
    max_role_name_chars: int = Field(gt=0)
    max_favourites_per_user: int = Field(gt=0)
    max_skills_per_user: int = Field(gt=0)
    max_skill_chars: int = Field(gt=0)
    max_users: int = Field(gt=0)


class LlmPrices(BaseModel):
    """What the LLM costs, in US dollars per million tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)

    @model_validator(mode="after")
    def _cached_is_not_dearer(self) -> "LlmPrices":
        if self.cached_input > self.input:
            raise ValueError("cached_input must not cost more than input (are the two swapped?)")
        return self


class LlmConfig(BaseModel):
    """Which LLM reads the jobs, how it is called, and the limits that keep its cost down."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    reasoning_effort: Literal["none", "low", "medium", "high"]
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    network_retries: int = Field(ge=0)
    daily_call_cap: int = Field(gt=0)
    warn_at_fraction: float = Field(gt=0, lt=1)
    monthly_budget_usd: float = Field(gt=0)
    prices_usd_per_million: LlmPrices


Fraction = Annotated[float, Field(ge=0, le=1)]  # a value between 0 and 1


class ScoringWeights(BaseModel):
    """The most points each part of the score can give. They add up to 100."""

    model_config = ConfigDict(extra="forbid")

    role_fit: float = Field(ge=0)
    entry_fit: float = Field(ge=0)
    skills: float = Field(ge=0)
    country: float = Field(ge=0)
    transfer: float = Field(ge=0)

    @model_validator(mode="after")
    def _add_up_to_100(self) -> "ScoringWeights":
        total = sum(self.model_dump().values())
        if abs(total - 100) > 1e-9:
            raise ValueError(f"the weights must add up to 100, not {total:g}")
        return self


class ScoringBonuses(BaseModel):
    """Points added on top of the weighted parts."""

    model_config = ConfigDict(extra="forbid")

    favourite_company: float = Field(ge=0)
    preferred_industry: float = Field(ge=0)


class RoleFitValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact: Fraction  # the title contains one of the role's keywords
    other: Fraction  # the role came from a near match or from the LLM only


class EntryFitValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graduate: Fraction  # graduate scheme or graduate role
    entry: Fraction  # entry level, or an internship for a user who wants them
    unknown: Fraction  # no clue about the level at all
    mid: Fraction
    required_without_years: Fraction  # the most it can be when experience is required with no number


class CountryValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_choice: Fraction  # the user's last country; their first is always 1.0
    unclear: Fraction  # the posting doesn't say where


class SkillsValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_counted: int = Field(gt=0)  # the job's skills counted at most: matching 3 of 6 or more = 0.5
    unknown: Fraction  # the job or the user lists no skills


class AlertThresholds(BaseModel):
    """The lowest score for an instant alert, and for a place in the daily digest."""

    model_config = ConfigDict(extra="forbid")

    instant: int = Field(ge=0, le=100)
    digest: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _digest_not_above_instant(self) -> "AlertThresholds":
        if self.digest > self.instant:
            raise ValueError(f"the digest threshold ({self.digest}) can't be above the instant one ({self.instant})")
        return self


class AlertStyles(BaseModel):
    """The thresholds for each alert style a user can choose."""

    model_config = ConfigDict(extra="forbid")

    fewer: AlertThresholds
    balanced: AlertThresholds
    more: AlertThresholds


class ScoringConfig(BaseModel):
    """How a job that passed a user's hard filters is scored out of 100 (matching/scoring.py),
    and where each score is sent (matching/router.py)."""

    model_config = ConfigDict(extra="forbid")

    weights: ScoringWeights
    bonuses: ScoringBonuses
    role_fit: RoleFitValues
    entry_fit: EntryFitValues
    country: CountryValues
    skills: SkillsValues
    evergreen_after_days: int = Field(gt=0)  # posted this long before we saw it: an old ad, re-listed
    alert_styles: AlertStyles  # where each score goes: instant alert, daily digest, or dashboard only


class AppSettings(BaseModel):
    """Everything in settings.yaml. A new section gets a new field here when it is added."""

    model_config = ConfigDict(extra="forbid")

    limits: LimitsConfig
    llm: LlmConfig
    scoring: ScoringConfig


# --- Loading ------------------------------------------------------------------------


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache
def load_countries(path: Path = CONFIG_DIR / "countries.yaml") -> list[CountryConfig]:
    """Every country in countries.yaml, enabled or not."""
    return CountriesFile.model_validate(_read_yaml(path)).countries


@lru_cache
def load_roles(path: Path = CONFIG_DIR / "roles.yaml") -> list[RoleConfig]:
    """The default roles from roles.yaml, in file order."""
    return RolesFile.model_validate(_read_yaml(path)).roles


@lru_cache
def load_seniority(path: Path = CONFIG_DIR / "seniority.yaml") -> SeniorityConfig:
    """The senior-title and internship words from seniority.yaml."""
    return SeniorityConfig.model_validate(_read_yaml(path))


@lru_cache
def load_industries(path: Path = CONFIG_DIR / "industries.yaml") -> list[IndustryConfig]:
    """Every industry in industries.yaml, in file order."""
    return IndustriesFile.model_validate(_read_yaml(path)).industries


@lru_cache
def load_languages(path: Path = CONFIG_DIR / "languages.yaml") -> list[LanguageConfig]:
    """Every language in languages.yaml, in file order."""
    return LanguagesFile.model_validate(_read_yaml(path)).languages


@lru_cache
def load_app_settings(path: Path = CONFIG_DIR / "settings.yaml") -> AppSettings:
    """The tunable settings from settings.yaml (secrets are in .env, read by joborbit.settings)."""
    return AppSettings.model_validate(_read_yaml(path))
