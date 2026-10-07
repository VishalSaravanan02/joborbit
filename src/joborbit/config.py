"""Loads JobOrbit's YAML configuration files from the config/ folder.

Each file is checked against a Pydantic model when it's loaded, so a typo
in a config file gives a clear error straight away instead of odd behaviour later.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from joborbit.settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"

# A matching term: words of letters or digits separated by single spaces, e.g. "data scientist".
_TERM = re.compile(r"[^\W_]+(?: [^\W_]+)*")
# A role slug: lower-case words joined by underscores, e.g. "data_scientist".
_SLUG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


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
    def _check_slug(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError(f"slug {value!r} must be lower-case words joined by _, e.g. data_scientist")
        return value

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
        for field in ("slug", "name"):
            values = [getattr(role, field).lower() for role in self.roles]
            repeated = sorted({value for value in values if values.count(value) > 1})
            if repeated:
                raise ValueError(f"each role needs its own {field}; repeated: {repeated}")
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
