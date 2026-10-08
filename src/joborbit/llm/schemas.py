"""What a valid answer from the LLM looks like, and the checks that make sure of it.

For every job that passes the pre-filter, the LLM reads the posting and answers with the
facts in JobAnalysisResult. Before anything is saved, the answer is checked:

- Answers that would mean something wrong are refused: a country, language or role code
  we don't know, a custom role we didn't ask about, an impossible number or date. A refused
  answer is retried once, with the reasons from AnswerError sent back to the LLM.
- Anything cosmetic is tidied quietly: repeated or surplus skills, cities and roles, stray
  spaces, a summary that is too long. Paying for a second call over those makes no sense.

None means "the posting doesn't say". Every field must be in the answer, even when it is
None or empty, so a forgotten field is noticed instead of silently becoming a default.

The allowed codes come from the config files (countries.yaml, languages.yaml, roles.yaml),
so adding a language or a role never needs a change here. answer_schema() describes the same
rules as a JSON schema, which is sent with each request so the LLM can only pick allowed values
and can't send longer lists than we keep.
"""

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from joborbit.config import load_countries, load_languages, load_roles

OTHER_LANGUAGE = "other"  # a required language that isn't in languages.yaml
MAX_ROLE_FAMILIES = 3
MAX_SKILLS = 15
MAX_SKILL_CHARS = 50  # a longer "skill" is a sentence, not a skill
MAX_CITIES = 10
MAX_SUMMARY_CHARS = 200  # the size of job_analysis.summary
MAX_EXPERIENCE_YEARS = 40


# --- The allowed codes, from the config files ------------------------------------------------


def country_codes() -> list[str]:
    """Every country in countries.yaml, switched on or not, so switching one on needs no re-analysis."""
    return [country.code for country in load_countries()]


def language_codes() -> list[str]:
    """Every language code in languages.yaml, plus "other" for a language that isn't listed."""
    return [language.code for language in load_languages()] + [OTHER_LANGUAGE]


def role_slugs() -> list[str]:
    """Every default role's slug from roles.yaml."""
    return [role.slug for role in load_roles()]


# --- Small helpers -----------------------------------------------------------------------------


def _tidy(value: str) -> str:
    """Trim and squash repeated spaces and line breaks: " Machine\\n learning " -> "Machine learning"."""
    return " ".join(value.split())


def _unique(values: Iterable[str]) -> list[str]:
    """Drop empty values and repeats (capitals ignored), keeping the first of each, in order."""
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value.lower() not in seen:
            seen.add(value.lower())
            result.append(value)
    return result


def _known(values: list[str], allowed: list[str], what: str) -> list[str]:
    """Refuse any value that isn't allowed, naming the allowed ones (the LLM reads this on a retry)."""
    unknown = [value for value in values if value not in allowed]
    if unknown:
        listed = ", ".join(repr(value) for value in unknown)
        raise ValueError(f"unknown {what} {listed}; choose from: {', '.join(allowed)}")
    return values


# --- The answer ------------------------------------------------------------------------------------


class JobAnalysisResult(BaseModel):
    """The LLM's reading of one job posting, checked and tidied.

    Custom roles are checked against the names sent with the request, given as validation
    context: JobAnalysisResult.model_validate(data, context={"custom_roles": [...]}).
    parse_answer does this for you.
    """

    model_config = ConfigDict(extra="forbid")  # a field we didn't ask for is an error, not ignored

    countries: list[str]  # where the job can be done, from countries.yaml
    cities: list[str] = Field(json_schema_extra={"maxItems": MAX_CITIES})
    work_mode: Literal["onsite", "hybrid", "remote"] | None
    seniority: Literal["intern", "graduate", "entry", "mid", "senior"] | None
    is_graduate_scheme: bool
    experience_years: int | None = Field(ge=0, le=MAX_EXPERIENCE_YEARS)
    experience_mandatory: bool  # True only if the experience is required, not just preferred
    required_languages: list[str]  # codes from languages.yaml, or "other"
    # Role slugs, best first; empty = none of our roles.
    role_families: list[str] = Field(json_schema_extra={"maxItems": MAX_ROLE_FAMILIES})
    matched_custom_roles: list[str]  # names from the custom roles sent with the request
    skills: list[str] = Field(json_schema_extra={"maxItems": MAX_SKILLS})
    min_degree: Literal["none", "bachelor", "master", "phd"] | None
    deadline: date | None
    summary: str  # one line for alerts

    # --- Codes: refused if unknown ---

    @field_validator("countries")
    @classmethod
    def _check_countries(cls, values: list[str]) -> list[str]:
        codes = _unique(_tidy(value).upper() for value in values)
        return _known(codes, country_codes(), "country code")

    @field_validator("required_languages")
    @classmethod
    def _check_languages(cls, values: list[str]) -> list[str]:
        codes = _unique(_tidy(value).lower() for value in values)
        return _known(codes, language_codes(), "language code")

    @field_validator("role_families")
    @classmethod
    def _check_roles(cls, values: list[str]) -> list[str]:
        slugs = _known(_unique(_tidy(value).lower() for value in values), role_slugs(), "role")
        return slugs[:MAX_ROLE_FAMILIES]  # best first, so the extra ones matter least

    @field_validator("matched_custom_roles")
    @classmethod
    def _check_custom_roles(cls, values: list[str], info: ValidationInfo) -> list[str]:
        asked = _unique(_tidy(name) for name in (info.context or {}).get("custom_roles", []))
        by_lower = {name.lower(): name for name in asked}
        names = _unique(_tidy(value) for value in values)
        unknown = [name for name in names if name.lower() not in by_lower]
        if unknown:
            allowed = ", ".join(asked) if asked else "none were given, so leave this empty"
            raise ValueError(f"not a custom role we asked about: {', '.join(unknown)}; allowed: {allowed}")
        return [by_lower[name.lower()] for name in names]  # written exactly as the user wrote it

    # --- Free text: tidied quietly ---

    @field_validator("skills")
    @classmethod
    def _tidy_skills(cls, values: list[str]) -> list[str]:
        skills = [_tidy(value) for value in values]
        return _unique(skill for skill in skills if len(skill) <= MAX_SKILL_CHARS)[:MAX_SKILLS]

    @field_validator("cities")
    @classmethod
    def _tidy_cities(cls, values: list[str]) -> list[str]:
        return _unique(_tidy(value) for value in values)[:MAX_CITIES]

    @field_validator("summary")
    @classmethod
    def _tidy_summary(cls, value: str) -> str:
        text = _tidy(value)
        if not text:
            raise ValueError("the summary is empty")
        if len(text) > MAX_SUMMARY_CHARS:
            text = text[: MAX_SUMMARY_CHARS - 2].rsplit(" ", 1)[0] + " …"  # cut at a word boundary
        return text


# --- Reading an answer -------------------------------------------------------------------------------


class AnswerError(ValueError):
    """The LLM's answer broke the rules. The message lists every problem, one per line."""


def parse_answer(text: str, custom_roles: Sequence[str] = ()) -> JobAnalysisResult:
    """Check the LLM's raw JSON answer. Raises AnswerError, listing every problem, if it isn't valid.

    `custom_roles` are the custom role names sent with the request: the only ones it may match.
    """
    try:
        return JobAnalysisResult.model_validate_json(text, context={"custom_roles": list(custom_roles)})
    except ValidationError as error:
        raise AnswerError(_describe(error)) from error


def _describe(error: ValidationError) -> str:
    """Pydantic's error list as short plain lines, e.g. "countries: unknown country code 'US'; ..."."""
    lines = []
    for problem in error.errors():
        where = " > ".join(str(part) for part in problem["loc"]) or "answer"
        if problem["type"] == "extra_forbidden":
            message = "not a field in the answer; remove it"
        else:
            message = problem["msg"].removeprefix("Value error, ")
        lines.append(f"{where}: {message}")
    return "\n".join(lines)


# --- The rules as a JSON schema, for the LLM ---------------------------------------------------------


def answer_schema(custom_roles: Sequence[str] = ()) -> dict[str, Any]:
    """The answer's shape as a JSON schema, with the allowed codes filled in from the config files.

    Sent with each request (llm/client.py), so the LLM can only choose allowed values. The checks
    above still run on every answer. With no custom roles, the schema says that list must be empty.
    The list limits (maxItems) only tell the LLM how many we keep: a longer list is tidied, not refused.
    """
    schema = JobAnalysisResult.model_json_schema()
    schema.pop("description", None)  # the class docstring is written for us; the prompt instructs the LLM
    allowed = {
        "countries": country_codes(),
        "required_languages": language_codes(),
        "role_families": role_slugs(),
        "matched_custom_roles": _unique(_tidy(name) for name in custom_roles),
    }
    for field, values in allowed.items():
        if values:
            schema["properties"][field]["items"]["enum"] = values
        else:
            schema["properties"][field]["maxItems"] = 0  # nothing to choose from: the list must be empty
    return schema
