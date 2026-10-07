"""Tests for the rules an LLM answer must follow (joborbit/llm/schemas.py)."""

import json
from datetime import date

import pytest

from joborbit.config import load_countries, load_languages, load_roles
from joborbit.db.models import JobAnalysis
from joborbit.llm.schemas import (
    MAX_CITIES,
    MAX_SKILL_CHARS,
    MAX_SKILLS,
    MAX_SUMMARY_CHARS,
    OTHER_LANGUAGE,
    AnswerError,
    JobAnalysisResult,
    answer_schema,
    parse_answer,
)

# The build plan's example answer, with the changes agreed for step 5:
# role_families instead of role_family, no industry, no is_internship.
VALID = {
    "countries": ["GB"],
    "cities": ["London"],
    "work_mode": "hybrid",
    "seniority": "graduate",
    "is_graduate_scheme": True,
    "experience_years": None,
    "experience_mandatory": False,
    "required_languages": ["en"],
    "role_families": ["data_scientist"],
    "matched_custom_roles": [],
    "skills": ["python", "sql", "machine learning"],
    "min_degree": "bachelor",
    "deadline": "2026-11-15",
    "summary": "Two-year data science graduate scheme with rotations across risk and marketing.",
}


def parse(custom_roles=(), **changes) -> JobAnalysisResult:
    """The valid answer with `changes` applied, sent through the real checks as JSON text."""
    return parse_answer(json.dumps({**VALID, **changes}), custom_roles)


def refused(match: str, custom_roles=(), **changes) -> None:
    with pytest.raises(AnswerError, match=match):
        parse(custom_roles, **changes)


# --- A good answer ---------------------------------------------------------------------------


def test_the_plans_example_answer_is_accepted():
    result = parse()
    assert result.countries == ["GB"] and result.role_families == ["data_scientist"]
    assert result.is_graduate_scheme is True and result.experience_mandatory is False
    assert result.deadline == date(2026, 11, 15)
    assert result.summary == VALID["summary"]


def test_none_means_the_posting_does_not_say():
    result = parse(work_mode=None, seniority=None, experience_years=None, min_degree=None, deadline=None)
    assert (result.work_mode, result.seniority, result.min_degree, result.deadline) == (None, None, None, None)


def test_no_degree_needed_is_a_real_answer():
    assert parse(min_degree="none").min_degree == "none"


@pytest.mark.parametrize("field", list(VALID))
def test_every_field_must_be_in_the_answer(field):
    answer = {name: value for name, value in VALID.items() if name != field}
    with pytest.raises(AnswerError, match=f"{field}: Field required"):
        parse_answer(json.dumps(answer))


@pytest.mark.parametrize("field", ["industry", "is_internship", "role_family"])
def test_fields_we_did_not_ask_for_are_refused(field):
    """Including the build plan's three fields that step 5 removed."""
    refused(f"{field}: not a field in the answer", **{field: "x"})


def test_an_answer_that_is_not_json_is_refused():
    with pytest.raises(AnswerError, match="answer: Invalid JSON"):
        parse_answer("Sure! Here is the analysis: {")


def test_every_problem_is_listed_at_once():
    with pytest.raises(AnswerError) as error:
        parse(countries=["US"], role_families=["data_wizard"])
    lines = str(error.value).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("countries: unknown country code 'US'")
    assert lines[1].startswith("role_families: unknown role 'data_wizard'")


# --- Codes from the config files ------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("countries", ["US"], "unknown country code 'US'"),
        ("required_languages", ["sv"], "unknown language code 'sv'"),
        ("role_families", ["data_wizard"], "unknown role 'data_wizard'"),
    ],
)
def test_unknown_codes_are_refused_and_the_allowed_ones_named(field, value, message):
    refused(f"{field}: {message}; choose from: ", **{field: value})


def test_codes_are_tidied():
    result = parse(
        countries=["gb", " GB "],
        required_languages=["EN", "es", "en"],
        role_families=[" Data_Scientist", "data_scientist"],
    )
    assert result.countries == ["GB"]
    assert result.required_languages == ["en", "es"]
    assert result.role_families == ["data_scientist"]


def test_countries_that_are_not_switched_on_are_still_allowed():
    """So jobs analysed now are still right when India or Singapore is switched on."""
    assert parse(countries=["GB", "IN", "SG"]).countries == ["GB", "IN", "SG"]


def test_a_language_not_in_our_list_can_be_reported_as_other():
    """A job requiring, say, Swedish must not look like a job with no extra language."""
    assert parse(required_languages=["en", "other"]).required_languages == ["en", OTHER_LANGUAGE]


def test_other_can_never_clash_with_a_real_language_code():
    assert OTHER_LANGUAGE not in [language.code for language in load_languages()]


def test_an_empty_role_list_means_none_of_our_roles():
    assert parse(role_families=[]).role_families == []


def test_roles_keep_their_order_and_at_most_three_are_kept():
    roles = ["ml_engineer", "applied_scientist", "ml_engineer", "data_scientist", "ai_engineer"]
    assert parse(role_families=roles).role_families == ["ml_engineer", "applied_scientist", "data_scientist"]


# --- Custom roles -------------------------------------------------------------------------------


def test_custom_roles_are_matched_and_written_as_the_user_wrote_them():
    result = parse(["Insights Analyst", "Pricing Analyst"], matched_custom_roles=["insights  analyst"])
    assert result.matched_custom_roles == ["Insights Analyst"]


def test_a_custom_role_we_did_not_ask_about_is_refused():
    refused(
        "not a custom role we asked about: Growth Hacker; allowed: Insights Analyst",
        ["Insights Analyst"],
        matched_custom_roles=["Growth Hacker"],
    )


def test_with_no_custom_roles_none_can_be_matched():
    refused("none were given, so leave this empty", matched_custom_roles=["Insights Analyst"])


# --- Free text: tidied, never refused ---------------------------------------------------------


def test_skills_are_tidied_and_limited():
    many = [f"skill {number}" for number in range(MAX_SKILLS + 5)]
    long = "x" * (MAX_SKILL_CHARS + 1)
    result = parse(skills=[" Python ", "python", "", "machine\nlearning", long, *many])
    assert result.skills[:2] == ["Python", "machine learning"]
    assert long not in result.skills
    assert len(result.skills) == MAX_SKILLS


def test_cities_are_tidied_and_limited():
    cities = [" London ", "london", *[f"City {number}" for number in range(MAX_CITIES + 5)]]
    result = parse(cities=cities)
    assert result.cities[0] == "London" and result.cities.count("London") == 1
    assert len(result.cities) == MAX_CITIES


def test_the_summary_becomes_one_tidy_line():
    assert parse(summary="  Data   science\ngraduate scheme. ").summary == "Data science graduate scheme."


def test_a_long_summary_is_cut_at_a_word_boundary():
    summary = parse(summary="word " * 100).summary
    assert len(summary) <= MAX_SUMMARY_CHARS
    assert summary.endswith("word …")


def test_an_empty_summary_is_refused():
    refused("summary: the summary is empty", summary="   ")


# --- Numbers, dates and choices ------------------------------------------------------------------


@pytest.mark.parametrize("years", [0, 2, 40])
def test_sensible_experience_is_accepted(years):
    assert parse(experience_years=years, experience_mandatory=True).experience_years == years


@pytest.mark.parametrize("years", [-1, 41])
def test_impossible_experience_is_refused(years):
    refused("experience_years", experience_years=years)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("work_mode", "office"),
        ("seniority", "junior"),
        ("seniority", "unknown"),  # "not stated" is None, not a word
        ("min_degree", "diploma"),
        ("deadline", "15 November"),
        ("is_graduate_scheme", "maybe"),
    ],
)
def test_values_outside_the_choices_are_refused(field, value):
    refused(field, **{field: value})


# --- The JSON schema sent to the LLM ------------------------------------------------------------


def items_enum(schema: dict, field: str) -> list[str] | None:
    return schema["properties"][field]["items"].get("enum")


def test_the_schema_lists_exactly_the_codes_in_the_config_files():
    schema = answer_schema()
    assert items_enum(schema, "countries") == [country.code for country in load_countries()]
    assert items_enum(schema, "required_languages") == [language.code for language in load_languages()] + ["other"]
    assert items_enum(schema, "role_families") == [role.slug for role in load_roles()]


def test_the_schema_lists_the_custom_roles_sent():
    schema = answer_schema(["Insights Analyst", " insights analyst ", "Pricing Analyst"])
    assert items_enum(schema, "matched_custom_roles") == ["Insights Analyst", "Pricing Analyst"]
    assert items_enum(answer_schema(), "matched_custom_roles") is None  # left open; the check refuses anything


def test_the_schema_requires_every_field_and_allows_no_others():
    schema = answer_schema()
    assert schema["required"] == list(VALID)
    assert schema["additionalProperties"] is False


def test_the_schema_carries_no_notes_meant_for_programmers():
    assert "description" not in answer_schema()


def test_the_schema_choices_match_the_checks():
    properties = answer_schema()["properties"]
    assert properties["work_mode"]["anyOf"][0]["enum"] == ["onsite", "hybrid", "remote"]
    assert properties["seniority"]["anyOf"][0]["enum"] == ["intern", "graduate", "entry", "mid", "senior"]
    assert properties["min_degree"]["anyOf"][0]["enum"] == ["none", "bachelor", "master", "phd"]


# --- The answer and the job_analysis table agree ----------------------------------------------


def test_every_answer_field_has_a_column_in_job_analysis():
    columns = set(JobAnalysis.__table__.columns.keys())
    assert set(JobAnalysisResult.model_fields) <= columns


def test_the_summary_limit_matches_its_column():
    assert JobAnalysis.__table__.columns["summary"].type.length == MAX_SUMMARY_CHARS
