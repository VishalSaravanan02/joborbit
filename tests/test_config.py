"""Tests for loading the config files (roles, seniority, industries, languages, settings) and their rules."""

import re

import pytest
import yaml
from pydantic import ValidationError

from joborbit.config import (
    load_app_settings,
    load_countries,
    load_industries,
    load_languages,
    load_roles,
    load_seniority,
)

# The 15 default roles agreed in the build plan, in order.
DEFAULT_SLUGS = [
    "data_scientist", "ml_engineer", "ai_engineer", "nlp_llm_engineer", "applied_scientist",
    "data_analyst", "business_analyst", "bi_analyst", "product_analyst", "analytics_engineer",
    "data_engineer", "credit_risk_analyst", "quant_analyst", "data_ai_consultant", "tech_graduate_scheme",
]


def contains_words(text: str, term: str) -> bool:
    """True if `term` appears in `text` as whole words."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


# --- The real config files --------------------------------------------------------


def test_the_15_default_roles_are_loaded_in_order():
    assert [role.slug for role in load_roles()] == DEFAULT_SLUGS


def test_every_role_has_a_name_and_keywords():
    for role in load_roles():
        assert role.name and role.keywords


def test_no_keyword_belongs_to_two_roles():
    keywords = [keyword for role in load_roles() for keyword in role.keywords]
    assert len(keywords) == len(set(keywords))


def test_no_keyword_contains_a_drop_or_internship_word():
    """Such a keyword could never let a job through: the title would always be dropped."""
    seniority = load_seniority()
    for role in load_roles():
        for keyword in role.keywords:
            for word in seniority.drop_words + seniority.internship_words:
                assert not contains_words(keyword, word), f"{role.slug}: {keyword!r} contains {word!r}"


def test_seniority_words_are_loaded():
    seniority = load_seniority()
    assert "senior" in seniority.drop_words and "head of" in seniority.drop_words
    assert "ii" not in seniority.drop_words  # level II is left for the LLM to judge
    assert "intern" in seniority.internship_words


def test_the_industries_include_the_three_added_in_phase_2():
    slugs = [industry.slug for industry in load_industries()]
    assert len(slugs) == 24
    assert {"ai", "travel", "trading", "ecommerce", "other"} <= set(slugs)


def test_every_country_language_is_in_the_language_list():
    """countries.yaml and languages.yaml must never drift apart."""
    codes = {language.code for language in load_languages()}
    for country in load_countries():
        missing = set(country.local_languages) - codes
        assert not missing, f"{country.code} uses languages missing from languages.yaml: {missing}"


def test_english_and_spanish_are_offered():
    names = {language.code: language.name for language in load_languages()}
    assert names["en"] == "English" and names["es"] == "Spanish" and names["yue"] == "Cantonese"


# --- The checks made when a file is loaded ----------------------------------------------


def write(tmp_path, text: str, name: str = "config.yaml"):
    """Write a config file for a test. Loaders remember each file by its path, so a test
    that loads two different versions must give each its own `name`."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def roles_file(tmp_path, *, slug="data_scientist", keywords="[data scientist]", exclude="[]", extra=""):
    text = f"roles:\n  - slug: {slug}\n    name: Data Scientist\n    keywords: {keywords}\n    exclude: {exclude}\n"
    return write(tmp_path, text + extra)


def test_capitals_and_extra_spaces_are_tidied(tmp_path):
    [role] = load_roles(roles_file(tmp_path, keywords='["Data   Scientist", " ML Engineer "]'))
    assert role.keywords == ["data scientist", "ml engineer"]


@pytest.mark.parametrize("keywords", ['["jr. data scientist"]', '["data-scientist"]', '[""]', '["data_scientist"]'])
def test_keywords_with_punctuation_or_nothing_in_them_are_refused(tmp_path, keywords):
    with pytest.raises(ValidationError, match="letters, digits and spaces only"):
        load_roles(roles_file(tmp_path, keywords=keywords))


def test_a_keyword_listed_twice_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="listed twice"):
        load_roles(roles_file(tmp_path, keywords="[data scientist, Data Scientist]"))


def test_a_role_without_keywords_is_refused(tmp_path):
    with pytest.raises(ValidationError):
        load_roles(roles_file(tmp_path, keywords="[]"))


@pytest.mark.parametrize("slug", ["Data_Scientist", "data scientist", "data-scientist", "_data"])
def test_badly_written_slugs_are_refused(tmp_path, slug):
    with pytest.raises(ValidationError, match="slug"):
        load_roles(roles_file(tmp_path, slug=f'"{slug}"'))


def test_two_roles_with_the_same_slug_are_refused(tmp_path):
    second = "  - slug: data_scientist\n    name: Other Name\n    keywords: [ds]\n"
    with pytest.raises(ValidationError, match="repeated"):
        load_roles(roles_file(tmp_path, extra=second))


def test_two_roles_with_the_same_name_are_refused(tmp_path):
    second = "  - slug: other\n    name: data scientist\n    keywords: [ds]\n"
    with pytest.raises(ValidationError, match="repeated"):
        load_roles(roles_file(tmp_path, extra=second))


def test_a_word_cannot_be_both_a_keyword_and_an_exclude(tmp_path):
    with pytest.raises(ValidationError, match="both a keyword and an exclude"):
        load_roles(roles_file(tmp_path, exclude="[Data Scientist]"))


def test_a_misspelt_field_name_is_an_error_not_a_silent_default(tmp_path):
    """'keyword' instead of 'keywords' must fail loudly rather than leave the role empty."""
    text = "roles:\n  - slug: data_scientist\n    name: Data Scientist\n    keyword: [data scientist]\n"
    path = write(tmp_path, text)
    with pytest.raises(ValidationError):
        load_roles(path)


@pytest.mark.parametrize(
    "text",
    [
        "roles:\n  - slug: data_scientist\n    name: Data Scientist\n    keywords: [ds]\n    excludes: [x]\n",
        "role:\n  - slug: data_scientist\n    name: Data Scientist\n    keywords: [ds]\n",
    ],
)
def test_unknown_field_names_are_refused(tmp_path, text):
    """A typo in an optional field ('excludes') would otherwise be silently ignored."""
    with pytest.raises(ValidationError):
        load_roles(write(tmp_path, text))


def test_unknown_seniority_fields_are_refused(tmp_path):
    with pytest.raises(ValidationError):
        load_seniority(write(tmp_path, "drop_words: [senior]\ninternship_words: [intern]\nentry_words: [junior]\n"))


@pytest.mark.parametrize("missing", ["drop_words", "internship_words"])
def test_seniority_needs_both_lists(tmp_path, missing):
    lists = {"drop_words": "[senior]", "internship_words": "[intern]"}
    lists[missing] = "[]"
    path = write(tmp_path, "".join(f"{name}: {value}\n" for name, value in lists.items()))
    with pytest.raises(ValidationError):
        load_seniority(path)


def test_seniority_words_are_tidied_and_checked(tmp_path):
    seniority = load_seniority(write(tmp_path, "drop_words: [Senior, Head  Of]\ninternship_words: [intern]\n"))
    assert seniority.drop_words == ["senior", "head of"]
    with pytest.raises(ValidationError, match="letters, digits and spaces only"):
        load_seniority(write(tmp_path, "drop_words: [sr.]\ninternship_words: [intern]\n", "bad.yaml"))


@pytest.mark.parametrize("slug", ["E-commerce", "transport and logistics", "ai!"])
def test_badly_written_industry_slugs_are_refused(tmp_path, slug):
    with pytest.raises(ValidationError, match="slug"):
        load_industries(write(tmp_path, f'industries:\n  - {{slug: "{slug}", name: Something}}\n'))


def test_two_industries_with_the_same_slug_are_refused(tmp_path):
    text = "industries:\n  - {slug: banking, name: Banking}\n  - {slug: banking, name: Banks}\n"
    with pytest.raises(ValidationError, match="repeated"):
        load_industries(write(tmp_path, text))


@pytest.mark.parametrize("code", ["EN", "e", "engl", "e1"])
def test_badly_written_language_codes_are_refused(tmp_path, code):
    with pytest.raises(ValidationError, match="language code"):
        load_languages(write(tmp_path, f'languages:\n  - {{code: "{code}", name: English}}\n'))


def test_two_languages_with_the_same_code_are_refused(tmp_path):
    text = "languages:\n  - {code: zh, name: Mandarin Chinese}\n  - {code: zh, name: Chinese}\n"
    with pytest.raises(ValidationError, match="repeated"):
        load_languages(write(tmp_path, text))


def test_unknown_industry_or_language_fields_are_refused(tmp_path):
    with pytest.raises(ValidationError):
        load_industries(write(tmp_path, "industries:\n  - {slug: ai, name: AI, label: x}\n", "industries.yaml"))
    with pytest.raises(ValidationError):
        load_languages(write(tmp_path, "languages:\n  - {code: en, nam: English}\n", "languages.yaml"))


def test_the_settings_limits_are_loaded():
    limits = load_app_settings().limits
    assert (limits.max_roles_per_user, limits.max_favourites_per_user, limits.max_users) == (20, 30, 6)


def test_the_llm_settings_are_loaded():
    llm = load_app_settings().llm
    assert (llm.model, llm.reasoning_effort) == ("gpt-6-luna", "medium")
    assert (llm.daily_call_cap, llm.warn_at_fraction, llm.monthly_budget_usd) == (300, 0.8, 8)
    prices = llm.prices_usd_per_million
    assert (prices.input, prices.cached_input, prices.output) == (0.10, 0.01, 0.50)


VALID_LIMITS = {
    "max_roles_per_user": 20,
    "max_role_name_chars": 50,
    "max_favourites_per_user": 30,
    "max_skills_per_user": 40,
    "max_skill_chars": 50,
    "max_users": 6,
}

VALID_LLM = {
    "model": "gpt-6-luna",
    "reasoning_effort": "low",
    "max_output_tokens": 4000,
    "timeout_seconds": 60,
    "network_retries": 3,
    "daily_call_cap": 150,
    "warn_at_fraction": 0.8,
    "monthly_budget_usd": 8,
    "prices_usd_per_million": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
}


def settings_text(limits: dict = VALID_LIMITS, llm: dict = VALID_LLM, extra: str = "") -> str:
    """A whole settings file: valid, except for whatever a test changes."""
    return yaml.safe_dump({"limits": limits, "llm": llm}, sort_keys=False) + extra


def without(data: dict, key: str) -> dict:
    return {name: value for name, value in data.items() if name != key}


def with_prices(**changes) -> dict:
    return {**VALID_LLM, "prices_usd_per_million": {**VALID_LLM["prices_usd_per_million"], **changes}}


def test_a_complete_settings_file_loads(tmp_path):
    """The baseline for the next tests: only the one deliberate mistake may make a file fail."""
    settings = load_app_settings(write(tmp_path, settings_text()))
    assert settings.limits.max_users == 6 and settings.llm.daily_call_cap == 150


@pytest.mark.parametrize(
    ("limits", "extra"),
    [
        ({**VALID_LIMITS, "max_roles_per_user": 0}, ""),  # limits must be above zero
        (without(VALID_LIMITS, "max_users"), ""),  # one limit missing
        ({**VALID_LIMITS, "max_role_per_user": 20}, ""),  # a misspelt limit
        (VALID_LIMITS, "limts:\n  max_users: 6\n"),  # a misspelt section
    ],
)
def test_bad_settings_are_refused(tmp_path, limits, extra):
    with pytest.raises(ValidationError):
        load_app_settings(write(tmp_path, settings_text(limits, extra=extra)))


def test_a_settings_file_without_the_llm_section_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="llm"):
        load_app_settings(write(tmp_path, yaml.safe_dump({"limits": VALID_LIMITS})))


BAD_LLM = {
    "no model": {**VALID_LLM, "model": ""},
    "unknown effort": {**VALID_LLM, "reasoning_effort": "extreme"},
    "no output tokens": {**VALID_LLM, "max_output_tokens": 0},
    "no timeout": {**VALID_LLM, "timeout_seconds": 0},
    "negative retries": {**VALID_LLM, "network_retries": -1},
    "no daily calls": {**VALID_LLM, "daily_call_cap": 0},
    "warn at 0": {**VALID_LLM, "warn_at_fraction": 0},
    "warn at the cap": {**VALID_LLM, "warn_at_fraction": 1},
    "no monthly budget": {**VALID_LLM, "monthly_budget_usd": 0},
    "cap missing": without(VALID_LLM, "daily_call_cap"),
    "prices missing": without(VALID_LLM, "prices_usd_per_million"),
    "misspelt setting": {**VALID_LLM, "daily_cap": 150},
    "negative price": with_prices(output=-0.5),
    "misspelt price": with_prices(inputs=0.10),
}


@pytest.mark.parametrize("llm", BAD_LLM.values(), ids=BAD_LLM.keys())
def test_bad_llm_settings_are_refused(tmp_path, llm):
    with pytest.raises(ValidationError):
        load_app_settings(write(tmp_path, settings_text(llm=llm)))


def test_swapped_prices_are_explained(tmp_path):
    with pytest.raises(ValidationError, match="are the two swapped"):
        load_app_settings(write(tmp_path, settings_text(llm=with_prices(cached_input=0.20))))


@pytest.mark.parametrize(
    "llm",
    [
        {**VALID_LLM, "reasoning_effort": "none"},  # no thinking at all
        {**VALID_LLM, "network_retries": 0},  # never retry
        with_prices(input=0.10, cached_input=0.10),  # no discount for cached input
    ],
)
def test_edge_values_are_allowed(tmp_path, llm):
    assert load_app_settings(write(tmp_path, settings_text(llm=llm))).llm.model_dump() == llm
