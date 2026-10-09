"""Tests for the score out of 100 (joborbit/matching/scoring.py), with the real settings.yaml values.

The sample job and user (tests/matching_samples.py) score 80: an exact role match (30), a
graduate role (25), half the job's skills (10) and the user's only country (15); no transfer.
"""

from datetime import datetime, timedelta

import pytest
from matching_samples import ACME, make_job, make_user

from joborbit.config import load_app_settings
from joborbit.matching.filters import Verdict
from joborbit.matching.scoring import (
    NOTE_COUNTRY_UNCLEAR,
    NOTE_EXPERIENCE_REQUIRED,
    is_evergreen,
    matched_skills,
    score_job,
    skill_words,
    skills_match,
)

CONFIG = load_app_settings().scoring
PASSED = Verdict(passed=True, reason="passed")
UNCLEAR = Verdict(passed=True, reason="passed", country_unclear=True)
LEVEL_REASONS = {"Graduate scheme", "Graduate role", "Entry level", "Internship"}


def score(job=None, user=None, verdict=PASSED):
    return score_job(job or make_job(), user or make_user(), verdict, CONFIG)


def points(part: str, job=None, user=None, verdict=PASSED) -> float:
    return score(job, user, verdict).points[part]


def test_the_build_plans_worked_example_scores_90():
    """A graduate data science scheme at a bank in London and Singapore, for a user ranking UK, India,
    Singapore with skills Python, SQL and PyTorch (build plan, section 9)."""
    job = make_job(
        is_graduate_scheme=True, company_industry="banking", company_countries=frozenset({"GB", "SG"}),
        skills=("Python", "SQL", "statistics", "machine learning"),
    )
    user = make_user(countries=("GB", "IN", "SG"), skills=("Python", "SQL", "PyTorch"))

    result = score(job, user)

    assert result.points == {"role_fit": 30, "entry_fit": 25, "skills": 10, "country": 15, "transfer": 10}
    assert result.score == 90
    assert result.reasons == ["Role: Data Scientist", "Graduate scheme", "London, your #1 country"]
    assert result.notes == [] and result.evergreen is False


def test_the_sample_job_scores_80():
    assert score().score == 80


# --- Role fit -------------------------------------------------------------------------------------


def test_an_exact_title_match_gives_full_role_fit():
    assert points("role_fit") == 30


def test_a_near_title_match_gives_less():
    assert points("role_fit", make_job(title="Graduate Data Scientst")) == pytest.approx(21)


def test_a_role_only_the_llm_found_gives_less():
    job = make_job(title="Graduate Decision Science Associate")
    result = score(job)
    assert result.points["role_fit"] == pytest.approx(21)
    assert result.reasons == ["Graduate role", "Role: Data Scientist", "London"]


def test_an_exact_match_of_any_of_the_users_roles_counts():
    """The LLM says data scientist; the title says data analyst, also one of the user's roles."""
    result = score(make_job(title="Junior Data Analyst"))
    assert result.points["role_fit"] == 30 and result.reasons[0] == "Role: Data Analyst"


def test_a_custom_role_in_the_title_gives_full_role_fit():
    user = make_user(roles=frozenset(), custom_roles=("Insights Analyst",))
    job = make_job(title="Insights Analyst", role_families=(), matched_custom_roles=frozenset({"insights analyst"}))
    result = score(job, user)
    assert result.points["role_fit"] == 30 and result.reasons[0] == "Role: Insights Analyst"


def test_a_custom_role_only_the_llm_found_gives_less():
    user = make_user(roles=frozenset(), custom_roles=("Insights Analyst",))
    job = make_job(
        title="Customer Insight Lead", role_families=(), matched_custom_roles=frozenset({"insights analyst"})
    )
    assert points("role_fit", job, user) == pytest.approx(21)


def test_a_never_miss_job_with_a_near_title_match_gets_less():
    """The LLM found none of the user's roles, but the title nearly matches one of them."""
    job = make_job(title="Graduate Data Scientst", role_families=("data_engineer",))
    result = score(job, make_user(never_miss_company_ids=frozenset({ACME})), Verdict(True, "passed", never_miss=True))
    assert result.points["role_fit"] == pytest.approx(21) and "Role: Data Scientist" in result.reasons


def test_a_never_miss_job_outside_the_users_roles_gets_no_role_fit():
    job = make_job(title="Software Engineer", role_families=("data_engineer",))
    result = score(job, make_user(never_miss_company_ids=frozenset({ACME})), Verdict(True, "passed", never_miss=True))
    assert result.points["role_fit"] == 0
    assert not any(reason.startswith("Role:") for reason in result.reasons)


# --- Entry fit ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seniority", "expected", "reason"),
    [
        ("graduate", 25, "Graduate role"),
        ("entry", 22.5, "Entry level"),
        ("intern", 22.5, "Internship"),
        (None, 15, None),
        ("mid", 6.25, None),
    ],
)
def test_entry_fit_follows_the_level(seniority, expected, reason):
    result = score(make_job(seniority=seniority))
    assert result.points["entry_fit"] == pytest.approx(expected)
    if reason:
        assert reason in result.reasons
    else:  # mid or no clue: nothing worth telling the user
        assert not LEVEL_REASONS & set(result.reasons)


def test_a_graduate_scheme_counts_as_graduate_whatever_the_level():
    result = score(make_job(is_graduate_scheme=True, seniority=None))
    assert result.points["entry_fit"] == 25 and "Graduate scheme" in result.reasons


def test_required_experience_without_a_number_caps_entry_fit():
    result = score(make_job(seniority="entry", experience_mandatory=True, experience_years=None))
    assert result.points["entry_fit"] == pytest.approx(12.5)
    assert "Entry level" not in result.reasons and result.notes == [NOTE_EXPERIENCE_REQUIRED]


def test_the_cap_never_raises_a_lower_level():
    result = score(make_job(seniority="mid", experience_mandatory=True, experience_years=None))
    assert result.points["entry_fit"] == pytest.approx(6.25) and result.notes == [NOTE_EXPERIENCE_REQUIRED]


@pytest.mark.parametrize(("mandatory", "years"), [(True, 0), (False, None), (False, 2)])
def test_no_cap_and_no_note_otherwise(mandatory, years):
    result = score(make_job(seniority="entry", experience_mandatory=mandatory, experience_years=years))
    assert result.points["entry_fit"] == pytest.approx(22.5) and result.notes == []


# --- Skills -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("skill", "expected"),
    [
        ("Data Visualisation", ("data", "visualization")),
        ("data visualization", ("data", "visualization")),
        ("LLMs", ("llm",)),
        ("Statistical modelling", ("statistical", "modeling")),
        ("scikit-learn", ("scikit", "learn")),
        ("C++", ("c++",)),
        ("C#", ("c#",)),
        ("AWS", ("aws",)),
        ("Pandas", ("panda",)),
        ("A/B testing", ("a", "b", "testing")),
        ("Business analysis", ("business", "analysis")),
        ("Analyse data", ("analyze", "data")),
    ],
)
def test_skills_are_written_in_one_standard_form(skill, expected):
    assert skill_words(skill) == expected


@pytest.mark.parametrize(
    ("job_skill", "user_skill"),
    [
        ("PyTorch", "deep learning in PyTorch"),
        ("LLM", "LLMs, RAG and agentic AI systems"),
        ("data visualization", "dashboards and data visualisation"),
        ("Python programming", "Python"),
        ("SQL", "sql"),
    ],
)
def test_skills_match_when_either_contains_the_other(job_skill, user_skill):
    assert skills_match(job_skill, user_skill)


@pytest.mark.parametrize(
    ("job_skill", "user_skill"),
    [
        ("Java", "JavaScript"),
        ("C", "C++"),
        ("machine learning", "learning machine"),
        ("A/B testing", "statistical testing"),
        ("", "Python"),
    ],
)
def test_different_skills_dont_match(job_skill, user_skill):
    assert not skills_match(job_skill, user_skill)


def test_matched_skills_keep_the_jobs_wording_once_each():
    found = matched_skills(("Python", "python", "SQL", "Tableau"), ("Python", "SQL and relational data modelling"))
    assert found == ["Python", "SQL"]


def test_skills_give_the_share_of_the_jobs_skills_the_user_has():
    job = make_job(skills=("Python", "SQL", "Spark", "dbt"), seniority="mid")  # mid: the skills reason makes the top 3
    result = score(job, make_user(skills=("python", "sql", "Excel")))
    assert result.points["skills"] == 10
    assert "2 of your skills: Python, SQL" in result.reasons


def test_at_most_six_of_the_jobs_skills_are_counted():
    job = make_job(skills=("Python", "SQL", "Spark", "dbt", "Airflow", "Kafka", "AWS", "Docker", "Git", "Linux"))
    assert points("skills", job, make_user(skills=("Python", "SQL", "Spark"))) == 10  # 3 of 6


def test_skills_never_give_more_than_their_weight():
    job = make_job(skills=("Python", "SQL", "Spark", "dbt", "Airflow", "Kafka", "AWS", "Docker"))
    user = make_user(skills=("Python", "SQL", "Spark", "dbt", "Airflow", "Kafka", "AWS"))
    assert points("skills", job, user) == 20


def test_a_repeated_job_skill_counts_once():
    assert points("skills", make_job(skills=("Python", "python", "SQL"))) == 10  # 1 of 2


@pytest.mark.parametrize(("job_skills", "user_skills"), [((), ("Python",)), (("Python",), ())])
def test_no_skills_on_either_side_is_neutral(job_skills, user_skills):
    result = score(make_job(skills=job_skills), make_user(skills=user_skills))
    assert result.points["skills"] == 10
    assert not any("of your skills" in reason for reason in result.reasons)


def test_no_shared_skills_give_nothing():
    assert points("skills", make_job(skills=("Java",))) == 0


def test_the_skills_reason_names_at_most_three():
    skills = ("Python", "SQL", "Spark", "dbt")
    result = score(make_job(skills=skills), make_user(skills=skills))
    assert result.points["skills"] == 20
    assert "4 of your skills: Python, SQL, Spark" in result.reasons


# --- Country --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("country", "expected", "reason"),
    [
        ("GB", 15, "United Kingdom, your #1 country"),
        ("IN", 11.25, "India, your #2 country"),
        ("SG", 7.5, None),  # fewer points than the skills (10), so not among the top three reasons
    ],
)
def test_country_falls_evenly_down_the_users_list(country, expected, reason):
    job = make_job(countries=frozenset({country}), cities=())
    result = score(job, make_user(countries=("GB", "IN", "SG")))
    assert result.points["country"] == pytest.approx(expected)
    if reason:
        assert reason in result.reasons
    else:
        assert not any("country" in text for text in result.reasons)


def test_the_jobs_best_country_counts():
    job = make_job(countries=frozenset({"SG", "IN"}), cities=())
    assert points("country", job, make_user(countries=("GB", "IN", "SG"))) == pytest.approx(11.25)


def test_a_single_country_list_gives_full_marks():
    assert points("country") == 15


def test_the_country_reason_names_the_city_when_there_is_one_country():
    assert "London" in score().reasons


def test_a_user_with_one_country_isnt_told_its_rank():
    """"Your #1 country" only means something when there are others."""
    assert "United Kingdom" in score(make_job(cities=())).reasons
    two = score(make_job(cities=()), make_user(countries=("GB", "IN"))).reasons
    assert "United Kingdom, your #1 country" in two


def test_the_country_reason_names_the_country_when_the_job_is_in_several():
    job = make_job(countries=frozenset({"GB", "IN"}))
    assert "United Kingdom, your #1 country" in score(job, make_user(countries=("GB", "IN"))).reasons


def test_an_unclear_country_scores_half_and_says_so():
    result = score(make_job(countries=frozenset(), location_vague=True, cities=()), verdict=UNCLEAR)
    assert result.points["country"] == 7.5
    assert result.notes == [NOTE_COUNTRY_UNCLEAR]
    assert not any("country" in reason for reason in result.reasons)


# --- Transfer -------------------------------------------------------------------------------------


def test_transfer_needs_a_top_two_country_and_another_of_the_users():
    job = make_job(company_countries=frozenset({"GB", "SG"}), seniority="mid", skills=("Java",))
    result = score(job, make_user(countries=("GB", "IN", "SG")))
    assert result.points["transfer"] == 10
    assert result.reasons == ["Role: Data Scientist", "London, your #1 country", "Company also in Singapore"]


def test_no_transfer_when_the_user_switched_it_off():
    job = make_job(company_countries=frozenset({"GB", "SG"}))
    assert points("transfer", job, make_user(countries=("GB", "IN", "SG"), transfer_boost=False)) == 0


def test_no_transfer_with_a_single_country():
    assert points("transfer", make_job(company_countries=frozenset({"GB", "IN", "SG"}))) == 0


def test_no_transfer_when_neither_shared_country_is_in_the_top_two():
    job = make_job(countries=frozenset({"SG"}), company_countries=frozenset({"SG", "ES"}))
    assert points("transfer", job, make_user(countries=("GB", "IN", "SG", "ES"))) == 0


def test_no_transfer_when_only_one_of_the_users_countries_is_shared():
    job = make_job(company_countries=frozenset({"GB", "US"}))
    assert points("transfer", job, make_user(countries=("GB", "IN"))) == 0


# --- Bonuses, weights and the total --------------------------------------------------------------


def test_a_favourite_company_adds_its_bonus():
    result = score(user=make_user(favourite_company_ids=frozenset({ACME})))
    assert result.points["favourite_company"] == 15 and result.score == 95


def test_a_preferred_industry_adds_its_bonus():
    result = score(user=make_user(preferred_industries=frozenset({"fintech"})))
    assert result.points["preferred_industry"] == 5 and result.score == 85


def test_the_preferred_industry_reason_uses_its_name():
    job = make_job(seniority="mid", skills=("Java",))
    result = score(job, make_user(preferred_industries=frozenset({"fintech"})))
    assert result.reasons == ["Role: Data Scientist", "London", "Preferred industry: Fintech"]


def test_bonuses_not_earned_are_not_listed():
    assert set(score().points) == {"role_fit", "entry_fit", "skills", "country", "transfer"}


def test_the_score_never_goes_above_100():
    user = make_user(favourite_company_ids=frozenset({ACME}), preferred_industries=frozenset({"fintech"}))
    assert score(make_job(skills=("Python",)), user).score == 100  # 90 + 15 + 5


def test_the_score_is_rounded_half_up():
    """21 + 22.5 + 10 + 15 = 68.5, which rounds to 69 (Python's round() would give 68)."""
    job = make_job(title="Graduate Data Scientst", seniority="entry")
    assert score(job).score == 69


def test_personal_weights_replace_the_defaults():
    user = make_user(weights={"skills": 40, "transfer": 0, "not_a_part": 99})
    result = score(user=user)
    assert result.points["skills"] == 20  # 0.5 x 40
    assert set(result.points) == {"role_fit", "entry_fit", "skills", "country", "transfer"}


# --- Reasons and notes -------------------------------------------------------------------------------


def test_at_most_three_reasons_with_the_most_points_first():
    """Role 30, country 15, favourite 15, skills 10, industry 5: the top three, in that order."""
    user = make_user(favourite_company_ids=frozenset({ACME}), preferred_industries=frozenset({"fintech"}))
    reasons = score(make_job(seniority="mid"), user).reasons
    assert reasons == ["Role: Data Scientist", "London", "Favourite company"]


def test_a_part_with_a_personal_weight_of_0_is_never_a_reason():
    """Only the role and the country are worth telling here; the role gave 0 points."""
    job = make_job(seniority="mid", skills=("Java",))
    result = score(job, make_user(weights={"role_fit": 0, "country": 45}))
    assert result.points["role_fit"] == 0
    assert result.reasons == ["London"]


def test_parts_that_gave_no_points_are_never_reasons():
    reasons = score(make_job(skills=("Java",), seniority="mid", title="Decision Scientist")).reasons
    assert not any("of your skills" in reason for reason in reasons)


# --- Evergreen ads ---------------------------------------------------------------------------------


NOW = datetime(2026, 10, 9, 9, 0)


def test_an_ad_posted_long_before_we_saw_it_is_evergreen():
    job = make_job(posted_at=datetime(2022, 1, 14), became_new_at=NOW)
    result = score(job)
    assert result.evergreen is True and result.notes == ["Originally posted Jan 2022"]
    assert result.score == 80  # scored as normal; the router keeps it out of instant alerts


@pytest.mark.parametrize("age", [timedelta(days=30), timedelta(days=1), timedelta(0)])
def test_a_recent_ad_is_not_evergreen(age):
    assert not is_evergreen(make_job(posted_at=NOW - age, became_new_at=NOW), CONFIG)


def test_the_age_counts_from_when_the_job_last_became_new():
    """A job back after a long gap became new again: an old posting date then means an old ad."""
    job = make_job(posted_at=datetime(2026, 8, 1), became_new_at=NOW)
    assert is_evergreen(job, CONFIG)
    assert not is_evergreen(make_job(posted_at=datetime(2026, 8, 1), became_new_at=datetime(2026, 8, 20)), CONFIG)


def test_a_job_with_no_posting_date_is_not_evergreen():
    assert score(make_job(posted_at=None)).evergreen is False


def test_notes_come_in_a_fixed_order():
    job = make_job(
        countries=frozenset(), location_vague=True, experience_mandatory=True, experience_years=None,
        posted_at=datetime(2025, 3, 2), became_new_at=NOW,
    )
    assert score(job, verdict=UNCLEAR).notes == [
        NOTE_COUNTRY_UNCLEAR, NOTE_EXPERIENCE_REQUIRED, "Originally posted Mar 2025",
    ]
