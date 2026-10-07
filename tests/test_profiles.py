"""Tests for the profile rules (joborbit/profiles.py)."""

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.config import load_app_settings, load_roles
from joborbit.db.models import Base, Company, User, UserCompanyPref
from joborbit.db.session import create_sqlite_engine
from joborbit.profiles import FavouriteCompany, ProfileInput, UnknownCompaniesError, save_profile

LIMITS = load_app_settings().limits


def profile(**fields) -> ProfileInput:
    """A valid profile with only the required field filled in, plus any changes given."""
    return ProfileInput.model_validate({"countries": ["GB"], **fields})


def refused(match: str, **fields) -> None:
    with pytest.raises(ValidationError, match=match):
        profile(**fields)


# --- Defaults ------------------------------------------------------------------------------


def test_only_a_country_is_needed_and_the_defaults_are_sensible():
    result = profile()
    assert result.roles == [role.slug for role in load_roles()]  # all 15 default roles
    assert result.custom_roles == [] and result.languages == ["en"]
    assert (result.alert_style, result.include_internships, result.transfer_boost) == ("balanced", False, True)
    assert result.highest_degree is None and result.favourite_companies == []


def test_values_are_tidied():
    result = profile(
        roles=[" Data_Scientist "],
        custom_roles=["  Insights   Analyst "],
        countries=["gb"],
        languages=["EN", " es"],
        preferred_industries=["AI"],
        skills=["  Python ", "scikit-learn"],
        favourite_companies=[{"name": "  Monzo  ", "never_miss": True}],
        excluded_companies=[" Big   Bank "],
    )
    assert result.roles == ["data_scientist"]
    assert result.custom_roles == ["Insights Analyst"]  # custom text keeps its capitals
    assert result.countries == ["GB"] and result.languages == ["en", "es"]
    assert result.preferred_industries == ["ai"] and result.skills == ["Python", "scikit-learn"]
    assert result.favourite_companies == [FavouriteCompany(name="Monzo", never_miss=True)]
    assert result.excluded_companies == ["Big Bank"]


# --- Roles ----------------------------------------------------------------------------------


def test_custom_roles_alone_are_enough():
    assert profile(roles=[], custom_roles=["Insights Analyst"]).roles == []


def test_at_least_one_role_is_needed():
    refused("at least one role", roles=[], custom_roles=[])


def test_unknown_default_roles_are_refused():
    refused("unknown role 'data_wizard'", roles=["data_wizard"])


def test_the_role_limit_counts_default_and_custom_roles_together():
    customs = [f"Custom Role {n}" for n in range(LIMITS.max_roles_per_user - 15 + 1)]
    refused(f"at most {LIMITS.max_roles_per_user} roles in total", custom_roles=customs)
    assert len(profile(custom_roles=customs[:-1]).custom_roles) == LIMITS.max_roles_per_user - 15


def test_long_custom_role_names_are_refused():
    refused("longer than", custom_roles=["x" * (LIMITS.max_role_name_chars + 1)])


@pytest.mark.parametrize("custom", ["   ", "!!!", ""])
def test_custom_roles_need_a_letter_or_digit(custom):
    refused("at least one letter or digit", custom_roles=[custom])


def test_repeated_custom_roles_are_refused_ignoring_capitals():
    refused("listed twice", custom_roles=["Insights Analyst", "insights analyst"])


def test_a_custom_role_cannot_copy_a_default_role():
    refused("already a default role", custom_roles=["data scientist"])


def test_repeated_default_roles_are_refused():
    refused("listed twice", roles=["data_scientist", "Data_Scientist"])


# --- Countries and languages -------------------------------------------------------------


def test_at_least_one_country_is_needed():
    refused("at least one country", countries=[])


def test_a_country_that_is_not_switched_on_is_refused_with_a_clear_message():
    refused("IN is not switched on yet", countries=["GB", "IN"])


def test_an_unknown_country_is_refused():
    refused("unknown country 'XX'", countries=["XX"])


def test_country_order_is_kept_and_repeats_refused():
    assert profile(countries=["GB"]).countries == ["GB"]
    refused("listed twice", countries=["GB", "gb"])


def test_languages_must_be_known_and_at_least_one():
    refused("unknown language 'klingon'", languages=["klingon"])
    refused("at least one language", languages=[])


# --- Industries and skills ----------------------------------------------------------------


def test_unknown_industries_are_refused():
    refused("unknown industry 'fintek'", excluded_industries=["fintek"])


def test_an_industry_cannot_be_both_preferred_and_excluded():
    refused(
        "both preferred and excluded: gambling",
        preferred_industries=["gambling"],
        excluded_industries=["gambling"],
    )


def test_skill_limits():
    too_many = [f"skill {n}" for n in range(LIMITS.max_skills_per_user + 1)]
    refused(f"at most {LIMITS.max_skills_per_user} skills", skills=too_many)
    refused("longer than", skills=["x" * (LIMITS.max_skill_chars + 1)])
    refused("listed twice", skills=["Python", "python"])


# --- Choices ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), [("highest_degree", "diploma"), ("alert_style", "loud")])
def test_choices_must_be_from_the_list(field, value):
    with pytest.raises(ValidationError):
        profile(**{field: value})


def test_a_misspelt_field_name_is_refused():
    with pytest.raises(ValidationError):
        profile(langauges=["en"])


# --- Companies --------------------------------------------------------------------------------


def test_favourites_have_a_limit():
    favourites = [{"name": f"Company {n}"} for n in range(LIMITS.max_favourites_per_user + 1)]
    refused(f"at most {LIMITS.max_favourites_per_user} favourite companies", favourite_companies=favourites)


def test_the_same_favourite_twice_is_refused_however_it_is_written():
    refused("listed twice", favourite_companies=[{"name": "Checkout.com"}, {"name": "checkout com"}])


def test_a_company_cannot_be_both_favourite_and_excluded():
    refused(
        "both a favourite and excluded: Monzo",
        favourite_companies=[{"name": "Monzo"}],
        excluded_companies=["monzo"],
    )


def test_never_miss_belongs_only_to_favourites():
    """Excluded companies are plain names, so 'never miss' can't be set on one."""
    with pytest.raises(ValidationError):
        profile(excluded_companies=[{"name": "Monzo", "never_miss": True}])


def test_empty_company_names_are_refused():
    with pytest.raises(ValidationError):
        profile(favourite_companies=[{"name": "  "}])
    refused("empty", excluded_companies=["  "])


# --- Saving a profile --------------------------------------------------------------------------


@pytest.fixture
def session():
    """An empty in-memory database with three companies and one user without a profile."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        for name in ("Monzo", "Checkout.com", "Big Bank"):
            s.add(Company(name=name, slug=name.lower().replace(".", "-").replace(" ", "-"), size_category="medium"))
        s.add(User(telegram_id=1, display_name="Vishal"))
        s.commit()
        yield s


def user(session) -> User:
    return session.scalars(select(User)).one()


def choices(session) -> dict[str, tuple[str, bool]]:
    rows = session.scalars(select(UserCompanyPref))
    return {pref.company.name: (pref.kind, pref.never_miss) for pref in rows}


def test_saving_creates_the_profile_and_company_choices(session):
    save_profile(
        session,
        user(session),
        profile(
            languages=["en", "es"],
            favourite_companies=[{"name": "monzo", "never_miss": True}],
            excluded_companies=["Big Bank"],
        ),
    )
    session.commit()

    stored = user(session).profile
    assert stored.countries == ["GB"] and stored.languages == ["en", "es"] and len(stored.roles) == 15
    assert stored.weights is None
    assert choices(session) == {"Monzo": ("favourite", True), "Big Bank": ("excluded", False)}


def test_saving_again_makes_the_database_match_the_new_profile_exactly(session):
    first = profile(
        favourite_companies=[{"name": "Monzo", "never_miss": True}, {"name": "Checkout.com"}], skills=["Python"]
    )
    save_profile(session, user(session), first)
    session.commit()

    second = profile(
        favourite_companies=[{"name": "Monzo", "never_miss": False}],  # never miss switched off
        excluded_companies=["checkout com"],  # was a favourite, now excluded
        skills=[],  # emptied
        alert_style="more",
    )
    save_profile(session, user(session), second)
    session.commit()

    assert choices(session) == {"Monzo": ("favourite", False), "Checkout.com": ("excluded", False)}
    assert user(session).profile.skills == [] and user(session).profile.alert_style == "more"


def test_removed_company_choices_are_deleted(session):
    save_profile(session, user(session), profile(favourite_companies=[{"name": "Monzo"}]))
    session.commit()
    save_profile(session, user(session), profile())
    session.commit()
    assert choices(session) == {}


def test_personal_weights_are_left_alone(session):
    save_profile(session, user(session), profile())
    user(session).profile.weights = {"role_fit": 40}  # as if the weekly tuning had run
    session.commit()
    save_profile(session, user(session), profile(alert_style="fewer"))
    session.commit()
    assert user(session).profile.weights == {"role_fit": 40}


def test_unknown_companies_change_nothing(session):
    save_profile(session, user(session), profile(favourite_companies=[{"name": "Monzo"}], skills=["SQL"]))
    session.commit()

    with pytest.raises(UnknownCompaniesError) as error:
        save_profile(
            session,
            user(session),
            profile(favourite_companies=[{"name": "Monzo"}, {"name": "Monzoo"}], excluded_companies=["Nobody Ltd"]),
        )
    assert error.value.names == ["Monzoo", "Nobody Ltd"]
    assert user(session).profile.skills == ["SQL"]  # the earlier profile is untouched
    assert choices(session) == {"Monzo": ("favourite", False)}
