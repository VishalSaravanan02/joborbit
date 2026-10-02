"""Tests for the location parser, mostly using real location text from job feeds."""

import pytest

from joborbit.config import load_countries
from joborbit.pipeline.location import LocationResult, parse_location


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Real examples from the saved Greenhouse, Lever and Ashby replies
        ("Cardiff, London or Remote (UK)", ["GB"]),
        ("Barcelona", ["ES"]),
        ("Singapore, Singapore", ["SG"]),
        ("London, United Kingdom", ["GB"]),
        ("India", ["IN"]),
        # Other common ways of writing UK locations
        ("Remote (UK)", ["GB"]),
        ("Remote - UK", ["GB"]),
        ("Manchester, England", ["GB"]),
        ("Edinburgh, Scotland, United Kingdom", ["GB"]),
        ("Milton Keynes", ["GB"]),
        ("U.K.", ["GB"]),
        # Other target countries
        ("Bengaluru, Karnataka", ["IN"]),
        ("New Delhi", ["IN"]),
        ("Kuala Lumpur, Malaysia", ["MY"]),
        ("Ho Chi Minh City", ["VN"]),
        ("Hong Kong SAR", ["HK"]),
        ("Bangkok", ["TH"]),
        ("Madrid, Spain", ["ES"]),
        # Several countries in one job
        ("London / Singapore", ["GB", "SG"]),
        ("London, UK or Bengaluru, India", ["GB", "IN"]),
    ],
)
def test_finds_our_countries(text, expected):
    assert parse_location(text) == LocationResult(countries=expected, ambiguous=False)


@pytest.mark.parametrize(
    "text",
    ["", "   ", None, "Remote", "Fully Remote", "EMEA", "Europe", "APAC", "Worldwide", "Remote - EMEA", "Hybrid"],
)
def test_vague_locations_are_ambiguous(text):
    assert parse_location(text) == LocationResult(countries=[], ambiguous=True)


@pytest.mark.parametrize(
    "text",
    [
        # Real examples from the saved replies
        "Washington, D.C.",
        "United States / New York / San Francisco",
        # Other places that aren't our countries
        "Remote - US",
        "New York, NY",  # must not match the UK city of York
        "Sydney, New South Wales",  # must not match Wales
        "Berlin, Germany",
        "Toronto, Canada",
        "Sydney, Australia",
    ],
)
def test_other_countries_are_neither_ours_nor_ambiguous(text):
    assert parse_location(text) == LocationResult(countries=[], ambiguous=False)


@pytest.mark.parametrize("text", ["Ukulele Street", "Indiana", "Hongkonger Strasse", "Thailandia Plaza"])
def test_only_whole_words_match(text):
    """'UK' inside 'Ukulele' or 'India' inside 'Indiana' must not count."""
    assert parse_location(text).countries == []


def test_capitals_and_spacing_do_not_matter():
    assert parse_location("  LONDON,   united KINGDOM ").countries == ["GB"]


def test_every_country_in_the_config_is_loaded():
    codes = [country.code for country in load_countries()]
    assert codes == ["GB", "IN", "SG", "ES", "VN", "TH", "HK", "MY"]


def test_only_the_uk_is_switched_on_for_now():
    assert [country.code for country in load_countries() if country.enabled] == ["GB"]
