"""Tests for the shared fetcher template and job format."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from joborbit.fetchers.base import Fetcher, NormalisedJob, parse_timestamp


def test_job_text_is_tidied():
    job = NormalisedJob(
        external_id=12345,
        title="  Graduate   Data\nScientist ",
        url="https://example.com/jobs/12345",
        location_raw=" London,   UK ",
    )
    assert job.external_id == "12345"
    assert job.title == "Graduate Data Scientist"
    assert job.location_raw == "London, UK"


def test_optional_fields_have_sensible_defaults():
    job = NormalisedJob(external_id="a1", title="Data Analyst", url="https://example.com/a1")
    assert job.location_raw == ""
    assert job.description_html is None
    assert job.posted_at is None


@pytest.mark.parametrize("missing", ["external_id", "title", "url"])
def test_required_fields_cannot_be_empty(missing):
    data = {"external_id": "a1", "title": "Data Analyst", "url": "https://example.com/a1"}
    data[missing] = "   " if missing != "url" else ""
    with pytest.raises(ValidationError):
        NormalisedJob(**data)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-30T10:15:00Z", datetime(2026, 9, 30, 10, 15)),
        ("2026-09-30T11:15:00+01:00", datetime(2026, 9, 30, 10, 15)),  # converted to UTC
        ("2026-09-30T10:15:00", datetime(2026, 9, 30, 10, 15)),  # no zone: assumed UTC
        (1790763300, datetime(2026, 9, 30, 10, 15)),  # Unix seconds
        (1790763300000, datetime(2026, 9, 30, 10, 15)),  # Unix milliseconds (Lever)
    ],
)
def test_parse_timestamp_understands_common_formats(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", [None, "", "not a date", "2026-13-45"])
def test_parse_timestamp_returns_none_for_unusable_values(value):
    assert parse_timestamp(value) is None


def test_a_fetcher_must_implement_fetch():
    class Incomplete(Fetcher):
        ats_type = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()
