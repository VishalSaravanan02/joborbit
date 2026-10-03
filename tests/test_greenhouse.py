"""Tests for the Greenhouse fetcher, using a real saved reply (tests/fixtures/greenhouse.json)."""

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest
import respx

from joborbit.fetchers.greenhouse import API_URL, GreenhouseFetcher
from joborbit.pipeline.location import parse_location
from joborbit.utils.http import FetchError, HTTPStatusFetchError, PoliteClient

pytestmark = pytest.mark.anyio

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "greenhouse.json").read_text())
FEED_URL = API_URL.format(token="monzo")


def make_client() -> PoliteClient:
    return PoliteClient(per_host_delay=0, retry_wait_seconds=0, block_private_addresses=False)


async def fetch_with_reply(reply) -> list:
    respx.get(FEED_URL, params={"content": "true"}).respond(json=reply)
    async with make_client() as client:
        return await GreenhouseFetcher().fetch("monzo", client)


@respx.mock
async def test_fetches_every_job_in_the_reply():
    jobs = await fetch_with_reply(FIXTURE)
    assert len(jobs) == len(FIXTURE["jobs"])


@respx.mock
async def test_first_job_is_translated_correctly():
    job = (await fetch_with_reply(FIXTURE))[0]
    assert job.external_id == "8143930"
    assert job.title == "Anaplan Support Analyst"
    assert job.url == "https://job-boards.greenhouse.io/monzo/jobs/8143930"
    assert job.location_raw == "Cardiff, London or Remote (UK)"
    assert job.posted_at == datetime(2026, 8, 24, 13, 4, 42)  # 09:04:42 at UTC-4, in UTC


@respx.mock
async def test_descriptions_are_unescaped_into_real_html():
    for job in await fetch_with_reply(FIXTURE):
        assert job.description_html is not None
        assert "&lt;" not in job.description_html
        assert "<" in job.description_html


@respx.mock
async def test_every_job_has_the_essentials():
    for job in await fetch_with_reply(FIXTURE):
        assert job.external_id and job.title and job.location_raw
        assert job.url.startswith("https://")


@respx.mock
async def test_falls_back_to_updated_at_when_first_published_is_missing():
    reply = copy.deepcopy(FIXTURE)
    del reply["jobs"][0]["first_published"]
    job = (await fetch_with_reply(reply))[0]
    assert job.posted_at == datetime(2026, 9, 25, 9, 17, 9)  # updated_at, in UTC


@respx.mock
async def test_a_malformed_job_is_skipped_not_fatal():
    reply = copy.deepcopy(FIXTURE)
    reply["jobs"].append({"title": "Job with no id"})
    jobs = await fetch_with_reply(reply)
    assert len(jobs) == len(FIXTURE["jobs"])  # the broken one was skipped


@respx.mock
async def test_unexpected_reply_shape_is_a_fetch_error():
    with pytest.raises(FetchError, match="no 'jobs' list"):
        await fetch_with_reply({"error": "something else"})


@respx.mock
async def test_unknown_company_gives_a_404_error():
    respx.get(FEED_URL, params={"content": "true"}).respond(404)
    async with make_client() as client:
        with pytest.raises(HTTPStatusFetchError):
            await GreenhouseFetcher().fetch("monzo", client)


# --- Locations: the location field plus the offices list ------------------------------


def location_of(location, offices) -> str:
    raw = {"id": 1, "title": "Data Analyst", "absolute_url": "https://e.com/1", "location": location}
    if offices is not None:
        raw["offices"] = offices
    return GreenhouseFetcher().parse_job(raw).location_raw


def test_offices_are_added_when_the_location_is_only_a_way_of_working():
    """Cloudflare-style: the location says "Hybrid"; the cities are only in offices."""
    offices = [{"name": "Austin, TX", "location": None}, {"name": "London, United Kingdom", "location": None}]
    text = location_of({"name": "Hybrid"}, offices)
    assert text == "Hybrid / Austin, TX / London, United Kingdom"
    assert parse_location(text).countries == ["GB"]


def test_an_office_already_in_the_location_is_not_repeated():
    """Monzo-style: the office "London" is already in the location text."""
    text = location_of({"name": "Cardiff, London or Remote (UK)"}, [{"name": "London"}])
    assert text == "Cardiff, London or Remote (UK)"


def test_the_same_office_twice_is_listed_once():
    text = location_of({"name": "Hybrid"}, [{"name": "Singapore"}, {"name": "singapore"}])
    assert text == "Hybrid / Singapore"


def test_offices_are_used_when_there_is_no_location():
    assert location_of(None, [{"name": "London, United Kingdom"}]) == "London, United Kingdom"
    assert location_of({}, [{"name": "Remote India"}]) == "Remote India"


def test_missing_or_odd_offices_are_ignored():
    assert location_of({"name": "London"}, None) == "London"
    assert location_of({"name": "London"}, []) == "London"
    assert location_of({"name": "London"}, [None, "Paris", {"name": None}, {"name": "  "}]) == "London"


def test_real_monzo_locations_are_unchanged():
    """Our saved Monzo reply has offices on every job; none of its locations should change."""
    jobs = GreenhouseFetcher().parse_jobs(FIXTURE["jobs"], "monzo")
    assert [job.location_raw for job in jobs] == [
        (raw.get("location") or {}).get("name") for raw in FIXTURE["jobs"]
    ]
