"""Tests for the Ashby fetcher, using a real saved reply (tests/fixtures/ashby.json)."""

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest
import respx

from joborbit.fetchers.ashby import API_URL, AshbyFetcher
from joborbit.utils.http import FetchError, PoliteClient

pytestmark = pytest.mark.anyio

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ashby.json").read_text())
FEED_URL = API_URL.format(token="elevenlabs")


def make_client() -> PoliteClient:
    return PoliteClient(per_host_delay=0, retry_wait_seconds=0, block_private_addresses=False)


async def fetch_with_reply(reply) -> list:
    respx.get(FEED_URL).respond(json=reply)
    async with make_client() as client:
        return await AshbyFetcher().fetch("elevenlabs", client)


@respx.mock
async def test_fetches_every_listed_job():
    assert len(await fetch_with_reply(FIXTURE)) == len(FIXTURE["jobs"])


@respx.mock
async def test_first_job_is_translated_correctly():
    job = (await fetch_with_reply(FIXTURE))[0]
    assert job.external_id == "a571b8e4-8176-4e31-aab6-2287ee810236"
    assert job.title == "Account Manager - India"  # trailing space tidied away
    assert job.url == "https://jobs.ashbyhq.com/elevenlabs/a571b8e4-8176-4e31-aab6-2287ee810236"
    assert job.location_raw == "India"
    assert job.posted_at == datetime(2026, 7, 21, 16, 3, 51, 100000)
    assert job.description_html.startswith("<h2>About ElevenLabs</h2>")


@respx.mock
async def test_unlisted_jobs_are_skipped():
    reply = copy.deepcopy(FIXTURE)
    reply["jobs"][0]["isListed"] = False
    assert len(await fetch_with_reply(reply)) == len(FIXTURE["jobs"]) - 1


@respx.mock
async def test_secondary_locations_are_added():
    reply = copy.deepcopy(FIXTURE)
    reply["jobs"][0]["location"] = "London"
    reply["jobs"][0]["secondaryLocations"] = [{"location": "Singapore"}, {"location": "London"}]
    job = (await fetch_with_reply(reply))[0]
    assert job.location_raw == "London / Singapore"  # duplicates removed


@respx.mock
async def test_a_malformed_job_is_skipped_not_fatal():
    reply = copy.deepcopy(FIXTURE)
    reply["jobs"].append({"title": "Job with no id", "isListed": True})
    assert len(await fetch_with_reply(reply)) == len(FIXTURE["jobs"])


@respx.mock
async def test_unexpected_reply_shape_is_a_fetch_error():
    with pytest.raises(FetchError, match="no 'jobs' list"):
        await fetch_with_reply([])


@respx.mock
async def test_an_entry_that_is_not_a_job_is_skipped_not_fatal():
    """Anything in the list that isn't a job object must be skipped, not crash the fetch."""
    reply = copy.deepcopy(FIXTURE)
    reply["jobs"].append("not a job")
    assert len(await fetch_with_reply(reply)) == len(FIXTURE["jobs"])
