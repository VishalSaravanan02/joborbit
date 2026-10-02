"""Tests for the Lever fetcher, using a real saved reply (tests/fixtures/lever.json)."""

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest
import respx

from joborbit.fetchers.lever import LeverFetcher, api_url_for
from joborbit.utils.http import FetchError, PoliteClient

pytestmark = pytest.mark.anyio

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "lever.json").read_text())


def make_client() -> PoliteClient:
    return PoliteClient(per_host_delay=0, retry_wait_seconds=0, block_private_addresses=False)


async def fetch_with_reply(reply, token: str = "palantir") -> list:
    respx.get(api_url_for(token), params={"mode": "json"}).respond(json=reply)
    async with make_client() as client:
        return await LeverFetcher().fetch(token, client)


def test_eu_tokens_use_the_eu_servers():
    assert api_url_for("palantir") == "https://api.lever.co/v0/postings/palantir"
    assert api_url_for("eu:examplecompany") == "https://api.eu.lever.co/v0/postings/examplecompany"


@respx.mock
async def test_fetches_every_job_in_the_reply():
    assert len(await fetch_with_reply(FIXTURE)) == len(FIXTURE)


@respx.mock
async def test_first_job_is_translated_correctly():
    job = (await fetch_with_reply(FIXTURE))[0]
    assert job.external_id == "6ed76ce8-4156-4b60-b120-403538bd66cd"
    assert job.title == "Administrative Business Partner"
    assert job.url == "https://jobs.lever.co/palantir/6ed76ce8-4156-4b60-b120-403538bd66cd"
    assert job.location_raw == "Singapore, Singapore"
    assert job.posted_at == datetime(2026, 8, 11, 17, 38, 11, 368000)  # from Unix milliseconds


@respx.mock
async def test_description_includes_every_list_section():
    """Requirements live in Lever's 'lists', so they must be part of the description."""
    jobs = await fetch_with_reply(FIXTURE)
    for raw, job in zip(FIXTURE, jobs, strict=True):
        for section in raw.get("lists") or []:
            assert section["text"] in job.description_html


@respx.mock
async def test_multiple_locations_are_combined():
    reply = copy.deepcopy(FIXTURE)
    reply[0]["categories"]["allLocations"] = ["London, United Kingdom", "Singapore, Singapore"]
    job = (await fetch_with_reply(reply))[0]
    assert job.location_raw == "London, United Kingdom / Singapore, Singapore"


@respx.mock
async def test_a_malformed_job_is_skipped_not_fatal():
    reply = copy.deepcopy(FIXTURE) + [{"text": "Job with no id"}]
    assert len(await fetch_with_reply(reply)) == len(FIXTURE)


@respx.mock
async def test_unexpected_reply_shape_is_a_fetch_error():
    with pytest.raises(FetchError, match="expected a list"):
        await fetch_with_reply({"ok": False})


@respx.mock
async def test_eu_company_is_fetched_from_the_eu_servers():
    jobs = await fetch_with_reply(FIXTURE, token="eu:palantir")
    assert len(jobs) == len(FIXTURE)
