"""Tests for the Greenhouse fetcher, using a real saved reply (tests/fixtures/greenhouse.json)."""

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest
import respx
from click import clear

from joborbit.fetchers.greenhouse import API_URL, GreenhouseFetcher
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
