"""Tests for recognising a company's ATS from URLs and careers pages."""

import pytest
import respx

from joborbit.fetchers.ashby import API_URL as ASHBY_API
from joborbit.fetchers.detect import Detection, detect_ats, find_ats
from joborbit.fetchers.greenhouse import API_URL as GREENHOUSE_API
from joborbit.fetchers.lever import api_url_for
from joborbit.fetchers.registry import get_fetcher, is_supported
from joborbit.utils.http import PoliteClient


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Greenhouse
        ("https://job-boards.greenhouse.io/monzo/jobs/8143930", ("greenhouse", "monzo")),
        ("https://boards.greenhouse.io/examplecorp", ("greenhouse", "examplecorp")),
        ("https://job-boards.eu.greenhouse.io/example-eu", ("greenhouse", "example-eu")),
        ('<script src="https://boards.greenhouse.io/embed/job_board/js?for=acme"></script>', ("greenhouse", "acme")),
        ("https://boards-api.greenhouse.io/v1/boards/monzo/jobs", ("greenhouse", "monzo")),
        # Lever, including Lever's EU servers
        ("https://jobs.lever.co/palantir/6ed76ce8-4156-4b60-b120-403538bd66cd", ("lever", "palantir")),
        ("https://jobs.eu.lever.co/examplecompany", ("lever", "eu:examplecompany")),
        # Ashby
        ("https://jobs.ashbyhq.com/elevenlabs/a571b8e4-8176-4e31-aab6-2287ee810236", ("ashby", "elevenlabs")),
        # Recognised but not supported yet
        ("https://bigbank.wd3.myworkdayjobs.com/en-US/External_Careers", ("workday", "bigbank|wd3|External_Careers")),
        ("https://bigbank.wd103.myworkdayjobs.com/Careers/job/London/123", ("workday", "bigbank|wd103|Careers")),
        ("https://apply.workable.com/acme-ltd/", ("workable", "acme-ltd")),
        ("https://jobs.smartrecruiters.com/AcmeCorp", ("smartrecruiters", "AcmeCorp")),
        ("https://acme.recruitee.com/o/data-analyst", ("recruitee", "acme")),
        ("https://acme.jobs.personio.de/", ("personio", "acme")),
    ],
)
def test_recognises_ats_links(text, expected):
    assert find_ats(text) == expected


@pytest.mark.parametrize("text", ["https://www.example.com/careers", "<p>No links here</p>", ""])
def test_returns_none_when_no_ats_is_mentioned(text):
    assert find_ats(text) is None


def test_finds_the_ats_inside_a_careers_page():
    page = '<html><body><h1>Join us</h1><a href="https://jobs.lever.co/acme">See open roles</a></body></html>'
    assert find_ats(page) == ("lever", "acme")


def test_registry_knows_which_sites_are_supported():
    assert get_fetcher("greenhouse").ats_type == "greenhouse"
    assert is_supported("lever") and is_supported("ashby")
    assert not is_supported("workday")
    assert not is_supported(None)


# --- Full detection, with fake websites ---------------------------------------


def make_client() -> PoliteClient:
    return PoliteClient(per_host_delay=0, retry_wait_seconds=0, block_private_addresses=False)


@pytest.mark.anyio
@respx.mock
async def test_detects_from_the_url_alone_and_counts_jobs():
    respx.get(GREENHOUSE_API.format(token="monzo"), params={"content": "true"}).respond(
        json={"jobs": [{"id": 1, "title": "Data Analyst", "absolute_url": "https://example.com/1"}]}
    )
    async with make_client() as client:
        result = await detect_ats("https://job-boards.greenhouse.io/monzo", client)
    assert result == Detection("ready", "greenhouse", "monzo", job_count=1)


@pytest.mark.anyio
@respx.mock
async def test_detects_from_the_careers_page_html():
    respx.get("https://www.acme.com/careers").respond(text='<a href="https://jobs.ashbyhq.com/acme">Jobs</a>')
    respx.get(ASHBY_API.format(token="acme")).respond(json={"jobs": []})
    async with make_client() as client:
        result = await detect_ats("https://www.acme.com/careers", client)
    assert result == Detection("ready", "ashby", "acme", job_count=0)


@pytest.mark.anyio
@respx.mock
async def test_reports_unsupported_ats_without_fetching():
    async with make_client() as client:
        result = await detect_ats("https://bigbank.wd3.myworkdayjobs.com/en-US/External", client)
    assert result.status == "unsupported"
    assert result.ats_type == "workday"


@pytest.mark.anyio
@respx.mock
async def test_reports_not_found_when_page_has_no_ats():
    respx.get("https://www.acme.com/careers").respond(text="<p>Email us your CV</p>")
    async with make_client() as client:
        result = await detect_ats("https://www.acme.com/careers", client)
    assert result.status == "not_found"


@pytest.mark.anyio
@respx.mock
async def test_reports_failed_when_the_token_is_rejected():
    respx.get(api_url_for("ghost"), params={"mode": "json"}).respond(404)
    async with make_client() as client:
        result = await detect_ats("https://jobs.lever.co/ghost", client)
    assert result.status == "failed"
    assert "404" in result.message


@pytest.mark.anyio
@respx.mock
async def test_reports_failed_when_the_careers_page_is_unreachable():
    respx.get("https://www.acme.com/careers").respond(500)
    async with make_client() as client:
        result = await detect_ats("https://www.acme.com/careers", client)
    assert result.status == "failed"
