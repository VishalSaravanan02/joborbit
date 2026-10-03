"""Tests for recognising a company's ATS from URLs and careers pages."""

import httpx
import pytest
import respx

from joborbit.fetchers.ashby import API_URL as ASHBY_API
from joborbit.fetchers.detect import Detection, detect_ats, find_ats, name_tokens
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


def no_boards_anywhere(*tokens: str) -> None:
    """Make all three supported ATSs answer "not found" for these tokens."""
    for token in tokens:
        respx.get(GREENHOUSE_API.format(token=token), params={"content": "true"}).respond(404)
        respx.get(api_url_for(token), params={"mode": "json"}).respond(404)
        respx.get(ASHBY_API.format(token=token)).respond(404)


def one_job_board(url: str) -> dict:
    """A tiny reply in each ATS's shape, holding one job that links to `url`."""
    return {
        "greenhouse": {"jobs": [{"id": 1, "title": "Data Analyst", "absolute_url": url}]},
        "lever": [{"id": "a1", "text": "Data Analyst", "hostedUrl": url}],
        "ashby": {"jobs": [{"id": "b1", "title": "Data Analyst", "jobUrl": url}]},
    }


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
async def test_reports_failed_when_the_token_is_rejected_and_probing_finds_nothing():
    respx.get(api_url_for("ghost"), params={"mode": "json"}).respond(404)
    no_boards_anywhere("ghost")
    async with make_client() as client:
        result = await detect_ats("https://jobs.lever.co/ghost", client)
    assert result.status == "failed"
    assert "404" in result.message  # the original, useful error is kept


@pytest.mark.anyio
@respx.mock
async def test_reports_failed_when_the_careers_page_is_unreachable():
    respx.get("https://www.acme.com/careers").respond(500)
    async with make_client() as client:
        result = await detect_ats("https://www.acme.com/careers", client)
    assert result.status == "failed"


# --- Probing: trying every supported ATS when nothing else works --------------------


def test_name_tokens_are_made_from_the_company_name():
    assert name_tokens("Google DeepMind") == ["google-deepmind", "googledeepmind"]
    assert name_tokens("Checkout.com") == ["checkout-com", "checkoutcom"]
    assert name_tokens("Monzo") == ["monzo"]  # one word: one token
    assert name_tokens(None) == [] and name_tokens("") == []


@pytest.mark.anyio
@respx.mock
async def test_a_wrong_ats_guess_is_corrected_by_probing():
    """The careers link says Lever, but the company is really on Ashby."""
    respx.get(api_url_for("acme"), params={"mode": "json"}).respond(404)
    respx.get(GREENHOUSE_API.format(token="acme"), params={"content": "true"}).respond(404)
    respx.get(ASHBY_API.format(token="acme")).respond(json=one_job_board("https://jobs.ashbyhq.com/acme/1")["ashby"])
    async with make_client() as client:
        result = await detect_ats("https://jobs.lever.co/acme", client)
    assert result.status == "probed"
    assert (result.ats_type, result.token, result.job_count) == ("ashby", "acme", 1)
    assert result.sample_url == "https://jobs.ashbyhq.com/acme/1"
    assert "confirm" in result.message


@pytest.mark.anyio
@respx.mock
async def test_probes_by_name_when_the_careers_page_has_no_ats_link():
    respx.get("https://www.acme.com/careers").respond(text="<div id='app'></div>")  # built by JavaScript
    no_boards_anywhere("acme-labs")
    respx.get(GREENHOUSE_API.format(token="acmelabs"), params={"content": "true"}).respond(
        json=one_job_board("https://job-boards.greenhouse.io/acmelabs/jobs/1")["greenhouse"]
    )
    respx.get(api_url_for("acmelabs"), params={"mode": "json"}).respond(404)
    respx.get(ASHBY_API.format(token="acmelabs")).respond(404)
    async with make_client() as client:
        result = await detect_ats("https://www.acme.com/careers", client, name="Acme Labs")
    assert result.status == "probed"
    assert (result.ats_type, result.token) == ("greenhouse", "acmelabs")


@pytest.mark.anyio
@respx.mock
async def test_probes_by_name_when_there_is_no_careers_url():
    no_boards_anywhere("acme")
    respx.get(api_url_for("acme"), params={"mode": "json"}).respond(
        json=one_job_board("https://jobs.lever.co/acme/a1")["lever"]
    )
    async with make_client() as client:
        result = await detect_ats(None, client, name="Acme")
    assert (result.status, result.ats_type, result.token) == ("probed", "lever", "acme")


@pytest.mark.anyio
@respx.mock
async def test_several_probe_matches_are_ambiguous_and_not_guessed():
    respx.get(GREENHOUSE_API.format(token="acme"), params={"content": "true"}).respond(
        json=one_job_board("https://job-boards.greenhouse.io/acme/jobs/1")["greenhouse"]
    )
    respx.get(api_url_for("acme"), params={"mode": "json"}).respond(
        json=one_job_board("https://jobs.lever.co/acme/a1")["lever"]
    )
    respx.get(ASHBY_API.format(token="acme")).respond(404)
    async with make_client() as client:
        result = await detect_ats(None, client, name="Acme")
    assert result.status == "ambiguous"
    assert result.ats_type is None and result.token is None
    assert {(m.ats_type, m.token) for m in result.matches} == {("greenhouse", "acme"), ("lever", "acme")}
    assert "greenhouse/acme" in result.message and "lever/acme" in result.message


@pytest.mark.anyio
@respx.mock
async def test_a_board_with_no_jobs_does_not_count_as_a_probe_match():
    respx.get(GREENHOUSE_API.format(token="acme"), params={"content": "true"}).respond(json={"jobs": []})
    respx.get(api_url_for("acme"), params={"mode": "json"}).respond(404)
    respx.get(ASHBY_API.format(token="acme")).respond(404)
    async with make_client() as client:
        result = await detect_ats(None, client, name="Acme")
    assert result.status == "not_found"


@pytest.mark.anyio
@respx.mock
async def test_an_eu_lever_guess_is_also_probed_on_the_eu_servers():
    respx.get(api_url_for("eu:acme"), params={"mode": "json"}).mock(
        side_effect=[httpx.Response(404), httpx.Response(200, json=one_job_board("https://jobs.eu.lever.co/acme/a1")["lever"])]
    )
    no_boards_anywhere("acme")
    respx.get(GREENHOUSE_API.format(token="eu:acme"), params={"content": "true"}).respond(404)
    respx.get(ASHBY_API.format(token="eu:acme")).respond(404)
    async with make_client() as client:
        result = await detect_ats("https://jobs.eu.lever.co/acme", client)
    assert (result.status, result.ats_type, result.token) == ("probed", "lever", "eu:acme")


@pytest.mark.anyio
@respx.mock
async def test_a_working_link_on_the_careers_page_is_trusted_without_probing():
    respx.get("https://www.acme.com/careers").respond(text='<a href="https://jobs.lever.co/acme">Jobs</a>')
    respx.get(api_url_for("acme"), params={"mode": "json"}).respond(
        json=one_job_board("https://jobs.lever.co/acme/a1")["lever"]
    )
    async with make_client() as client:  # any probe request would fail: nothing else is faked
        result = await detect_ats("https://www.acme.com/careers", client, name="Acme")
    assert (result.status, result.ats_type) == ("ready", "lever")


@pytest.mark.anyio
@respx.mock
async def test_an_unsupported_ats_is_reported_without_probing():
    async with make_client() as client:
        result = await detect_ats("https://bigbank.wd3.myworkdayjobs.com/en-US/External", client, name="Big Bank")
    assert result.status == "unsupported"
