"""Tests for the polite web client. No real websites are contacted: respx fakes them."""

import gzip
import json
import time

import httpx
import pytest
import respx

from joborbit.utils.http import (
    BlockedURLError,
    FetchError,
    HTTPStatusFetchError,
    PoliteClient,
    RateLimitedError,
    ResponseTooLargeError,
    ensure_public_url,
)

pytestmark = pytest.mark.anyio  # lets these tests use async/await

URL = "https://jobs.example.com/api"


def make_client(**overrides) -> PoliteClient:
    """A client with no waiting, so tests run instantly."""
    options = {"per_host_delay": 0, "retry_wait_seconds": 0, "block_private_addresses": False}
    options.update(overrides)
    return PoliteClient(**options)


@respx.mock
async def test_get_json_returns_parsed_data():
    respx.get(URL).respond(json={"jobs": [{"id": 1}]})
    async with make_client() as client:
        assert await client.get_json(URL) == {"jobs": [{"id": 1}]}

@respx.mock
async def test_compressed_responses_are_unpacked_once():
    """Real sites send gzip-compressed data; it must be decompressed exactly once."""
    payload = json.dumps({"jobs": [{"id": 1}]}).encode()
    respx.get(URL).respond(content=gzip.compress(payload), headers={"Content-Encoding": "gzip"})
    async with make_client() as client:
        assert await client.get_json(URL) == {"jobs": [{"id": 1}]}


@respx.mock
async def test_corrupt_response_becomes_a_fetch_error():
    """A reply that claims to be compressed but isn't must not crash with a raw httpx error."""
    corrupt = httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"not gzip"))
    respx.get(URL).mock(return_value=corrupt)
    async with make_client() as client:
        with pytest.raises(FetchError, match="Bad response"):
            await client.get_json(URL)


@respx.mock
async def test_sends_a_clear_user_agent():
    route = respx.get(URL).respond(json={})
    async with make_client() as client:
        await client.get_json(URL)
    assert route.calls.last.request.headers["User-Agent"].startswith("JobOrbit personal job alerts")


@respx.mock
async def test_server_errors_are_retried_then_succeed():
    route = respx.get(URL)
    route.side_effect = [httpx.Response(503), httpx.Response(502), httpx.Response(200, json={"ok": True})]
    async with make_client() as client:
        assert await client.get_json(URL) == {"ok": True}
    assert route.call_count == 3


@respx.mock
async def test_gives_up_after_three_server_errors():
    route = respx.get(URL).respond(500)
    async with make_client() as client:
        with pytest.raises(FetchError, match="Gave up after 3 attempts"):
            await client.get_json(URL)
    assert route.call_count == 3


@respx.mock
async def test_timeouts_are_retried():
    route = respx.get(URL)
    route.side_effect = [httpx.ConnectTimeout("slow"), httpx.Response(200, json={"ok": True})]
    async with make_client() as client:
        assert await client.get_json(URL) == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_connection_errors_are_retried():
    route = respx.get(URL)
    route.side_effect = [httpx.ConnectError("connection refused"), httpx.Response(200, json={"ok": True})]
    async with make_client() as client:
        assert await client.get_json(URL) == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_not_found_is_not_retried():
    route = respx.get(URL).respond(404)
    async with make_client() as client:
        with pytest.raises(HTTPStatusFetchError) as error:
            await client.get_json(URL)
    assert error.value.status_code == 404
    assert route.call_count == 1


@respx.mock
async def test_rate_limit_stops_immediately_and_reports_retry_after():
    route = respx.get(URL).respond(429, headers={"Retry-After": "120"})
    async with make_client() as client:
        with pytest.raises(RateLimitedError) as error:
            await client.get_json(URL)
    assert error.value.retry_after == 120
    assert route.call_count == 1


@respx.mock
async def test_huge_responses_are_refused():
    respx.get(URL).respond(content=b"x" * 2000)
    async with make_client(max_bytes=1000) as client:
        with pytest.raises(ResponseTooLargeError):
            await client.get_text(URL)


@respx.mock
async def test_non_json_reply_gives_a_clear_error():
    respx.get(URL).respond(text="<html>not json</html>")
    async with make_client() as client:
        with pytest.raises(FetchError, match="Expected JSON"):
            await client.get_json(URL)


@respx.mock
async def test_requests_to_the_same_site_are_spaced_out():
    respx.get(URL).respond(json={})
    async with make_client(per_host_delay=0.3) as client:
        start = time.monotonic()
        await client.get_json(URL)
        await client.get_json(URL)
        elapsed = time.monotonic() - start
    assert elapsed >= 0.3


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",  # this machine
        "http://10.0.0.5/",  # private network
        "http://192.168.1.1/",  # home router range
        "http://169.254.169.254/latest/meta-data/",  # cloud server secrets endpoint
        "http://[::1]/",  # this machine, IPv6
        "ftp://example.com/file",  # not http(s)
    ],
)
async def test_private_and_non_web_addresses_are_blocked(url):
    with pytest.raises(BlockedURLError):
        await ensure_public_url(httpx.URL(url))


async def test_public_ip_address_is_allowed():
    await ensure_public_url(httpx.URL("https://8.8.8.8/"))  # no error means allowed


async def test_client_blocks_private_address_before_connecting():
    async with PoliteClient(per_host_delay=0) as client:
        with pytest.raises(BlockedURLError):
            await client.get_text("http://127.0.0.1:8000/")
