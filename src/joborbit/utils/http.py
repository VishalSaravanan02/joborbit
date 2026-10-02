"""A polite, safe web client shared by every fetcher.

"Polite" means it never hammers a website: at most a few requests at once,
a pause between requests to the same site, and a clear User-Agent saying who
we are. "Safe" means it refuses to talk to private network addresses, gives up
on huge responses, and retries temporary failures a limited number of times.

Usage:
    async with PoliteClient() as client:
        data = await client.get_json("https://boards-api.greenhouse.io/v1/boards/acme/jobs")
"""

import asyncio
import ipaddress
import socket
import time
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from joborbit.settings import get_settings

MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB


# --- Errors -----------------------------------------------------------------
# Every failure is a FetchError, so the fetch cycle only needs to catch one type.


class FetchError(Exception):
    """A request failed and should be recorded against that company."""


class RateLimitedError(FetchError):
    """The site said "too many requests" (HTTP 429). Skip it this cycle."""

    def __init__(self, url: str, retry_after: float | None) -> None:
        super().__init__(f"Rate limited by {url} (retry after {retry_after} s)")
        self.retry_after = retry_after


class HTTPStatusFetchError(FetchError):
    """The site answered with an error code, e.g. 404 Not Found."""

    def __init__(self, url: str, status_code: int) -> None:
        super().__init__(f"HTTP {status_code} from {url}")
        self.status_code = status_code


class ResponseTooLargeError(FetchError):
    """The response was bigger than we are willing to download."""


class BlockedURLError(FetchError):
    """The URL points somewhere we must never connect to (e.g. a private address)."""


class _RetryableError(Exception):
    """Internal: a temporary failure worth trying again (timeouts, 5xx errors)."""


# --- Safety check: no private addresses (SSRF guard) ------------------------


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global and not ip.is_multicast


async def ensure_public_url(url: httpx.URL) -> None:
    """Refuse anything except http(s) URLs that resolve only to public internet addresses.

    This matters once users can type in careers-page URLs: without it, someone could
    point JobOrbit at the server's own internal services.
    """
    if url.scheme not in ("http", "https"):
        raise BlockedURLError(f"Only http and https are allowed: {url}")
    host = url.host
    if not host:
        raise BlockedURLError(f"URL has no host: {url}")

    try:
        addresses = [str(ipaddress.ip_address(host))]  # the host is already an IP
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise FetchError(f"Could not resolve {host}") from exc
        addresses = [info[4][0] for info in infos]

    for address in addresses:
        if not _is_public_ip(address):
            raise BlockedURLError(f"{host} resolves to a non-public address ({address})")


# --- The client ---------------------------------------------------------------


class PoliteClient:
    """An async HTTP client with concurrency limits, per-site pauses, retries and safety checks."""

    def __init__(
        self,
        *,
        max_concurrency: int = 5,
        per_host_delay: float = 1.0,
        timeout: float = 20.0,
        max_attempts: int = 3,
        retry_wait_seconds: float = 1.0,
        max_bytes: int = MAX_RESPONSE_BYTES,
        block_private_addresses: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        contact = get_settings().contact_email or "not set"
        hooks = {"request": [self._check_request]} if block_private_addresses else {}
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": f"JobOrbit personal job alerts (contact: {contact})",
                "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            },
            timeout=timeout,
            follow_redirects=True,
            event_hooks=hooks,
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._per_host_delay = per_host_delay
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request_at: dict[str, float] = {}
        self._max_attempts = max_attempts
        self._retry_wait_seconds = retry_wait_seconds
        self._max_bytes = max_bytes

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # -- public helpers --

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        response = await self.request("GET", url, **kwargs)
        return _parse_json(response)

    async def post_json(self, url: str, payload: Any, **kwargs: Any) -> Any:
        response = await self.request("POST", url, json=payload, **kwargs)
        return _parse_json(response)

    async def get_text(self, url: str, **kwargs: Any) -> str:
        response = await self.request("GET", url, **kwargs)
        return response.text

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Make one request politely, retrying temporary failures. Raises FetchError on failure."""
        async with self._semaphore:
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(_RetryableError),
                    stop=stop_after_attempt(self._max_attempts),
                    wait=wait_exponential(multiplier=self._retry_wait_seconds, max=10),
                    reraise=True,
                ):
                    with attempt:
                        return await self._request_once(method, url, **kwargs)
            except _RetryableError as exc:
                raise FetchError(f"Gave up after {self._max_attempts} attempts: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    # -- internals --

    async def _check_request(self, request: httpx.Request) -> None:
        """Runs before every request, including each redirect."""
        await ensure_public_url(request.url)

    async def _wait_for_host(self, host: str) -> None:
        """Make sure requests to the same site are at least per_host_delay seconds apart."""
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last_request_at.get(host)
            if last is not None:
                remaining = self._per_host_delay - (time.monotonic() - last)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_request_at[host] = time.monotonic()

    async def _request_once(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        await self._wait_for_host(httpx.URL(url).host)
        try:
            async with self._client.stream(method, url, **kwargs) as response:
                if response.status_code == 429:
                    raise RateLimitedError(url, _retry_after_seconds(response))
                if response.status_code >= 500:
                    raise _RetryableError(f"HTTP {response.status_code} from {url}")
                if response.status_code >= 400:
                    raise HTTPStatusFetchError(url, response.status_code)

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        raise ResponseTooLargeError(f"Response from {url} is over {self._max_bytes} bytes")

                # The body is already decompressed, so drop the headers describing the
                # compressed form; otherwise httpx would try to decompress it a second time.
                headers = httpx.Headers(response.headers)
                for name in ("content-encoding", "content-length", "transfer-encoding"):
                    headers.pop(name, None)

                return httpx.Response(
                    status_code=response.status_code,
                    headers=headers,
                    content=bytes(body),
                    request=response.request,
                )
        except httpx.TimeoutException as exc:
            raise _RetryableError(f"Timed out: {url}") from exc
        except httpx.TransportError as exc:
            raise _RetryableError(f"Connection problem with {url}: {exc}") from exc
        except httpx.HTTPError as exc:  # anything else httpx can raise, e.g. a corrupt reply
            raise FetchError(f"Bad response from {url}: {exc}") from exc


def _parse_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise FetchError(f"Expected JSON from {response.request.url}, got something else") from exc


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
