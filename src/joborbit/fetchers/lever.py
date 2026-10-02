"""Fetcher for companies whose jobs are hosted on Lever.

Lever's public postings API returns every open job at a company as a list.
The token is the name in the company's job-board address, e.g. "palantir" in
jobs.lever.co/palantir. Companies on Lever's European servers
(jobs.eu.lever.co/...) use a token starting with "eu:", e.g. "eu:examplecompany".
"""

from typing import Any

from joborbit.fetchers.base import Fetcher, NormalisedJob, parse_timestamp
from joborbit.utils.http import FetchError, PoliteClient

API_URL = "https://api.lever.co/v0/postings/{name}"
EU_API_URL = "https://api.eu.lever.co/v0/postings/{name}"
EU_PREFIX = "eu:"


def api_url_for(token: str) -> str:
    """The right API address for a token, choosing Lever's EU servers for "eu:" tokens."""
    if token.startswith(EU_PREFIX):
        return EU_API_URL.format(name=token.removeprefix(EU_PREFIX))
    return API_URL.format(name=token)


class LeverFetcher(Fetcher):
    ats_type = "lever"

    async def fetch(self, token: str, client: PoliteClient) -> list[NormalisedJob]:
        data = await client.get_json(api_url_for(token), params={"mode": "json"})
        if not isinstance(data, list):
            raise FetchError(f"Unexpected reply from Lever for {token!r}: expected a list of jobs")
        return self.parse_jobs(data, token)

    def parse_job(self, raw: dict[str, Any]) -> NormalisedJob:
        categories = raw.get("categories") or {}
        all_locations = categories.get("allLocations") or []
        location = " / ".join(all_locations) if all_locations else categories.get("location") or ""
        return NormalisedJob(
            external_id=raw["id"],
            title=raw.get("text") or "",
            url=raw.get("hostedUrl") or "",
            location_raw=location,
            description_html=_full_description(raw),
            posted_at=parse_timestamp(raw.get("createdAt")),
        )


def _full_description(raw: dict[str, Any]) -> str | None:
    """Join Lever's description pieces: the intro, each list section, then the closing text.

    The list sections ("Responsibilities", "Requirements"...) hold the details that
    matter most, such as required experience, so they must not be dropped.
    """
    parts = [raw.get("description") or ""]
    for section in raw.get("lists") or []:
        heading = section.get("text") or ""
        items = section.get("content") or ""
        parts.append(f"<h3>{heading}</h3><ul>{items}</ul>")
    parts.append(raw.get("additional") or "")
    html = "\n".join(part for part in parts if part)
    return html or None
