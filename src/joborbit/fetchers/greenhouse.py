"""Fetcher for companies whose jobs are hosted on Greenhouse.

Greenhouse offers a public job-board API: one request returns every open job
at a company, including full descriptions when `content=true` is added.
The company's token is the name in its board address, e.g. "monzo" in
job-boards.greenhouse.io/monzo.
"""

import html
from typing import Any

from joborbit.fetchers.base import Fetcher, NormalisedJob, parse_timestamp
from joborbit.utils.http import FetchError, PoliteClient

API_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


class GreenhouseFetcher(Fetcher):
    ats_type = "greenhouse"

    async def fetch(self, token: str, client: PoliteClient) -> list[NormalisedJob]:
        data = await client.get_json(API_URL.format(token=token), params={"content": "true"})
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            raise FetchError(f"Unexpected reply from Greenhouse for {token!r}: no 'jobs' list")
        return self.parse_jobs(data["jobs"], token)

    def parse_job(self, raw: dict[str, Any]) -> NormalisedJob:
        content = raw.get("content")
        return NormalisedJob(
            external_id=raw["id"],
            title=raw.get("title") or "",
            url=raw.get("absolute_url") or "",
            location_raw=_location_with_offices(raw),
            # Greenhouse sends the description HTML "escaped" (&lt;p&gt; instead of <p>).
            description_html=html.unescape(content) if content else None,
            # first_published = when the job went live; updated_at changes on every edit.
            posted_at=parse_timestamp(raw.get("first_published") or raw.get("updated_at")),
        )


def _location_with_offices(raw: dict[str, Any]) -> str:
    """The job's location, plus any office names it doesn't already mention.

    Some companies use the location field for the way of working ("Hybrid",
    "Distributed") and list the actual cities only under offices, e.g. location
    "Hybrid" with offices "Austin, TX" and "London, United Kingdom". Adding the office
    names gives "Hybrid / Austin, TX / London, United Kingdom", which the location
    parser understands. Office names already in the text are not repeated.
    """
    location = raw.get("location")
    text = (location.get("name") if isinstance(location, dict) else None) or ""
    parts = [text.strip()] if text.strip() else []
    for office in raw.get("offices") or []:
        name = (office.get("name") if isinstance(office, dict) else None) or ""
        name = name.strip()
        if name and not any(name.lower() in part.lower() for part in parts):
            parts.append(name)
    return " / ".join(parts)
