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
        location = raw.get("location") or {}
        content = raw.get("content")
        return NormalisedJob(
            external_id=raw["id"],
            title=raw.get("title") or "",
            url=raw.get("absolute_url") or "",
            location_raw=location.get("name") or "",
            # Greenhouse sends the description HTML "escaped" (&lt;p&gt; instead of <p>).
            description_html=html.unescape(content) if content else None,
            # first_published = when the job went live; updated_at changes on every edit.
            posted_at=parse_timestamp(raw.get("first_published") or raw.get("updated_at")),
        )
