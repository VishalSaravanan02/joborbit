"""Fetcher for companies whose jobs are hosted on Ashby.

Ashby's public job-board API returns every open job at a company in one
request. The token is the name in the company's job-board address, e.g.
"elevenlabs" in jobs.ashbyhq.com/elevenlabs.
"""

from typing import Any

from joborbit.fetchers.base import Fetcher, NormalisedJob, parse_timestamp
from joborbit.utils.http import FetchError, PoliteClient

API_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"


class AshbyFetcher(Fetcher):
    ats_type = "ashby"

    async def fetch(self, token: str, client: PoliteClient) -> list[NormalisedJob]:
        data = await client.get_json(API_URL.format(token=token))
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            raise FetchError(f"Unexpected reply from Ashby for {token!r}: no 'jobs' list")
        listed = [raw for raw in data["jobs"] if not _is_hidden(raw)]
        return self.parse_jobs(listed, token)

    def parse_job(self, raw: dict[str, Any]) -> NormalisedJob:
        return NormalisedJob(
            external_id=raw["id"],
            title=raw.get("title") or "",
            url=raw.get("jobUrl") or "",
            location_raw=_all_locations(raw),
            description_html=raw.get("descriptionHtml") or None,
            posted_at=parse_timestamp(raw.get("publishedAt")),
        )


def _is_hidden(raw: Any) -> bool:
    """True for a job Ashby marks as not listed (hidden from its public board).

    Anything that isn't a proper job object counts as not hidden, so it is passed
    on to parse_jobs, which skips and logs it instead of crashing the whole fetch.
    """
    return isinstance(raw, dict) and not raw.get("isListed", True)


def _all_locations(raw: dict[str, Any]) -> str:
    """The main location plus any secondary ones, e.g. "London / Berlin"."""
    names = [raw.get("location") or ""]
    for extra in raw.get("secondaryLocations") or []:
        names.append((extra.get("location") or "") if isinstance(extra, dict) else str(extra))
    unique = list(dict.fromkeys(name.strip() for name in names if name and name.strip()))
    return " / ".join(unique)
