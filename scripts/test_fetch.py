"""Fetch one company's jobs and print them. Nothing is saved.

Useful for checking a company's ATS and token before adding it, or for seeing
what a job site is returning right now.

Usage:
    python scripts/test_fetch.py ATS TOKEN [--limit 10]

Examples:
    python scripts/test_fetch.py greenhouse monzo
    python scripts/test_fetch.py lever palantir --limit 20
    python scripts/test_fetch.py ashby elevenlabs --limit 0     (0 = show every job)
"""

import argparse
import asyncio

from joborbit.fetchers.base import NormalisedJob
from joborbit.fetchers.registry import FETCHERS, get_fetcher
from joborbit.pipeline.location import parse_location
from joborbit.utils.http import FetchError, PoliteClient


async def fetch(ats_type: str, token: str) -> list[NormalisedJob]:
    fetcher = get_fetcher(ats_type)
    if fetcher is None:
        raise SystemExit(f"Unknown ATS {ats_type!r}. Supported: {', '.join(sorted(FETCHERS))}")
    async with PoliteClient() as client:
        return await fetcher.fetch(token, client)


def describe(job: NormalisedJob) -> str:
    """A few lines about one job, including the countries our parser finds in its location."""
    location = parse_location(job.location_raw)
    countries = ", ".join(location.countries) or ("ambiguous" if location.ambiguous else "none of ours")
    posted = job.posted_at.strftime("%Y-%m-%d") if job.posted_at else "unknown"
    return (
        f"{job.title}\n"
        f"    location: {job.location_raw or '(blank)'}  ->  {countries}\n"
        f"    posted:   {posted}    id: {job.external_id}\n"
        f"    {job.url}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ats", help="the ATS name, e.g. greenhouse, lever or ashby")
    parser.add_argument("token", help="the company's name on that ATS, e.g. monzo")
    parser.add_argument("--limit", type=int, default=10, help="how many jobs to show (default 10, 0 = all)")
    args = parser.parse_args()

    try:
        jobs = asyncio.run(fetch(args.ats.lower(), args.token))
    except FetchError as exc:
        raise SystemExit(f"Fetch failed: {exc}") from exc

    shown = jobs if args.limit == 0 else jobs[: args.limit]
    for job in shown:
        print(describe(job))
    uk_jobs = sum(1 for job in jobs if "GB" in parse_location(job.location_raw).countries)
    print(f"\n{len(jobs)} jobs found ({uk_jobs} in the UK); showing {len(shown)}. Nothing was saved.")


if __name__ == "__main__":
    main()
