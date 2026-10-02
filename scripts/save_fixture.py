"""Download a real job-feed response and save it as a test fixture.

Fetchers are tested against real responses saved in tests/fixtures/, so we
know they handle what the job sites actually send. Only the first few jobs
are kept, so the saved file stays small.

Usage:
    python scripts/save_fixture.py URL NAME [--limit 3]

Example:
    python scripts/save_fixture.py "https://boards-api.greenhouse.io/v1/boards/monzo/jobs?content=true" greenhouse
"""

import argparse
import asyncio
import json
from typing import Any

from joborbit.settings import PROJECT_ROOT
from joborbit.utils.http import FetchError, PoliteClient

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def trim(data: Any, limit: int) -> tuple[Any, int | None]:
    """Keep only the first `limit` jobs. Returns (trimmed data, original number of jobs)."""
    if isinstance(data, list):
        return data[:limit], len(data)
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):  # the first list in the reply is the jobs list
                return {**data, key: value[:limit]}, len(value)
    return data, None


async def download(url: str) -> Any:
    async with PoliteClient() as client:
        return await client.get_json(url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="the job feed URL to download")
    parser.add_argument("name", help="file name to save as, without .json")
    parser.add_argument("--limit", type=int, default=3, help="how many jobs to keep (default 3)")
    args = parser.parse_args()

    try:
        data = asyncio.run(download(args.url))
    except FetchError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    trimmed, total = trim(data, args.limit)
    path = FIXTURES_DIR / f"{args.name}.json"
    path.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    if total is not None:
        print(f"The feed had {total} jobs; kept the first {min(total, args.limit)}.")


if __name__ == "__main__":
    main()
