"""Fill in the ATS and token for every company in the company CSV, and report the results.

Usage:
    python scripts/detect_ats_bulk.py [CSV_FILE] [--recheck]

CSV_FILE defaults to data/companies_seed.csv. Rows that already have an ATS and
token are skipped unless --recheck is given. Rows without a careers URL are still
checked, by probing with the company's name.

Results are written back to the same file. Companies found by probing are written
in too, but check each one's sample job link before importing: probing can find
a different company with the same name. Ambiguous, failed and not-found rows are
left unchanged so you can fix them by hand.
"""

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

from joborbit.companies import CompanyRow, apply_detections, read_company_csv, write_company_csv
from joborbit.fetchers.detect import Detection, detect_ats
from joborbit.settings import PROJECT_ROOT
from joborbit.utils.http import PoliteClient

DEFAULT_CSV = PROJECT_ROOT / "data" / "companies_seed.csv"


async def detect_rows(rows: list[CompanyRow]) -> dict[str, Detection]:
    """Run detection for every row at once (the polite client keeps it gentle)."""
    async with PoliteClient() as client:
        results = await asyncio.gather(*(detect_ats(row.careers_url, client, name=row.name) for row in rows))
    return {row.slug: result for row, result in zip(rows, results, strict=True)}


def print_report(rows: list[CompanyRow], detections: dict[str, Detection], skipped: list[CompanyRow]) -> None:
    by_status: dict[str, list[tuple[CompanyRow, Detection]]] = defaultdict(list)
    for row in rows:
        by_status[detections[row.slug].status].append((row, detections[row.slug]))

    titles = {
        "ready": "READY (supported, test fetch worked)",
        "probed": "FOUND BY PROBING (written in; open each sample job to confirm it's the right company)",
        "ambiguous": "AMBIGUOUS (probing found several boards; choose one by hand)",
        "unsupported": "UNSUPPORTED ATS (recognised, no fetcher yet)",
        "not_found": "NOT FOUND (no ATS link on the careers page, and probing found nothing)",
        "failed": "FAILED (fix these by hand)",
    }
    for status, title in titles.items():
        entries = by_status.get(status, [])
        print(f"\n{title}: {len(entries)}")
        for row, result in entries:
            if status == "ready":
                print(f"  {row.name}: {result.ats_type} / {result.token} ({result.job_count} jobs)")
            elif status == "probed":
                print(f"  {row.name}: {result.ats_type} / {result.token} ({result.job_count} jobs)")
                print(f"      sample job: {result.sample_url}")
            elif status == "ambiguous":
                print(f"  {row.name}:")
                for match in result.matches:
                    print(f"      {match.ats_type} / {match.token} ({match.job_count} jobs)  e.g. {match.sample_url}")
            elif status == "unsupported":
                print(f"  {row.name}: {result.ats_type} / {result.token}")
            else:
                print(f"  {row.name}: {result.message}")

    if skipped:
        print(f"\nSKIPPED (already had an ATS and token): {len(skipped)}")
        for row in skipped:
            print(f"  {row.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--recheck", action="store_true", help="also re-detect rows that already have an ATS")
    args = parser.parse_args()

    try:
        rows = read_company_csv(args.csv_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    to_check = [r for r in rows if args.recheck or not (r.ats_type and r.ats_token)]
    skipped = [r for r in rows if r not in to_check]
    print(f"Checking {len(to_check)} companies (this can take a few minutes)...")

    detections = asyncio.run(detect_rows(to_check))
    write_company_csv(args.csv_file, apply_detections(rows, detections))
    print_report(to_check, detections, skipped)
    print(f"\nUpdated {args.csv_file}")


if __name__ == "__main__":
    main()
