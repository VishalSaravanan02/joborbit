"""Run one fetch cycle against the real job sites, save the results, and print a summary.

Fetches every active company in the database, saves what changed, and records a
fetch run for each. The first cycle for a company only builds its baseline, so it
reports no new jobs; later cycles report only jobs posted since.

Usage:
    python scripts/run_cycle.py

A log of the run is also written to var/logs/run_cycle.log.
"""

import argparse
import asyncio

from sqlalchemy import select

from joborbit.db.models import Company, Job
from joborbit.db.session import session_scope
from joborbit.pipeline.ingest import CycleResult, run_fetch_cycle
from joborbit.utils.logging import setup_logging


def print_summary(result: CycleResult) -> None:
    seconds = (result.finished_at - result.started_at).total_seconds()
    print(f"\nCycle finished in {seconds:.1f} s\n")
    print(f"{'Company':<24} {'Status':<7} {'Jobs':>5} {'New':>4} {'Baseline':>9} {'Closed':>7}")
    for outcome in result.outcomes:
        if outcome.ok and outcome.saved:
            saved = outcome.saved
            print(
                f"{outcome.company_name:<24} {'ok':<7} {saved.jobs_returned:>5} {len(saved.new_job_ids):>4} "
                f"{saved.baseline:>9} {saved.closed:>7}"
            )
        else:
            print(f"{outcome.company_name:<24} {'FAILED':<7}  {outcome.error}")

    succeeded = len(result.outcomes) - len(result.failed)
    new = len(result.new_job_ids)
    jobs_word = "job" if new == 1 else "jobs"
    print(f"\n{succeeded} of {len(result.outcomes)} companies fetched successfully; {new} new {jobs_word}.")


def print_new_jobs(job_ids: list[int]) -> None:
    if not job_ids:
        return
    print("\nNew jobs:")
    with session_scope() as session:
        rows = session.execute(
            select(Company.name, Job.title, Job.location_raw, Job.url)
            .join(Company, Job.company_id == Company.id)
            .where(Job.id.in_(job_ids))
            .order_by(Company.name, Job.title)
        )
        for company, title, location, url in rows:
            print(f"  {company}: {title} ({location or 'no location'})\n    {url}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    setup_logging("run_cycle")
    result = asyncio.run(run_fetch_cycle())
    print_summary(result)
    print_new_jobs(result.new_job_ids)


if __name__ == "__main__":
    main()
