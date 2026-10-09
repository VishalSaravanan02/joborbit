"""Run one full cycle against the real job sites: fetch, pre-filter, then LLM analysis.

Fetches every active company in the database, saves what changed, and records a fetch run for
each. The first cycle for a company only builds its baseline, so it reports no new jobs; later
cycles report only jobs posted since. Then the pre-filter decides every waiting job, and the LLM
analyses the ones that pass (newest first). If the LLM can't be used (no key, unreachable, or the
daily cap or budget reached), the jobs wait for the next run: nothing is lost.

Usage:
    python scripts/run_cycle.py [--no-analysis] [--limit N] [--retry-failed] [--show-rejected]

--no-analysis    fetch and pre-filter only: no LLM calls, no cost
--limit N        analyse at most N jobs (the rest wait for the next run)
--retry-failed   first send jobs whose analysis failed back for another try (e.g. after a prompt fix)
--show-rejected  list every new job with its pre-filter verdict, not only the ones that passed

A log of the run is also written to var/logs/run_cycle.log.
"""

import argparse
import asyncio

from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.db.models import Company, Job, JobAnalysis
from joborbit.db.session import get_engine
from joborbit.pipeline.cycle import Processing, process_new_jobs
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
    print(f"\n{succeeded} of {len(result.outcomes)} companies fetched successfully; {new} new {plural(new, 'job')}.")


def plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def verdict(job: Job) -> str:
    """The pre-filter's verdict on one job, as a short tag: "[passed]", "[rejected: senior title: lead]"."""
    if job.prefilter_status == "rejected" and job.prefilter_reason:
        return f"[rejected: {job.prefilter_reason}]"
    return f"[{job.prefilter_status}]"


def print_new_jobs(session: Session, job_ids: list[int], show_rejected: bool = False) -> None:
    """The new jobs that passed the pre-filter, or every new job with its verdict if `show_rejected`.

    The rejected ones are already counted by reason in the pre-filter line.
    """
    query = (
        select(Company.name, Job)
        .join(Company, Job.company_id == Company.id)
        .where(Job.id.in_(job_ids))
        .order_by(Company.name, Job.title)
    )
    if not show_rejected:
        query = query.where(Job.prefilter_status == "passed")
    rows = session.execute(query).all()
    hidden = len(job_ids) - len(rows)
    if rows:
        print("\nNew jobs:" if show_rejected else "\nNew jobs that passed the pre-filter:")
        for company, job in rows:
            print(f"  {company}: {job.title} ({job.location_raw or 'no location'})  {verdict(job)}\n    {job.url}")
    if hidden:
        print(f"\n{hidden} other new {plural(hidden, 'job')} not listed (--show-rejected lists them).")


def experience(analysis: JobAnalysis) -> str:
    """The experience asked for, in a few words: "no experience asked", "2+ years required"."""
    if analysis.experience_years is None:
        return "experience required" if analysis.experience_mandatory else "no experience asked"
    need = "required" if analysis.experience_mandatory else "preferred"
    return f"{analysis.experience_years}+ {plural(analysis.experience_years, 'year')} {need}"


def describe_analysis(analysis: JobAnalysis) -> str:
    """One line with what matters for matching: countries, seniority, experience and roles."""
    roles = list(analysis.role_families) + [f"{name} (custom)" for name in analysis.matched_custom_roles]
    return " | ".join(
        [
            ", ".join(analysis.countries) or "no country found",
            analysis.seniority or "seniority unclear",
            experience(analysis),
            ", ".join(roles) or "no role",
        ]
    )


def print_processing(session: Session, processing: Processing) -> None:
    prefilter = processing.prefilter
    rejected = sum(prefilter.rejected.values())
    reasons = ", ".join(f"{count} {kind}" for kind, count in prefilter.rejected.most_common())
    print(
        f"\nPre-filter: {prefilter.checked} checked, {len(prefilter.passed_job_ids)} passed, {rejected} rejected"
        + (f" ({reasons})" if reasons else "")
    )
    if processing.retried:
        print(f"{processing.retried} failed {plural(processing.retried, 'job')} sent back for another try.")

    analysis = processing.analysis
    if analysis is None:
        print(f"Analysis switched off; {processing.waiting} {plural(processing.waiting, 'job')} waiting for it.")
        return
    print(
        f"Analysis: {len(analysis.done)} done, {len(analysis.failed)} failed, {processing.waiting} waiting, "
        f"about ${analysis.cost_usd:.4f}"
    )
    if analysis.stopped:
        print(f"Analysis stopped: {analysis.stopped.rstrip('.')}. The waiting jobs are kept for a later run.")

    if analysis.done:
        print("\nAnalysed jobs (countries | seniority | experience | roles):")
        rows = session.execute(
            select(Company.name, Job.title, JobAnalysis)
            .join(Company, Job.company_id == Company.id)
            .join(JobAnalysis, JobAnalysis.job_id == Job.id)
            .where(Job.id.in_(analysis.done))
            .order_by(Company.name, Job.title)
        )
        for company, title, job_analysis in rows:
            print(f"  {company}: {title}\n    {describe_analysis(job_analysis)}")
    if analysis.failed:
        print("\nNo usable analysis (details in var/logs/run_cycle.log; --retry-failed tries them again):")
        rows = session.execute(
            select(Company.name, Job.title)
            .join(Company, Job.company_id == Company.id)
            .where(Job.id.in_(analysis.failed))
            .order_by(Company.name, Job.title)
        )
        for company, title in rows:
            print(f"  {company}: {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-analysis", action="store_true", help="fetch and pre-filter only (no LLM calls)")
    parser.add_argument("--limit", type=int, help="analyse at most this many jobs")
    parser.add_argument("--retry-failed", action="store_true", help="retry jobs whose analysis failed")
    parser.add_argument("--show-rejected", action="store_true", help="list rejected new jobs too")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    setup_logging("run_cycle")
    result = asyncio.run(run_fetch_cycle())
    processing = process_new_jobs(analyse=not args.no_analysis, limit=args.limit, retry=args.retry_failed)
    print_summary(result)
    with Session(get_engine()) as session:
        print_new_jobs(session, result.new_job_ids, args.show_rejected)
        print_processing(session, processing)


if __name__ == "__main__":
    main()
