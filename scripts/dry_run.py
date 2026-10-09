"""Show what JobOrbit would send over recent jobs, without changing or sending anything.

Takes every open job posted in the last --days days (baseline jobs included), runs the
pre-filter, analyses the jobs that pass (reusing saved answers; new answers cost a little and are
kept in var/dry_run/answers.json), then matches them against every active user. Prints, for each
user, how many jobs would be instant alerts, digest lines or silent, the rate per day, and every
instant and digest job with its score in parts, reasons and notes. Run it again after changing
roles.yaml, seniority.yaml or the scoring settings: only jobs not seen before cost anything.

Also writes var/dry_run/review.csv: every job above for every user, plus the jobs the pre-filter
rejected as "no matching role", with an empty "want" column. Fill it in (yes, maybe or no) and save
it as CSV under the same name: running again keeps your answers.

Usage:
    python scripts/dry_run.py [--days 14] [--limit 100] [--no-analysis] [--show-silent] [--show-dropped]

--days N        the window: jobs posted in the last N days (default 14)
--limit N       ask the LLM about at most N new jobs in this run, newest first (default 100)
--no-analysis   use only analyses we already have: no LLM calls, no cost
--show-silent   also list the jobs that would only appear on the dashboard
--show-dropped  also list the jobs a user's hard filters dropped, with the reason
"""

import argparse
from collections import Counter

from sqlalchemy.orm import Session

from joborbit.db.session import get_engine
from joborbit.llm.client import make_client
from joborbit.pipeline.dry_run import (
    DEFAULT_DAYS,
    DEFAULT_LIMIT,
    DEFAULT_REVIEW_FILE,
    DryResult,
    DryRun,
    points_line,
    run_dry_run,
    write_review,
)
from joborbit.settings import PROJECT_ROOT
from joborbit.utils.logging import setup_logging
from joborbit.utils.timeutil import utcnow


def plural(count: int, word: str, many: str | None = None) -> str:
    return word if count == 1 else (many or f"{word}s")


def print_job(result: DryResult) -> None:
    tag = f"{result.tier} {result.score}" if result.tier else f"dropped: {result.dropped}"
    print(f"  [{tag}] {result.company}: {result.title} (posted {result.posted_at:%d %b})")
    if result.tier:
        print(f"    {points_line(result.points)}")
        if result.reasons:
            print(f"    {' | '.join(result.reasons)}")
        for note in result.notes:
            print(f"    ! {note}")
    print(f"    {result.url}")


def print_summary(run: DryRun) -> None:
    print(
        f"Dry run over the last {run.days:g} days ({run.since:%d %b} to {run.until:%d %b}). "
        "Nothing is saved or sent."
    )
    print(
        f"\nJobs: {run.jobs} open {plural(run.jobs, 'job')} posted in the window "
        f"({run.no_date} more have no posting date and are left out)."
    )
    reasons = ", ".join(f"{count} {kind}" for kind, count in run.rejected.most_common())
    rejected = sum(run.rejected.values())
    print(f"Pre-filter: {run.passed} passed, {rejected} rejected" + (f" ({reasons})" if reasons else ""))
    print(
        f"Analysis: {run.from_database} from the database, {run.from_file} saved, {run.asked} asked "
        f"(about ${run.cost_usd:.4f}), {run.failed} failed, {run.not_analysed} not analysed"
    )
    if run.stopped:
        print(f"Analysis stopped: {run.stopped.rstrip('.')}.")
    if run.not_analysed:
        print("  Jobs not analysed are left out below: run again (or with a higher --limit) to include them.")


def print_user(name: str, results: list[DryResult], days: float, show_silent: bool, show_dropped: bool) -> None:
    tiers = Counter(result.tier for result in results if result.tier)
    dropped = Counter(result.dropped.split(":", 1)[0] for result in results if result.dropped)
    print(
        f"\n{name}: {tiers['instant']} instant, {tiers['digest']} digest, {tiers['silent']} silent, "
        f"{sum(dropped.values())} dropped"
        + (f" ({', '.join(f'{count} {kind}' for kind, count in dropped.most_common())})" if dropped else "")
    )
    digest_days = Counter(result.posted_at.date() for result in results if result.tier == "digest")
    busiest = max(digest_days.values(), default=0)
    print(
        f"  Per day: {tiers['instant'] / days:.1f} instant, {tiers['digest'] / days:.1f} digest "
        f"(busiest day: {busiest} digest {plural(busiest, 'job')})"
    )

    shown = ["instant", "digest"] + (["silent"] if show_silent else [])
    for tier in shown:
        listed = sorted((r for r in results if r.tier == tier), key=lambda r: (-r.score, r.company, r.title))
        if listed:
            print(f"\n  {tier.capitalize()}:")
            for result in listed:
                print_job(result)
    if show_dropped:
        listed = sorted((r for r in results if r.dropped), key=lambda r: (r.dropped, r.company, r.title))
        if listed:
            print("\n  Dropped by the hard filters:")
            for result in listed:
                print_job(result)


def print_report(run: DryRun, show_silent: bool = False, show_dropped: bool = False) -> None:
    print_summary(run)
    by_user: dict[str, list[DryResult]] = {}
    for result in run.results:
        by_user.setdefault(result.user_name, []).append(result)
    if not by_user:
        print("\nNo analysed jobs to match.")
    for name, results in by_user.items():
        print_user(name, results, run.days, show_silent, show_dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="jobs posted in the last N days")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="ask about at most N new jobs")
    parser.add_argument("--no-analysis", action="store_true", help="use only analyses we already have")
    parser.add_argument("--show-silent", action="store_true", help="also list silent jobs")
    parser.add_argument("--show-dropped", action="store_true", help="also list dropped jobs")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.limit < 0:
        parser.error("--limit can't be negative")

    setup_logging("dry_run")
    with Session(get_engine()) as session:  # closes without saving: the dry run changes nothing
        run = run_dry_run(
            session, utcnow(), days=args.days, limit=args.limit,
            client_factory=None if args.no_analysis else make_client,
        )
    print_report(run, args.show_silent, args.show_dropped)
    rows, kept = write_review(run, DEFAULT_REVIEW_FILE)
    print(
        f"\nReview sheet: {DEFAULT_REVIEW_FILE.relative_to(PROJECT_ROOT)} ({rows} {plural(rows, 'row')}, "
        f"{kept} {plural(kept, 'answer')} kept from the last sheet). Fill in the want column: yes, maybe or no."
    )


if __name__ == "__main__":
    main()
