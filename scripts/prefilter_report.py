"""Show what the pre-filter would decide for the jobs already in the database. Changes nothing.

Runs the same rules as the real pre-filter, built from every active user's roles, over every
open job (baseline jobs included), then prints how many would pass, why the rest would be
rejected, and sample titles from each group. Use it after editing roles.yaml or
seniority.yaml to see the effect straight away.

Usage:
    python scripts/prefilter_report.py [--samples N] [--all-passed] [--reason TEXT] [--include-closed]

Examples:
    python scripts/prefilter_report.py
    python scripts/prefilter_report.py --all-passed
    python scripts/prefilter_report.py --reason "no matching role"
    python scripts/prefilter_report.py --reason "senior title: lead"
"""

import argparse
import random
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.db.models import Company, Job
from joborbit.db.session import get_engine
from joborbit.pipeline.prefilter import Decision, check_job, reason_kind, rules_for_active_users

TITLE_WIDTH = 60


def line(company: str, job: Job, decision: Decision) -> str:
    """One job as a table row: company, title, location and the verdict's reason."""
    title = job.title if len(job.title) <= TITLE_WIDTH else job.title[: TITLE_WIDTH - 1] + "…"
    return f"  {company[:18]:<18} {title:<{TITLE_WIDTH}} {(job.location_raw or '-')[:30]:<30} {decision.reason}"


def sample(rows: list, count: int, seed: str) -> list:
    """Up to `count` rows, picked at random but the same every run (so reports can be compared)."""
    if count == 0 or len(rows) <= count:
        return rows
    return sorted(random.Random(seed).sample(rows, count), key=lambda row: row[1].id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=10, help="sample titles per group (default 10, 0 = all)")
    parser.add_argument("--all-passed", action="store_true", help="list every job that would pass")
    parser.add_argument("--reason", metavar="TEXT", help="list every job rejected with a reason starting with TEXT")
    parser.add_argument("--include-closed", action="store_true", help="also check jobs that have closed")
    args = parser.parse_args()

    with Session(get_engine()) as session:  # closes without saving: this report changes nothing
        rules = rules_for_active_users(session)
        query = select(Company.name, Job).join(Company, Job.company_id == Company.id).order_by(Job.id)
        if not args.include_closed:
            query = query.where(Job.closed_at.is_(None))

        passed: list = []
        rejected: dict[str, list] = defaultdict(list)
        role_counts: Counter[str] = Counter()
        for company, job in session.execute(query):
            decision = check_job(job.title, job.location_raw, rules)
            row = (company, job, decision)
            if decision.passed:
                passed.append(row)
                role_counts.update(match.role for match in decision.roles)
            else:
                rejected[reason_kind(decision.reason)].append(row)

        total = len(passed) + sum(len(rows) for rows in rejected.values())
        which = "jobs" if args.include_closed else "open jobs"
        print(f"Checked {total:,} {which} with the roles of the active users:")
        custom = sum(rule.custom for rule in rules.roles)
        print(
            f"  {len(rules.roles) - custom} default roles, {custom} custom; "
            f"internships {'allowed' if rules.allow_internships else 'off'}\n"
        )
        share = f"{len(passed) / total:.1%}" if total else "-"
        print(f"Would pass: {len(passed):>7,}  ({share})")
        print("Rejected:")
        for kind, rows in sorted(rejected.items(), key=lambda item: -len(item[1])):
            print(f"  {kind:<20} {len(rows):>7,}")
        if role_counts:
            print("\nPassing jobs by role (a job can match several):")
            for role, count in role_counts.most_common():
                print(f"  {role:<28} {count:>5,}")

        if args.reason:
            matching = [
                row for rows in rejected.values() for row in rows if row[2].reason.startswith(args.reason)
            ]
            print(f'\nREJECTED, reason starting "{args.reason}": {len(matching):,}')
            for row in sorted(matching, key=lambda row: (row[0], row[1].title)):
                print(line(*row))
            return

        shown = passed if args.all_passed else sample(passed, args.samples, "passed")
        print(f"\nWOULD PASS ({len(shown):,} of {len(passed):,}):")
        for row in shown:
            print(line(*row))
        for kind, rows in sorted(rejected.items(), key=lambda item: -len(item[1])):
            shown = sample(rows, args.samples, kind)
            print(f"\nREJECTED: {kind} ({len(shown):,} of {len(rows):,}):")
            for row in shown:
                print(line(*row))


if __name__ == "__main__":
    main()
