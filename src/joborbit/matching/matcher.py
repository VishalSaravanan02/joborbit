"""Matching: every analysed job against every user, saved as matches.

run_matching takes every job waiting for matching (open, not baseline, analysis done, and
matched_at still empty), newest first, and for each active user with a profile:

1. runs the hard filters (filters.py); a job that breaks one is not that user's, and only the
   reason is counted;
2. scores the job (scoring.py) and picks its tier (router.py);
3. saves a match: a new row, or the existing one updated if the job was matched before (a job
   back after a long gap is matched again). created_at keeps the first time.

Then it sets the job's matched_at, so a job is matched once, however many runs see it.
Like the pre-filter, it takes every waiting job, not only this cycle's: jobs left by a run that
stopped between analysis and matching are picked up by the next one.

Paused users are matched too: pausing stops alerts, not matching, so nothing is missing from the
dashboard when they resume. Sending alerts is a later step; matching never sends anything.
Saving never commits: the caller decides when to save.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from joborbit.db.models import Company, Job, JobAnalysis, Match, User, UserProfile
from joborbit.matching.facts import JobFacts, UserFacts
from joborbit.matching.filters import check_job
from joborbit.matching.router import route
from joborbit.matching.scoring import Score, score_job
from joborbit.utils.timeutil import utcnow

DONE = "done"  # the analysis status of a job ready for matching


@dataclass(frozen=True)
class NewMatch:
    """A match made in this run, with what a person needs to see it."""

    match_id: int
    user_name: str
    job_id: int
    company: str
    title: str
    url: str
    score: int
    tier: str
    reasons: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass
class MatchingRun:
    """What one matching run did."""

    jobs: int = 0  # jobs matched against every user
    matches: list[NewMatch] = field(default_factory=list)  # one per user and job that passed
    dropped: Counter[str] = field(default_factory=Counter)  # user-job pairs dropped, by hard filter


def waiting_job_ids(session: Session) -> list[int]:
    """Open, non-baseline, analysed jobs not matched yet, newest first."""
    return list(
        session.scalars(
            select(Job.id)
            .where(
                Job.analysis_status == DONE,
                Job.matched_at.is_(None),
                Job.closed_at.is_(None),
                Job.is_baseline.is_(False),
            )
            .order_by(Job.became_new_at.desc(), Job.id.desc())
        )
    )


def active_users(session: Session) -> list[tuple[User, UserFacts]]:
    """Every active user with a profile (paused ones included), with their facts."""
    users = session.scalars(
        select(User)
        .join(UserProfile)
        .where(User.is_active)
        .options(selectinload(User.profile), selectinload(User.company_prefs))
        .order_by(User.id)
    )
    return [(user, UserFacts.from_rows(user.profile, user.company_prefs)) for user in users]


def _filter_kind(reason: str) -> str:
    """The rule in a hard filter's reason, for counting: "level: senior" -> "level"."""
    return reason.split(":", 1)[0]


def run_matching(session: Session) -> MatchingRun:
    """Match every waiting job against every active user and save the matches. Never commits."""
    run = MatchingRun()
    users = active_users(session)
    if not users:
        return run  # nobody to match for: the jobs keep waiting until someone is

    for job_id in waiting_job_ids(session):
        job, analysis, company = session.execute(
            select(Job, JobAnalysis, Company)
            .join(JobAnalysis, JobAnalysis.job_id == Job.id)
            .join(Company, Job.company_id == Company.id)
            .where(Job.id == job_id)
        ).one()
        facts = JobFacts.from_rows(job, analysis, company)
        now = utcnow()

        for user, user_facts in users:
            verdict = check_job(facts, user_facts)
            if not verdict.passed:
                run.dropped[_filter_kind(verdict.reason)] += 1
                continue
            score = score_job(facts, user_facts, verdict)
            tier = route(score, verdict, user_facts.alert_style)
            match = _save_match(session, user.id, job.id, score, tier, now)
            run.matches.append(
                NewMatch(
                    match_id=match.id, user_name=user.display_name, job_id=job.id, company=company.name,
                    title=job.title, url=job.url, score=score.score, tier=tier,
                    reasons=tuple(score.reasons), notes=tuple(score.notes),
                )
            )

        job.matched_at = now
        run.jobs += 1
    return run


def _save_match(session: Session, user_id: int, job_id: int, score: Score, tier: str, now: datetime) -> Match:
    """Add the match, or update the one already there (created_at keeps the first time).

    The reasons column holds the reasons, then the notes: everything the alert will show.
    """
    match = session.scalars(select(Match).where(Match.user_id == user_id, Match.job_id == job_id)).one_or_none()
    if match is None:
        match = Match(user_id=user_id, job_id=job_id, created_at=now)
        session.add(match)
    match.score, match.tier = score.score, tier
    match.components, match.reasons = score.points, score.reasons + score.notes
    match.scored_at = now
    session.flush()  # gives a new match its id
    return match
