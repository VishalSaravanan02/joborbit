"""Saves what a fetch returned: new jobs, edited jobs, and jobs that have disappeared.

A fetcher returns every job a company currently has open. This module compares
that list with the database and works out what changed:

- New job: saved. It goes on to the pre-filter, unless it is part of the
  company's baseline (the jobs it already had when we started tracking it) or a
  re-post of a job we already have open (a duplicate).
- Known job: marked as seen. If the company edited the ad, the stored text is
  updated, but nobody is alerted again.
- Job missing from the list: counted. After CLOSE_AFTER_MISSING fetches in a row
  without it, it is closed.
- Closed job that comes back: reopened. After a short gap it is reopened quietly;
  after a long gap it is treated as a fresh posting (and became_new_at records when).

An empty list closes nothing at first, because it is usually a glitch on the
company's side. Only if a company has shown no jobs at all for EMPTY_REPLY_GRACE
are its jobs closed as normal.

Saving never commits: the caller decides when to save, so one company's changes
are saved or undone together.

run_fetch_cycle ties it together: it fetches every active company at once
(politely, through one shared client), then saves each company's results in its
own transaction and records a fetch run for it. One company failing, for any
reason, never stops the others.
"""

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from joborbit.db.models import Company, FetchRun, Job
from joborbit.db.session import get_engine
from joborbit.fetchers.base import NormalisedJob
from joborbit.fetchers.registry import get_fetcher
from joborbit.pipeline.location import parse_location
from joborbit.pipeline.normalise import content_hash, html_to_text, job_fingerprint
from joborbit.utils.http import FetchError, PoliteClient
from joborbit.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

CLOSE_AFTER_MISSING = 2  # successful fetches in a row without the job before it is closed
REPROCESS_AFTER_CLOSED = timedelta(days=7)  # a job back after this long counts as a fresh posting
EMPTY_REPLY_GRACE = timedelta(hours=24)  # how long a company may show no jobs before we close them

# prefilter_status for jobs that must never go through the pre-filter or be alerted.
SKIPPED = "skipped"


@dataclass
class SaveResult:
    """What one fetch changed for one company."""

    jobs_returned: int = 0  # distinct jobs in the fetch
    new_job_ids: list[int] = field(default_factory=list)  # genuinely new: these go to the pre-filter
    baseline: int = 0  # saved as baseline (never alerted)
    duplicates: int = 0  # saved as re-posts of a job we already have open
    edited: int = 0  # known jobs whose ad changed
    reopened: int = 0  # closed jobs that came back
    closed: int = 0  # jobs closed by this fetch
    empty_reply_ignored: bool = False  # True if an empty reply was treated as a glitch


def save_fetched_jobs(
    session: Session,
    company: Company,
    fetched: list[NormalisedJob],
    now: datetime | None = None,
) -> SaveResult:
    """Compare one company's freshly fetched jobs with the database and record the changes.

    `now` can be given by tests; normally it is the current UTC time.
    """
    now = now or utcnow()
    result = SaveResult()
    is_baseline_run = not company.baseline_done

    fetched = _drop_repeated_ids(fetched, company)
    result.jobs_returned = len(fetched)

    known = {job.external_id: job for job in session.scalars(select(Job).where(Job.company_id == company.id))}
    open_by_fingerprint = _open_jobs_by_fingerprint(known.values())

    for item in fetched:
        job = known.get(item.external_id)
        if job is None:
            job = _add_new_job(session, company, item, now, is_baseline_run, open_by_fingerprint, result)
            known[item.external_id] = job
        else:
            _update_known_job(job, item, now, is_baseline_run, open_by_fingerprint, result)

    seen_ids = {item.external_id for item in fetched}
    _count_missing_jobs(company, known.values(), seen_ids, now, result)

    if is_baseline_run:
        company.baseline_done = True
    return result


# --- New jobs ------------------------------------------------------------------


def _add_new_job(
    session: Session,
    company: Company,
    item: NormalisedJob,
    now: datetime,
    is_baseline_run: bool,
    open_by_fingerprint: dict[str, Job],
    result: SaveResult,
) -> Job:
    description_text = html_to_text(item.description_html)
    job = Job(
        company_id=company.id,
        ats_type=company.ats_type,
        external_id=item.external_id,
        url=item.url,
        title=item.title,
        location_raw=item.location_raw,
        country_codes=parse_location(item.location_raw).countries,
        description_text=description_text,
        posted_at=item.posted_at,
        first_seen_at=now,
        became_new_at=now,
        last_seen_at=now,
        content_hash=_raw_hash(item),
        fingerprint=job_fingerprint(company.id, item.title, item.location_raw),
        is_baseline=is_baseline_run,
    )
    original = open_by_fingerprint.get(job.fingerprint)
    if original is not None:
        job.duplicate_of_id = original.id  # kept even for baseline jobs, so a saved job can be followed

    if is_baseline_run:
        _skip(job, "baseline")
        result.baseline += 1
    elif original is not None:
        _skip(job, f"duplicate of job {original.id}")
        result.duplicates += 1

    session.add(job)
    session.flush()  # gives the job its id
    if job.prefilter_status != SKIPPED:
        result.new_job_ids.append(job.id)
    open_by_fingerprint.setdefault(job.fingerprint, job)
    return job


# --- Jobs we already know --------------------------------------------------------


def _update_known_job(
    job: Job,
    item: NormalisedJob,
    now: datetime,
    is_baseline_run: bool,
    open_by_fingerprint: dict[str, Job],
    result: SaveResult,
) -> None:
    job.last_seen_at = now
    job.missing_count = 0
    job.url = item.url  # the apply link can move without the ad changing
    if job.posted_at is None:
        job.posted_at = item.posted_at

    new_hash = _raw_hash(item)
    if new_hash != job.content_hash:
        # The company edited the ad: keep our copy current, but don't alert again.
        job.title = item.title
        job.location_raw = item.location_raw
        job.country_codes = parse_location(item.location_raw).countries
        job.description_text = html_to_text(item.description_html)
        job.content_hash = new_hash
        job.fingerprint = job_fingerprint(job.company_id, item.title, item.location_raw)
        result.edited += 1

    if job.closed_at is not None:
        _reopen(job, now, is_baseline_run, open_by_fingerprint, result)


def _reopen(
    job: Job,
    now: datetime,
    is_baseline_run: bool,
    open_by_fingerprint: dict[str, Job],
    result: SaveResult,
) -> None:
    """A closed job is back. Short gap: reopen quietly. Long gap: treat it as a fresh posting."""
    was_closed_for = now - job.closed_at
    job.closed_at = None
    result.reopened += 1
    if is_baseline_run or was_closed_for < REPROCESS_AFTER_CLOSED:
        open_by_fingerprint.setdefault(job.fingerprint, job)
        return

    # Fresh posting: it follows exactly the same rules as a brand-new job.
    job.is_baseline = False
    original = open_by_fingerprint.get(job.fingerprint)
    if original is not None and original.id != job.id:
        job.duplicate_of_id = original.id
        _skip(job, f"duplicate of job {original.id}")
        result.duplicates += 1
    else:
        job.duplicate_of_id = None
        job.prefilter_status = "pending"
        job.prefilter_reason = None
        job.analysis_status = "pending"
        job.matched_at = None  # matched again once analysed
        job.became_new_at = now  # first_seen_at keeps the first time; this records the comeback
        result.new_job_ids.append(job.id)
        open_by_fingerprint.setdefault(job.fingerprint, job)


# --- Jobs that have disappeared ------------------------------------------------


def _count_missing_jobs(
    company: Company,
    known: Iterable[Job],
    seen_ids: set[str],
    now: datetime,
    result: SaveResult,
) -> None:
    open_jobs = [job for job in known if job.closed_at is None]
    missing = [job for job in open_jobs if job.external_id not in seen_ids]
    if not missing:
        return

    if not seen_ids:
        # The company showed no jobs at all. Usually a glitch: only act once it has lasted a while.
        last_seen_any = max(job.last_seen_at for job in open_jobs)
        if now - last_seen_any < EMPTY_REPLY_GRACE:
            logger.warning(
                "%s returned no jobs but has %d open; not closing anything yet", company.name, len(open_jobs)
            )
            result.empty_reply_ignored = True
            return

    for job in missing:
        job.missing_count += 1
        if job.missing_count >= CLOSE_AFTER_MISSING:
            job.closed_at = now
            result.closed += 1


# --- Small helpers ---------------------------------------------------------------


def _drop_repeated_ids(fetched: list[NormalisedJob], company: Company) -> list[NormalisedJob]:
    """Keep the first copy of each job ID; a feed listing a job twice must not break saving."""
    unique: dict[str, NormalisedJob] = {}
    for item in fetched:
        unique.setdefault(item.external_id, item)
    repeats = len(fetched) - len(unique)
    if repeats:
        logger.warning("%s listed %d job(s) more than once; kept the first copy", company.name, repeats)
    return list(unique.values())


def _open_jobs_by_fingerprint(jobs: Iterable[Job]) -> dict[str, Job]:
    """Open jobs by fingerprint. If several share one, the earliest seen is the original."""
    by_fingerprint: dict[str, Job] = {}
    for job in sorted(jobs, key=lambda job: (job.first_seen_at, job.id)):
        if job.closed_at is None and job.fingerprint:
            by_fingerprint.setdefault(job.fingerprint, job)
    return by_fingerprint


def _raw_hash(item: NormalisedJob) -> str:
    """Changes whenever the company edits the ad.

    Hashes the description's raw HTML rather than our cleaned text, so spotting an
    edit is instant; the slower HTML-to-text conversion only runs for new or edited jobs.
    """
    return content_hash(item.title, item.location_raw, item.description_html)


def _skip(job: Job, reason: str) -> None:
    """Mark a job so the pre-filter never picks it up."""
    job.prefilter_status = SKIPPED
    job.prefilter_reason = reason


# --- The fetch cycle -------------------------------------------------------------

MAX_ERROR_CHARS = 1000  # longest error message stored on a fetch run


@dataclass
class CompanyOutcome:
    """How one company's fetch went in a cycle."""

    company_id: int
    company_name: str
    ok: bool
    error: str | None = None
    saved: SaveResult | None = None  # only for successful fetches


@dataclass
class CycleResult:
    """Everything one fetch cycle did, company by company."""

    started_at: datetime
    finished_at: datetime
    outcomes: list[CompanyOutcome] = field(default_factory=list)

    @property
    def new_job_ids(self) -> list[int]:
        """Every genuinely new job found in this cycle: the input to the pre-filter."""
        return [job_id for outcome in self.outcomes if outcome.saved for job_id in outcome.saved.new_job_ids]

    @property
    def failed(self) -> list[CompanyOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.ok]


@dataclass(frozen=True)
class _Target:
    """The few details needed to fetch one company, read before any fetching starts."""

    id: int
    name: str
    ats_type: str | None
    ats_token: str | None


@dataclass
class _Fetched:
    """The raw outcome of fetching one company: its jobs, or the error that stopped it."""

    target: _Target
    started_at: datetime
    finished_at: datetime
    jobs: list[NormalisedJob] | None = None
    error: str | None = None


async def run_fetch_cycle(
    session_factory: Callable[[], Session] | None = None,
    client: PoliteClient | None = None,
) -> CycleResult:
    """Fetch every active company once, save what changed, and record a fetch run for each.

    Both arguments are for tests; normally the real database and a new polite client are used.
    """
    session_factory = session_factory or sessionmaker(bind=get_engine(), expire_on_commit=False)
    cycle_started = utcnow()

    with session_factory() as session:
        targets = [
            _Target(company.id, company.name, company.ats_type, company.ats_token)
            for company in session.scalars(select(Company).where(Company.active).order_by(Company.id))
        ]
    logger.info("Fetch cycle started: %d active companies", len(targets))

    # First half: fetch everything at once. The client's limits keep this polite.
    if client is None:
        async with PoliteClient() as own_client:
            fetched = await asyncio.gather(*(_fetch_one(target, own_client) for target in targets))
    else:
        fetched = await asyncio.gather(*(_fetch_one(target, client) for target in targets))

    # Second half: save one company at a time, each in its own transaction.
    result = CycleResult(started_at=cycle_started, finished_at=cycle_started)
    for item in fetched:
        result.outcomes.append(_save_one(session_factory, item))
    result.finished_at = utcnow()

    logger.info(
        "Fetch cycle finished in %.1f s: %d ok, %d failed, %d new jobs",
        (result.finished_at - result.started_at).total_seconds(),
        len(result.outcomes) - len(result.failed),
        len(result.failed),
        len(result.new_job_ids),
    )
    return result


async def _fetch_one(target: _Target, client: PoliteClient) -> _Fetched:
    """Fetch one company's jobs. Never raises: any problem is returned as an error message."""
    started_at = utcnow()
    fetcher = get_fetcher(target.ats_type)
    if fetcher is None or not target.ats_token:
        return _Fetched(target, started_at, utcnow(), error=f"No fetcher or token for ATS {target.ats_type!r}")
    try:
        jobs = await fetcher.fetch(target.ats_token, client)
    except FetchError as exc:
        logger.warning("Fetching %s failed: %s", target.name, exc)
        return _Fetched(target, started_at, utcnow(), error=str(exc))
    except Exception as exc:  # a bug in our code must not stop the other companies
        logger.exception("Unexpected error while fetching %s", target.name)
        return _Fetched(target, started_at, utcnow(), error=f"Unexpected error: {type(exc).__name__}: {exc}")
    return _Fetched(target, started_at, utcnow(), jobs=jobs)


def _save_one(session_factory: Callable[[], Session], item: _Fetched) -> CompanyOutcome:
    """Save one company's results and its fetch run, as one transaction."""
    target = item.target
    if item.error is None:
        try:
            with session_factory() as session, session.begin():
                company = session.get_one(Company, target.id)
                saved = save_fetched_jobs(session, company, item.jobs or [])
                company.consecutive_failures = 0
                company.last_fetch_ok_at = item.finished_at
                session.add(_fetch_run(item, status="ok", saved=saved))
            logger.info(
                "%s: %d jobs, %d new, %d baseline, %d closed",
                target.name, saved.jobs_returned, len(saved.new_job_ids), saved.baseline, saved.closed,
            )
            return CompanyOutcome(target.id, target.name, ok=True, saved=saved)
        except Exception as exc:  # saving failed: everything for this company was undone
            logger.exception("Unexpected error while saving %s", target.name)
            item.error = f"Unexpected error while saving: {type(exc).__name__}: {exc}"

    # The fetch (or the save) failed: never close jobs, just record the failure.
    with session_factory() as session, session.begin():
        company = session.get_one(Company, target.id)
        company.consecutive_failures += 1
        session.add(_fetch_run(item, status="error"))
    return CompanyOutcome(target.id, target.name, ok=False, error=item.error)


def _fetch_run(item: _Fetched, status: str, saved: SaveResult | None = None) -> FetchRun:
    return FetchRun(
        company_id=item.target.id,
        started_at=item.started_at,
        finished_at=item.finished_at,
        status=status,
        jobs_returned=saved.jobs_returned if saved else None,
        new_jobs=len(saved.new_job_ids) if saved else None,
        error=item.error[:MAX_ERROR_CHARS] if item.error else None,
    )
