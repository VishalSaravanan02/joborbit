"""The LLM step: reads each job that passed the pre-filter and saves what it found.

run_analysis takes every open job with prefilter_status "passed" and analysis_status "pending",
newest first, and asks the LLM about each one (llm/client.py, with the prompt in llm/prompts.py).
For each job:

- a checked answer is saved in job_analysis (a reopened job already has a row: it is updated) and
  the job becomes "done";
- a request with no usable answer (refused, cut off, or invalid twice) makes the job "failed", and
  the run carries on with the next job. Failed jobs are not retried by themselves: after a prompt
  fix, retry_failed sends them back to "pending";
- if the LLM can't be used at all (no key, unreachable, or the budget is used up), the run stops
  and the remaining jobs stay "pending" for the next run.

No database write is ever open while the LLM is being asked: SQLite allows one writer at a time,
and a call can take seconds. Each job is read in one short session and saved in another.

Only the job and the names of the active users' custom roles are sent to the LLM: no user data.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from joborbit.db.models import Company, Job, JobAnalysis, User, UserProfile
from joborbit.db.session import get_engine
from joborbit.llm.client import Answer, AnswerFailed, LlmClient, LlmUnavailable
from joborbit.llm.prompts import PROMPT_VERSION, SCHEMA_NAME, instructions, job_message
from joborbit.llm.schemas import answer_schema, parse_answer
from joborbit.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

PENDING, DONE, FAILED = "pending", "done", "failed"
PASSED = "passed"  # the pre-filter's verdict that sends a job here


@dataclass
class AnalysisRun:
    """What one run did."""

    done: list[int] = field(default_factory=list)  # job ids analysed and saved
    failed: list[int] = field(default_factory=list)  # job ids with no usable answer
    stopped: str | None = None  # why the run stopped before the end, if it did
    cost_usd: float = 0.0  # estimated cost of the saved answers


def pending_job_ids(session: Session) -> list[int]:
    """Open jobs waiting for analysis, newest first."""
    return list(
        session.scalars(
            select(Job.id)
            .where(Job.prefilter_status == PASSED, Job.analysis_status == PENDING, Job.closed_at.is_(None))
            .order_by(Job.first_seen_at.desc(), Job.id.desc())
        )
    )


def active_custom_roles(session: Session) -> list[str]:
    """Every active user's custom role names, each once (capitals ignored), in alphabetical order.

    Paused users count, as in the pre-filter: pausing stops alerts, not analysis.
    """
    names: dict[str, str] = {}
    for profile in session.scalars(select(UserProfile).join(User).where(User.is_active)):
        for name in profile.custom_roles:
            names.setdefault(name.lower(), name)
    return sorted(names.values(), key=str.lower)


def retry_failed(session: Session) -> int:
    """Send every failed job back to "pending" (e.g. after a prompt fix). Returns how many. Never commits."""
    result = session.execute(update(Job).where(Job.analysis_status == FAILED).values(analysis_status=PENDING))
    return result.rowcount


def run_analysis(
    client: LlmClient,
    session_factory: Callable[[], Session] | None = None,
    limit: int | None = None,
) -> AnalysisRun:
    """Analyse waiting jobs, newest first: at most `limit` of them, if given."""
    sessions = session_factory or sessionmaker(bind=get_engine(), expire_on_commit=False)
    run = AnalysisRun()
    with sessions() as session:
        job_ids = pending_job_ids(session)[:limit]
        custom_roles = active_custom_roles(session)
    schema = answer_schema(custom_roles)

    for job_id in job_ids:
        message = _message_for(sessions, job_id, custom_roles)
        if message is None:
            continue  # analysed or closed meanwhile (e.g. by another run)
        try:
            answer = client.ask(
                instructions=instructions(),
                text=message,
                schema_name=SCHEMA_NAME,
                schema=schema,
                check=lambda text: parse_answer(text, custom_roles),
            )
        except LlmUnavailable as error:
            run.stopped = str(error)
            logger.warning("Analysis stopped after %d jobs: %s", len(run.done) + len(run.failed), error)
            break
        except AnswerFailed as error:
            _mark_failed(sessions, job_id)
            run.failed.append(job_id)
            logger.warning("No usable analysis for job %d: %s", job_id, error)
            continue
        _save(sessions, job_id, answer)
        run.done.append(job_id)
        run.cost_usd += answer.cost_usd

    logger.info(
        "Analysis: %d done, %d failed, %d waiting%s",
        len(run.done), len(run.failed), len(job_ids) - len(run.done) - len(run.failed),
        f" (stopped: {run.stopped})" if run.stopped else "",
    )
    return run


def _message_for(sessions: Callable[[], Session], job_id: int, custom_roles: list[str]) -> str | None:
    """The job's message for the LLM, read in its own short session; None if it no longer needs analysis."""
    with sessions() as session:
        row = session.execute(select(Job, Company.name).join(Company).where(Job.id == job_id)).one_or_none()
        if row is None:
            return None
        job, company = row
        if job.analysis_status != PENDING or job.closed_at is not None:
            return None
        posted = job.posted_at or job.first_seen_at
        return job_message(
            company, job.title, job.location_raw, posted.date() if posted else None, job.description_text, custom_roles
        )


def _save(sessions: Callable[[], Session], job_id: int, answer: Answer) -> None:
    """Insert or update the job's analysis and mark the job done, in one short save."""
    with sessions() as session, session.begin():
        job = session.get_one(Job, job_id)
        analysis = job.analysis or JobAnalysis(job_id=job_id)
        for name, value in answer.result.model_dump().items():
            setattr(analysis, name, value)
        analysis.model = answer.model
        analysis.prompt_version = PROMPT_VERSION
        analysis.input_tokens = answer.input_tokens
        analysis.output_tokens = answer.output_tokens
        analysis.analysed_at = utcnow()
        job.analysis = analysis
        job.analysis_status = DONE


def _mark_failed(sessions: Callable[[], Session], job_id: int) -> None:
    with sessions() as session, session.begin():
        session.get_one(Job, job_id).analysis_status = FAILED
