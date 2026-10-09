"""The dry run: what JobOrbit would send, over recent jobs, without changing anything stored.

Almost every stored job is part of a company's baseline, so the normal cycle never pre-filters,
analyses or matches it. To tune keywords, scores and thresholds on a realistic amount of data,
the dry run takes every open job posted in the last few days, baseline included, and runs the
same steps on it:

1. the pre-filter, with every active user's roles;
2. an analysis for each job that passes: the one already in the database if it was made with
   the current prompt, else a saved answer from the answers file, else a new LLM call (at most
   `limit` of them, newest jobs first);
3. matching for every active user: the hard filters, the score and the tier.

Each run can also write a review sheet (write_review): one row per user and job, plus the jobs the
pre-filter rejected as "no matching role", with an empty "want" column for a person to fill in.
Answers already given in the previous sheet are carried over, so running again never loses them.

Nothing in the database changes: no job statuses, no analyses, no matches, nothing sent. The LLM
calls still go through the normal client, so the daily cap and the monthly budget apply and the
cost is recorded in llm_usage. New answers are kept in the answers file instead
(var/dry_run/answers.json), so running again after a tuning change only asks about jobs it hasn't
seen. A saved answer is reused only while the prompt version, the job's text and the custom roles
it was asked with are all unchanged.
"""

import csv
import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from joborbit.db.models import Company, Job, JobAnalysis
from joborbit.llm.client import AnswerFailed, LlmClient, LlmUnavailable
from joborbit.llm.prompts import PROMPT_VERSION, SCHEMA_NAME, instructions, job_message
from joborbit.llm.schemas import JobAnalysisResult, answer_schema, parse_answer
from joborbit.matching.facts import JobFacts, UserFacts
from joborbit.matching.filters import check_job as check_filters
from joborbit.matching.matcher import active_users
from joborbit.matching.router import route
from joborbit.matching.scoring import score_job
from joborbit.pipeline.analyse import active_custom_roles
from joborbit.pipeline.prefilter import check_job as prefilter_job
from joborbit.pipeline.prefilter import reason_kind, rules_for_active_users
from joborbit.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_ANSWERS_FILE = PROJECT_ROOT / "var" / "dry_run" / "answers.json"
DEFAULT_REVIEW_FILE = PROJECT_ROOT / "var" / "dry_run" / "review.csv"
NO_ROLE = "no matching role"  # the pre-filter reason whose jobs go in the review sheet
DEFAULT_DAYS = 14
DEFAULT_LIMIT = 100  # jobs asked about per run at most (a retried answer is two calls for one job)


# --- The answers file ----------------------------------------------------------------------------


class AnswerStore:
    """LLM answers kept in a JSON file, by job id, so a dry run never pays twice for the same answer.

    An answer is reused only while the prompt version, the job's text (its content hash) and the
    custom roles it was asked with are the same as now; otherwise the job is asked about again.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._answers: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._answers = json.loads(path.read_text(encoding="utf-8")).get("answers", {})

    def get(self, job: Job, custom_roles: list[str]) -> JobAnalysisResult | None:
        saved = self._answers.get(str(job.id))
        if saved is None or saved["key"] != _key(job, custom_roles):
            return None
        return JobAnalysisResult.model_validate(saved["result"], context={"custom_roles": custom_roles})

    def put(self, job: Job, custom_roles: list[str], result: JobAnalysisResult, model: str) -> None:
        """Keep an answer and save the file at once: the money is already spent."""
        self._answers[str(job.id)] = {
            "key": _key(job, custom_roles),
            "model": model,
            "result": result.model_dump(mode="json"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"answers": self._answers}, indent=1, ensure_ascii=False) + "\n", "utf-8")
        os.replace(temporary, self.path)  # all or nothing: a crash never leaves half a file

    def __len__(self) -> int:
        return len(self._answers)


def _key(job: Job, custom_roles: list[str]) -> dict[str, Any]:
    return {"prompt_version": PROMPT_VERSION, "content_hash": job.content_hash, "custom_roles": custom_roles}


# --- The results ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DryResult:
    """What one user would get for one analysed job: a tier, or the hard filter it broke."""

    user_name: str
    job_id: int
    company: str
    title: str
    url: str
    posted_at: datetime
    tier: str | None  # "instant", "digest" or "silent"; None when dropped
    score: int | None
    points: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    dropped: str | None = None  # the hard filter's reason, when dropped
    location: str | None = None  # as the company wrote it
    seniority: str | None = None  # from the analysis
    experience_years: int | None = None
    experience_mandatory: bool = False


@dataclass(frozen=True)
class Unmatched:
    """A job the pre-filter rejected because its title matched none of the roles: worth a look by a person."""

    job_id: int
    company: str
    title: str
    location: str | None
    url: str
    posted_at: datetime


@dataclass
class DryRun:
    """Everything one dry run found."""

    since: datetime
    until: datetime
    jobs: int = 0  # open jobs posted in the window
    no_date: int = 0  # open jobs left out because the ATS gives no posting date
    passed: int = 0  # jobs that passed the pre-filter
    rejected: Counter[str] = field(default_factory=Counter)  # by kind of reason
    from_database: int = 0  # analyses reused from the database
    from_file: int = 0  # answers reused from the answers file
    asked: int = 0  # new answers from the LLM
    failed: int = 0  # LLM calls with no usable answer
    not_analysed: int = 0  # passed, but no answer: over the limit, analysis off, or stopped
    stopped: str | None = None  # why the LLM stopped, if it did
    cost_usd: float = 0.0
    results: list[DryResult] = field(default_factory=list)
    unmatched: list[Unmatched] = field(default_factory=list)  # pre-filter: no matching role

    @property
    def days(self) -> float:
        return (self.until - self.since).total_seconds() / 86400


# --- The run ----------------------------------------------------------------------------------------


def recent_jobs(session: Session, since: datetime) -> list[tuple[Job, Company]]:
    """Open jobs posted since `since`, baseline included, newest first."""
    rows = session.execute(
        select(Job, Company)
        .join(Company, Job.company_id == Company.id)
        .where(Job.closed_at.is_(None), Job.posted_at >= since)
        .options(selectinload(Job.analysis))
        .order_by(Job.posted_at.desc(), Job.id.desc())
    )
    return [(job, company) for job, company in rows]


def count_without_date(session: Session) -> int:
    """Open jobs whose ATS gives no posting date: they can't be placed in a window, so they are left out."""
    return session.scalar(select(func.count(Job.id)).where(Job.closed_at.is_(None), Job.posted_at.is_(None))) or 0


def run_dry_run(
    session: Session,
    now: datetime,
    days: int = DEFAULT_DAYS,
    limit: int = DEFAULT_LIMIT,
    client: LlmClient | None = None,
    client_factory: Callable[[], LlmClient] | None = None,
    store: AnswerStore | None = None,
) -> DryRun:
    """Run the dry run over the last `days` days. Changes nothing in the database.

    New answers are asked for only when a `client` (or a `client_factory`, called the first time one
    is needed) is given, for at most `limit` jobs; without one, only saved answers are used.
    """
    since = now - timedelta(days=days)
    run = DryRun(since=since, until=now)
    store = store if store is not None else AnswerStore(DEFAULT_ANSWERS_FILE)  # an empty store is falsy
    custom_roles = active_custom_roles(session)
    schema = answer_schema(custom_roles)
    rules = rules_for_active_users(session)
    users = active_users(session)

    rows = recent_jobs(session, since)
    run.jobs, run.no_date = len(rows), count_without_date(session)

    for job, company in rows:
        decision = prefilter_job(job.title, job.location_raw, rules)
        if not decision.passed:
            run.rejected[reason_kind(decision.reason)] += 1
            if decision.reason == NO_ROLE:
                run.unmatched.append(
                    Unmatched(job.id, company.name, job.title, job.location_raw, job.url, job.posted_at)
                )
            continue
        run.passed += 1

        analysis = _analysis_for(job, custom_roles, store, run)
        if analysis is None and limit > run.asked + run.failed and run.stopped is None:
            if client is None and client_factory is not None:
                try:
                    client = client_factory()
                except LlmUnavailable as error:
                    run.stopped, client_factory = str(error), None
            if client is not None:
                analysis = _ask(client, job, company, custom_roles, schema, store, run)
        if analysis is None:
            run.not_analysed += 1
            continue

        facts = JobFacts.from_rows(job, analysis, company)
        for user, user_facts in users:
            run.results.append(_result_for(job, company, facts, user.display_name, user_facts))

    session.rollback()  # nothing was changed, and nothing is kept
    return run


def _analysis_for(job: Job, custom_roles: list[str], store: AnswerStore, run: DryRun) -> JobAnalysis | None:
    """The job's analysis from the database or the answers file, if there is a current one."""
    if job.analysis is not None and job.analysis.prompt_version == PROMPT_VERSION:
        run.from_database += 1
        return job.analysis
    saved = store.get(job, custom_roles)
    if saved is not None:
        run.from_file += 1
        return _unsaved_analysis(job, saved)
    return None


def _ask(
    client: LlmClient,
    job: Job,
    company: Company,
    custom_roles: list[str],
    schema: dict[str, Any],
    store: AnswerStore,
    run: DryRun,
) -> JobAnalysis | None:
    posted = job.posted_at or job.first_seen_at
    message = job_message(company.name, job.title, job.location_raw, posted.date(), job.description_text, custom_roles)
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
        logger.warning("Dry run: analysis stopped: %s", error)
        return None
    except AnswerFailed as error:
        run.failed += 1
        logger.warning("Dry run: no usable analysis for job %d: %s", job.id, error)
        return None
    run.asked += 1
    run.cost_usd += answer.cost_usd
    store.put(job, custom_roles, answer.result, answer.model)
    return _unsaved_analysis(job, answer.result)


def _unsaved_analysis(job: Job, result: JobAnalysisResult) -> JobAnalysis:
    """An analysis object for matching only: never added to the session, so never saved."""
    return JobAnalysis(job_id=job.id, **result.model_dump())


def _result_for(job: Job, company: Company, facts: JobFacts, user_name: str, user_facts: UserFacts) -> DryResult:
    base = {
        "user_name": user_name, "job_id": job.id, "company": company.name, "title": job.title,
        "url": job.url, "posted_at": job.posted_at, "location": job.location_raw, "seniority": facts.seniority,
        "experience_years": facts.experience_years, "experience_mandatory": facts.experience_mandatory,
    }
    verdict = check_filters(facts, user_facts)
    if not verdict.passed:
        return DryResult(**base, tier=None, score=None, dropped=verdict.reason)
    score = score_job(facts, user_facts, verdict)
    tier = route(score, verdict, user_facts.alert_style)
    return DryResult(
        **base, tier=tier, score=score.score, points=score.points,
        reasons=tuple(score.reasons), notes=tuple(score.notes),
    )


# --- The review sheet ----------------------------------------------------------------------------------

PART_NAMES = {
    "role_fit": "role", "entry_fit": "level", "skills": "skills", "country": "country", "transfer": "transfer",
    "favourite_company": "favourite", "preferred_industry": "industry",
}
REVIEW_COLUMNS = [
    "want", "stage", "score", "user", "company", "title", "location", "posted", "level", "experience",
    "points", "why", "url", "job id",
]
STAGE_ORDER = {"instant": 0, "digest": 1, "silent": 2, "dropped": 3, "prefilter": 4}


def points_line(points: dict[str, float]) -> str:
    """ "role 30 | level 6.25 | skills 13.33 | country 15 | transfer 0" """
    return " | ".join(f"{PART_NAMES.get(part, part)} {value:g}" for part, value in points.items())


def experience_words(mandatory: bool, years: int | None) -> str:
    """The experience asked for, in a few words: "required, no number", "2+ years preferred", "none asked"."""
    if years is None:
        return "required, no number" if mandatory else "none asked"
    return f"{years}+ years {'required' if mandatory else 'preferred'}"


def review_rows(run: DryRun) -> list[dict[str, str]]:
    """Every user's results, then the pre-filter's "no matching role" jobs, best first, without answers."""
    rows = []
    for result in run.results:
        stage = result.tier or "dropped"
        why = "; ".join([*result.reasons, *result.notes]) if result.tier else result.dropped
        rows.append({
            "want": "", "stage": stage, "score": "" if result.score is None else str(result.score),
            "user": result.user_name, "company": result.company, "title": result.title,
            "location": result.location or "", "posted": f"{result.posted_at:%Y-%m-%d}",
            "level": result.seniority or "unclear",
            "experience": experience_words(result.experience_mandatory, result.experience_years),
            "points": points_line(result.points), "why": why or "", "url": result.url, "job id": str(result.job_id),
        })
    for job in run.unmatched:
        rows.append({
            "want": "", "stage": "prefilter", "score": "", "user": "", "company": job.company, "title": job.title,
            "location": job.location or "", "posted": f"{job.posted_at:%Y-%m-%d}", "level": "", "experience": "",
            "points": "", "why": f"pre-filter: {NO_ROLE}", "url": job.url, "job id": str(job.job_id),
        })
    rows.sort(key=lambda row: (STAGE_ORDER[row["stage"]], -int(row["score"] or 0), row["company"], row["title"]))
    return rows


def write_review(run: DryRun, path: Path) -> tuple[int, int]:
    """Write the review sheet, keeping every "want" answer from the sheet already there.

    Answers are matched by user and job id, so they follow a job whatever its new tier or score.
    Returns (rows written, answers kept). Written with a byte-order mark so Excel reads the accents.
    """
    answers: dict[tuple[str, str], str] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as file:
            for old in csv.DictReader(file):
                if old.get("want", "").strip():
                    answers[(old.get("user", ""), old.get("job id", ""))] = old["want"].strip()

    rows = review_rows(run)
    kept = 0
    for row in rows:
        answer = answers.get((row["user"], row["job id"]))
        if answer:
            row["want"] = answer
            kept += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)  # a crash never leaves half a sheet, or loses the answers
    return len(rows), kept
