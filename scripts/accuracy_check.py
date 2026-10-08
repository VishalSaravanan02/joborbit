"""The accuracy check: score the LLM against real jobs that you have labelled by hand.

Usage:
    python scripts/accuracy_check.py pick [--count 30] [--per-company 3]
    python scripts/accuracy_check.py run [--effort none low medium]
    python scripts/accuracy_check.py score

pick   Chooses jobs at random (the same ones every time) from the open jobs that pass the
       pre-filter, at most --per-company from one company, and writes them to
       var/accuracy/jobs.yaml with every label TODO. Changes nothing in the database, and never
       replaces an existing jobs file.
run    Asks the LLM about every job, once per reasoning effort (default: the one in
       settings.yaml), prints the score, and saves the answers to var/accuracy/answers-<effort>.json.
       Every job must be labelled first. Each job is one call, or two if its answer is retried.
score  Scores the saved answers again against the current labels. Free: use it after
       correcting a label.

--folder chooses another folder than var/accuracy (used by the tests).

The target is that at least 90% of the jobs (27 of 30) are right on all five scored things:
country, seniority, experience, languages and role.
"""

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.config import load_app_settings
from joborbit.db.models import Company, Job
from joborbit.db.session import get_engine
from joborbit.llm.accuracy import (
    SCORED,
    EvalJob,
    Outcome,
    Summary,
    load_outcomes,
    read_jobs,
    save_outcomes,
    summarise,
    write_jobs,
)
from joborbit.llm.client import AnswerFailed, LlmClient, LlmUnavailable, make_client
from joborbit.llm.prompts import PROMPT_VERSION, SCHEMA_NAME, instructions, job_message
from joborbit.llm.schemas import answer_schema, parse_answer
from joborbit.pipeline.prefilter import check_job, rules_for_active_users
from joborbit.settings import PROJECT_ROOT

DEFAULT_FOLDER = PROJECT_ROOT / "var" / "accuracy"
EFFORTS = ("none", "low", "medium", "high")  # cheapest first
TARGET_SHARE = 0.9
SEED = "accuracy-check-1"  # the same seed always picks the same jobs


# --- pick ---------------------------------------------------------------------------------------


def candidates(session: Session) -> list[tuple[Company, Job]]:
    """Every open job that would pass the pre-filter for the active users, baseline jobs included."""
    rules = rules_for_active_users(session)
    rows = session.execute(
        select(Company, Job).join(Company, Job.company_id == Company.id).where(Job.closed_at.is_(None)).order_by(Job.id)
    )
    return [(company, job) for company, job in rows if check_job(job.title, job.location_raw, rules).passed]


def choose(rows: list[tuple[Company, Job]], count: int, per_company: int) -> list[tuple[Company, Job]]:
    """Up to `count` rows, picked at random but the same every time, at most `per_company` per company."""
    shuffled = list(rows)
    random.Random(SEED).shuffle(shuffled)
    chosen: list[tuple[Company, Job]] = []
    taken: Counter[int] = Counter()
    for company, job in shuffled:
        if len(chosen) == count:
            break
        if taken[company.id] < per_company:
            chosen.append((company, job))
            taken[company.id] += 1
    return sorted(chosen, key=lambda row: (row[0].name.lower(), row[1].title.lower()))


def as_eval_job(company: Company, job: Job) -> EvalJob:
    posted = job.posted_at or job.first_seen_at
    return EvalJob(
        id=f"{company.slug}/{job.external_id}",
        company=company.name,
        title=job.title,
        location=job.location_raw,
        posted=posted.date() if posted else None,
        url=job.url,
        description=job.description_text,
        expected={},
    )


def pick(folder: Path, count: int, per_company: int) -> None:
    path = folder / "jobs.yaml"
    if path.exists():
        raise SystemExit(f"{path} already exists and may hold your labels; nothing was changed.")
    with Session(get_engine()) as session:  # closes without saving: picking changes nothing
        rows = candidates(session)
        chosen = choose(rows, count, per_company)
        jobs = [as_eval_job(company, job) for company, job in chosen]
    write_jobs(path, jobs)
    companies = len({job.id.split("/")[0] for job in jobs})
    print(f"Picked {len(jobs)} of the {len(rows)} open jobs that pass the pre-filter, from {companies} companies.")
    print(f"Written to {path}")
    print("Next: replace every TODO with your labels, then run:  python scripts/accuracy_check.py run")


# --- run ----------------------------------------------------------------------------------------


def labelled_jobs(folder: Path) -> list[EvalJob]:
    """The jobs, refusing to go on unless every one is fully and correctly labelled."""
    path = folder / "jobs.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run first:  python scripts/accuracy_check.py pick")
    jobs = read_jobs(path)
    problems = []
    for job in jobs:
        try:
            job.labels()
        except ValueError as error:
            problems.append(str(error))
    if problems:
        shown = "\n".join(f"  {problem}" for problem in problems[:10])
        more = f"\n  ...and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise SystemExit(f"{len(problems)} of {len(jobs)} jobs need their labels fixed first:\n{shown}{more}")
    return jobs


def ask_about(client: LlmClient, jobs: list[EvalJob]) -> tuple[list[Outcome], str]:
    """Ask the LLM about every job exactly as analysis will. Stops at once if the LLM is unavailable.

    Returns the outcomes, and the model's name as the provider reported it (or as set, if no job got an answer).
    """
    outcomes = []
    model = client.config.model
    for job in jobs:
        started = time.monotonic()
        message = job_message(job.company, job.title, job.location, job.posted, job.description)
        try:
            answer = client.ask(
                instructions=instructions(),
                text=message,
                schema_name=SCHEMA_NAME,
                schema=answer_schema(),
                check=parse_answer,
            )
        except AnswerFailed as error:
            outcomes.append(Outcome(job.id, None, str(error), seconds=time.monotonic() - started))
            continue
        model = answer.model
        outcomes.append(
            Outcome(
                job.id,
                answer.result.model_dump(mode="json"),
                None,
                answer.input_tokens,
                answer.output_tokens,
                answer.cost_usd,
                time.monotonic() - started,
            )
        )
    return outcomes, model


def answers_path(folder: Path, effort: str) -> Path:
    return folder / f"answers-{effort}.json"


def run(folder: Path, efforts: list[str]) -> None:
    efforts = list(dict.fromkeys(efforts))  # each effort once, in the order given
    jobs = labelled_jobs(folder)
    print(f"Asking about each job at effort {', '.join(efforts)}: at least {len(jobs) * len(efforts)} calls.\n")
    results = {}
    for effort in efforts:
        try:
            outcomes, model = ask_about(make_client(reasoning_effort=effort), jobs)
        except LlmUnavailable as error:
            raise SystemExit(f"Stopped during effort {effort!r} (nothing saved for it): {error}") from error
        save_outcomes(answers_path(folder, effort), effort, PROMPT_VERSION, model, outcomes)
        results[effort] = ({"effort": effort, "prompt_version": PROMPT_VERSION, "model": model}, outcomes)
    print_results(jobs, results)


# --- score --------------------------------------------------------------------------------------


def score_saved(folder: Path) -> None:
    jobs = labelled_jobs(folder)
    results = {}
    for effort in EFFORTS:
        path = answers_path(folder, effort)
        if path.exists():
            results[effort] = load_outcomes(path)
    if not results:
        raise SystemExit(f"No saved answers in {folder}. Run first:  python scripts/accuracy_check.py run")
    print_results(jobs, results)


# --- The report ---------------------------------------------------------------------------------


def target(jobs: int) -> int:
    return math.ceil(jobs * TARGET_SHARE)


def describe(summary: Summary, details: dict[str, str], titles: dict[str, str]) -> str:
    """One run's score, how each scored thing did, its cost per job, and every mistake."""
    jobs = max(summary.jobs, 1)
    lines = [
        f"Effort {details['effort']!r} (prompt version {details['prompt_version']}, model {details['model']}): "
        f"{summary.right} of {summary.jobs} jobs right on all five (target: {target(summary.jobs)})",
    ]
    for name in SCORED:
        lines.append(f"  {name:<11} {summary.by_field[name]:>3} of {summary.jobs}")
    lines.append(f"  no answer   {summary.failed:>3}")
    lines.append(
        f"  per job: {summary.input_tokens / jobs:,.0f} input tokens, "
        f"{summary.output_tokens / jobs:,.0f} output tokens, "
        f"about ${summary.cost_usd / jobs:.5f}, {summary.seconds / jobs:.1f} s"
    )
    if summary.wrong:
        lines.append("  Mistakes (answer, then your label):")
        for job_id, name, given, label in summary.wrong:
            lines.append(f"    {job_id}  {titles.get(job_id, '')}")
            lines.append(f"      {name}: {given!r}  /  label: {label!r}")
    return "\n".join(lines)


def print_results(jobs: list[EvalJob], results: dict[str, tuple[dict[str, str], list[Outcome]]]) -> None:
    titles = {job.id: job.title for job in jobs}
    summaries = {}
    for effort, (details, outcomes) in results.items():
        summary = summarise(jobs, outcomes)
        summaries[effort] = summary
        print(describe(summary, details, titles))
        print()
    if len(summaries) > 1:
        print("Compared:")
        for effort, summary in summaries.items():
            per_job = summary.cost_usd / max(summary.jobs, 1)
            print(f"  {effort:<7} {summary.right:>3} of {summary.jobs} right   about ${per_job:.5f} per job")
    good = [effort for effort in EFFORTS if effort in summaries and summaries[effort].right >= target(len(jobs))]
    if good:
        print(f"Cheapest effort reaching the target: {good[0]}")
    else:
        print("No effort reached the target yet: improve the prompt (llm/prompts.py) and run again.")


# --- Command line -----------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="where the jobs and answers are kept")
    commands = parser.add_subparsers(dest="command", required=True)
    pick_command = commands.add_parser("pick", help="choose the jobs and write them with blank labels")
    pick_command.add_argument("--count", type=int, default=30)
    pick_command.add_argument("--per-company", type=int, default=3)
    run_command = commands.add_parser("run", help="ask the LLM about every job and score the answers")
    run_command.add_argument("--effort", nargs="+", choices=EFFORTS, help="default: the effort in settings.yaml")
    commands.add_parser("score", help="score the saved answers again (free)")
    args = parser.parse_args()

    if args.command == "pick":
        pick(args.folder, args.count, args.per_company)
    elif args.command == "run":
        run(args.folder, args.effort or [load_app_settings().llm.reasoning_effort])
    else:
        score_saved(args.folder)


if __name__ == "__main__":
    main()
