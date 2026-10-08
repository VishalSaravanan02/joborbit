"""The accuracy check: real jobs labelled by hand, and how the LLM's answers are scored against them.

The jobs and their labels live in one YAML file (by default var/accuracy/jobs.yaml, which Git
ignores: the postings are other companies' text, and the repository is public). Labels start as
TODO, and are filled in by hand following the rules in llm/prompts.py, before seeing any answer.

Five things are scored for each job. A job is right only if all five are:

- countries: the same set as the label.
- seniority: one of the accepted values in the label.
- experience: the same "required or not" and the same number of years (or both not stated).
- languages: the same set as the label.
- roles: every role in the answer is one the label accepts, and at least one is given when the
  label accepts any (none when it accepts none).

Answers from a run are saved, so labels can be corrected and the answers scored again for free.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from joborbit.llm.schemas import MAX_EXPERIENCE_YEARS, JobAnalysisResult, country_codes, language_codes, role_slugs

TODO = "TODO"  # a label not filled in yet
SCORED = ("countries", "seniority", "experience", "languages", "roles")
LABELS = (
    "countries", "seniority", "experience_mandatory", "experience_years", "required_languages", "role_families",
)

Seniority = Literal["intern", "graduate", "entry", "mid", "senior"]


def _as_list(value: object) -> object:
    """Let a label give one value instead of a list: "graduate" means ["graduate"]."""
    return value if isinstance(value, list) else [value]


def _known(values: list[str], allowed: list[str], what: str) -> list[str]:
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; choose from: {', '.join(allowed)}")
    return values


class Expected(BaseModel):
    """The labels for one job: the answers a careful reader would give."""

    model_config = ConfigDict(extra="forbid")

    countries: list[str]
    seniority: list[Seniority | None] = Field(min_length=1)  # every value counted as right
    experience_mandatory: bool
    experience_years: int | None = Field(ge=0, le=MAX_EXPERIENCE_YEARS)
    required_languages: list[str]
    role_families: list[str]  # every role counted as right; [] means none of ours fits

    @field_validator("seniority", mode="before")
    @classmethod
    def _one_or_more(cls, value: object) -> object:
        return _as_list(value)

    @field_validator("countries")
    @classmethod
    def _check_countries(cls, values: list[str]) -> list[str]:
        return _known([value.upper() for value in values], country_codes(), "country")

    @field_validator("required_languages")
    @classmethod
    def _check_languages(cls, values: list[str]) -> list[str]:
        return _known([value.lower() for value in values], language_codes(), "language")

    @field_validator("role_families")
    @classmethod
    def _check_roles(cls, values: list[str]) -> list[str]:
        return _known([value.lower() for value in values], role_slugs(), "role")


class EvalJob(BaseModel):
    """One job in the accuracy check: what is sent to the LLM, and the labels (possibly still TODO)."""

    model_config = ConfigDict(extra="forbid")

    id: str  # "<company slug>/<the job's ID on its ATS>"
    company: str
    title: str
    location: str | None
    posted: date | None
    url: str
    description: str | None
    expected: dict[str, Any]

    def missing_labels(self) -> list[str]:
        """The labels still TODO (or left out)."""
        return [name for name in LABELS if self.expected.get(name, TODO) == TODO]

    def labels(self) -> Expected:
        """The checked labels. Raises ValueError, naming the job, if any is missing or wrong."""
        missing = self.missing_labels()
        if missing:
            raise ValueError(f"{self.id}: not labelled yet: {', '.join(missing)}")
        try:
            return Expected.model_validate(self.expected)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg'].removeprefix('Value error, ')}"
                for problem in error.errors()
            )
            raise ValueError(f"{self.id}: {problems}") from error


# --- The jobs file -------------------------------------------------------------------------------

FILE_HEADER = """\
# The accuracy check's jobs (scripts/accuracy_check.py). Not in Git: the postings belong to the
# companies that wrote them.
#
# Label each job by replacing every TODO, following the rules in src/joborbit/llm/prompts.py.
# Label before looking at any LLM answer. Where two answers are both fair, list both.
#   countries:            the countries among ours where the job can be done, e.g. [GB]; [] if none
#   seniority:            accepted levels: intern, graduate, entry, mid, senior (or null),
#                         e.g. graduate or [graduate, entry]
#   experience_mandatory: true only if experience is clearly required; false if preferred or not mentioned
#   experience_years:     the smallest number of years mentioned; 0 if none needed; null if not mentioned
#   required_languages:   only languages stated as required, e.g. [] or [es]; English only if stated
#   role_families:        every role slug you'd accept, e.g. [data_analyst]; [] if none of ours fits
"""

LABEL_LINES = """\
  expected:
    countries: TODO
    seniority: TODO
    experience_mandatory: TODO
    experience_years: TODO
    required_languages: TODO
    role_families: TODO
"""


def _quoted(value: str | None) -> str:
    """A YAML value that reads back exactly (JSON strings are valid YAML)."""
    return "null" if value is None else json.dumps(value, ensure_ascii=False)


def _block(text: str | None) -> str:
    """Multi-line text as a YAML literal block, so the posting stays easy to read while labelling.

    "|2-" says: the text is indented 2 spaces more than its key, and has no newline at the end.
    Stating the indent keeps a first line that begins with spaces from being misread.
    """
    if text is None:
        return " null"
    lines = "\n".join(f"    {line}" if line.strip() else "" for line in text.split("\n"))
    return f" |2-\n{lines}"


def write_jobs(path: Path, jobs: list[EvalJob]) -> None:
    """Write the jobs with every label TODO. Refuses to replace an existing file (it may hold labels)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; move it away first if you really want new jobs")
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [FILE_HEADER]
    for number, job in enumerate(jobs, start=1):
        parts.append(
            f"\n# --- Job {number} of {len(jobs)} ---\n"
            f"- id: {_quoted(job.id)}\n"
            f"  company: {_quoted(job.company)}\n"
            f"  title: {_quoted(job.title)}\n"
            f"  location: {_quoted(job.location)}\n"
            f"  posted: {job.posted.isoformat() if job.posted else 'null'}\n"
            f"  url: {_quoted(job.url)}\n"
            f"  description:{_block(job.description)}\n"
            f"{LABEL_LINES}"
        )
    path.write_text("".join(parts), encoding="utf-8")


def read_jobs(path: Path) -> list[EvalJob]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or []
    return [EvalJob.model_validate(item) for item in data]


# --- Scoring ------------------------------------------------------------------------------------


def _roles_right(given: list[str], accepted: list[str]) -> bool:
    if not accepted:
        return not given
    return bool(given) and set(given) <= set(accepted)


def score(expected: Expected, answer: JobAnalysisResult) -> dict[str, bool]:
    """Which of the five scored things the answer got right."""
    return {
        "countries": set(answer.countries) == set(expected.countries),
        "seniority": answer.seniority in expected.seniority,
        "experience": (answer.experience_mandatory, answer.experience_years)
        == (expected.experience_mandatory, expected.experience_years),
        "languages": set(answer.required_languages) == set(expected.required_languages),
        "roles": _roles_right(answer.role_families, expected.role_families),
    }


@dataclass
class Outcome:
    """What happened for one job in a run: the answer (or why there was none) and what it cost."""

    job_id: str
    answer: dict[str, Any] | None  # the checked answer, as JSON-ready values
    error: str | None = None  # why there is no answer (e.g. it broke the rules twice)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass
class Summary:
    """How one run scored."""

    jobs: int = 0
    right: int = 0  # jobs right on all five
    by_field: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SCORED, 0))
    failed: int = 0  # jobs with no usable answer (counted as wrong)
    wrong: list[tuple[str, str, Any, Any]] = field(default_factory=list)  # (job id, field, answer, label)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


def compared(item: Expected | JobAnalysisResult, name: str) -> Any:
    """What is compared for one scored thing, from the labels or from an answer (same field names)."""
    return {
        "countries": item.countries,
        "seniority": item.seniority,
        "experience": (item.experience_mandatory, item.experience_years),
        "languages": item.required_languages,
        "roles": item.role_families,
    }[name]


def summarise(jobs: list[EvalJob], outcomes: list[Outcome]) -> Summary:
    """Score every outcome against its job's labels."""
    by_id = {job.id: job for job in jobs}
    summary = Summary()
    for outcome in outcomes:
        expected = by_id[outcome.job_id].labels()
        summary.jobs += 1
        summary.input_tokens += outcome.input_tokens
        summary.output_tokens += outcome.output_tokens
        summary.cost_usd += outcome.cost_usd
        summary.seconds += outcome.seconds
        if outcome.answer is None:
            summary.failed += 1
            summary.wrong.append((outcome.job_id, "no answer", outcome.error, None))
            continue
        answer = JobAnalysisResult.model_validate(outcome.answer)
        fields = score(expected, answer)
        for name, right in fields.items():
            if right:
                summary.by_field[name] += 1
            else:
                summary.wrong.append((outcome.job_id, name, compared(answer, name), compared(expected, name)))
        if all(fields.values()):
            summary.right += 1
    return summary


# --- Saved answers ------------------------------------------------------------------------------


def save_outcomes(path: Path, effort: str, prompt_version: str, model: str, outcomes: list[Outcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "effort": effort,
        "prompt_version": prompt_version,
        "model": model,
        "outcomes": [outcome.__dict__ for outcome in outcomes],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_outcomes(path: Path) -> tuple[dict[str, str], list[Outcome]]:
    """The run's details (effort, prompt version, model) and its outcomes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    details = {name: data[name] for name in ("effort", "prompt_version", "model")}
    return details, [Outcome(**item) for item in data["outcomes"]]
