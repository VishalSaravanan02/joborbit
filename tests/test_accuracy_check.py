"""Tests for scripts/accuracy_check.py. Nothing here contacts a real LLM.

The script's parts are loaded straight from the file. Commands that would call the LLM are run
with a fake provider; the end-to-end runs set LLM_PROVIDER=anthropic as well, so even a mistake
could never send a request with the real key in .env.
"""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime

import pytest
import yaml
from sqlalchemy.orm import Session, sessionmaker

from joborbit.config import load_app_settings
from joborbit.db.models import Base, Company, Job, User, UserProfile
from joborbit.db.session import create_sqlite_engine
from joborbit.llm.accuracy import EvalJob, Outcome, load_outcomes, read_jobs, save_outcomes
from joborbit.llm.client import LlmClient, LlmUnavailable, Provider, Reply
from joborbit.llm.prompts import PROMPT_VERSION, SCHEMA_NAME, instructions, job_message
from joborbit.llm.schemas import answer_schema
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "accuracy_check.py"
spec = importlib.util.spec_from_file_location("accuracy_check", SCRIPT)
accuracy_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy_check)

LABELS = {
    "countries": ["GB"],
    "seniority": "graduate",
    "experience_mandatory": False,
    "experience_years": None,
    "required_languages": [],
    "role_families": ["data_analyst"],
}
ANSWER = {
    "countries": ["GB"], "cities": ["London"], "work_mode": None, "seniority": "graduate",
    "is_graduate_scheme": False, "experience_years": None, "experience_mandatory": False,
    "required_languages": [], "role_families": ["data_analyst"], "matched_custom_roles": [], "skills": [],
    "min_degree": None, "deadline": None, "summary": "A graduate data analyst job.",
}


def run_script(*arguments: str, database=None) -> subprocess.CompletedProcess:
    env = {**os.environ, "LLM_PROVIDER": "anthropic"}  # no request can ever be sent
    if database is not None:
        env["DATABASE_PATH"] = str(database)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], env=env, capture_output=True, text=True, check=False
    )


# --- pick ---------------------------------------------------------------------------------------


@pytest.fixture
def db_file(tmp_path):
    """A throwaway database: one user with three data roles, and jobs at four companies."""
    path = tmp_path / "test.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for number in range(1, 5):
            session.add(Company(name=f"Company {number}", slug=f"company-{number}", size_category="startup"))
        user = User(telegram_id=1, display_name="Vishal")
        user.profile = UserProfile(roles=["data_scientist", "data_analyst", "data_engineer"], countries=["GB"])
        session.add(user)
        seen = datetime(2026, 10, 1, 9, 0)
        jobs = [
            # Company 1 has five passing jobs: at most three may be picked.
            *[(1, f"Data Analyst {n}", "London") for n in range(5)],
            (2, "Graduate Data Scientist", "Remote"),
            (2, "Senior Data Engineer", "London"),  # too senior: never picked
            (3, "Data Engineer", "Berlin, Germany"),  # wrong country: never picked
            (4, "Junior Data Scientist", "Manchester"),
        ]
        for number, (company_id, title, location) in enumerate(jobs):
            session.add(
                Job(
                    company_id=company_id, ats_type="greenhouse", external_id=str(number),
                    url=f"https://e.com/{number}", title=title, location_raw=location,
                    description_text=f"About {title}.", first_seen_at=seen,
                    is_baseline=True, prefilter_status="skipped",
                )
            )
        session.add(
            Job(
                company_id=4, ats_type="greenhouse", external_id="closed", url="https://e.com/c",
                title="Closed Data Analyst", location_raw="London", first_seen_at=seen, closed_at=seen,
            )
        )
        session.commit()
    engine.dispose()
    return path


def test_pick_chooses_passing_open_jobs_with_at_most_three_per_company(db_file, tmp_path):
    folder = tmp_path / "accuracy"
    result = run_script("--folder", str(folder), "pick", database=db_file)
    assert result.returncode == 0, result.stderr
    assert "Picked 5 of the 7 open jobs that pass the pre-filter, from 3 companies." in result.stdout
    jobs = read_jobs(folder / "jobs.yaml")
    titles = [job.title for job in jobs]
    assert sum(title.startswith("Data Analyst") for title in titles) == 3
    assert {"Graduate Data Scientist", "Junior Data Scientist"} <= set(titles)
    assert not {"Senior Data Engineer", "Data Engineer", "Closed Data Analyst"} & set(titles)
    assert all(job.missing_labels() for job in jobs)


def test_pick_saves_what_the_llm_will_be_sent(db_file, tmp_path):
    folder = tmp_path / "accuracy"
    run_script("--folder", str(folder), "pick", database=db_file)
    job = next(job for job in read_jobs(folder / "jobs.yaml") if job.title == "Junior Data Scientist")
    assert (job.id, job.company, job.location) == ("company-4/8", "Company 4", "Manchester")
    assert job.posted == date(2026, 10, 1)
    assert (job.url, job.description) == ("https://e.com/8", "About Junior Data Scientist.")


def test_pick_never_replaces_a_jobs_file(db_file, tmp_path):
    folder = tmp_path / "accuracy"
    folder.mkdir()
    (folder / "jobs.yaml").write_text("my labels", encoding="utf-8")
    result = run_script("--folder", str(folder), "pick", database=db_file)
    assert result.returncode == 1 and "may hold your labels; nothing was changed" in result.stderr
    assert (folder / "jobs.yaml").read_text(encoding="utf-8") == "my labels"


def rows(company_and_job_ids: list[tuple[int, int]]) -> list[tuple[Company, Job]]:
    companies = {}
    result = []
    for company_id, job_id in company_and_job_ids:
        company = companies.setdefault(company_id, Company(id=company_id, name=f"C{company_id}", slug=f"c{company_id}"))
        result.append((company, Job(id=job_id, title=f"Job {job_id:03}")))
    return result


def test_choosing_is_the_same_every_time_and_keeps_to_the_limits():
    candidates = rows([(company, job) for company in range(1, 11) for job in range(company * 100, company * 100 + 8)])
    first = accuracy_check.choose(candidates, 30, 3)
    assert [job.id for _, job in first] == [job.id for _, job in accuracy_check.choose(list(candidates), 30, 3)]
    assert len(first) == 30
    per_company = {company.id: sum(1 for c, _ in first if c.id == company.id) for company, _ in first}
    assert max(per_company.values()) <= 3


def test_fewer_candidates_than_wanted_gives_them_all():
    assert len(accuracy_check.choose(rows([(1, 1), (2, 2)]), 30, 3)) == 2


# --- run --------------------------------------------------------------------------------------------


class FakeProvider(Provider):
    def __init__(self, *texts: str | Exception) -> None:
        self.texts = list(texts)
        self.requests: list[tuple] = []

    def send(self, instructions, messages, schema_name, schema):
        self.requests.append((instructions, list(messages), schema_name, schema))
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return Reply("done", text, "gpt-6-luna-2026-05-18", 2500, 2000, 300)


def eval_job(job_id: str, title: str = "Graduate Data Analyst") -> EvalJob:
    return EvalJob(
        id=job_id, company="Acme", title=title, location="London", posted=date(2026, 10, 1),
        url="https://e.com", description="Join us.", expected=LABELS,
    )


@pytest.fixture
def sessions(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def client_with(sessions, *texts) -> LlmClient:
    return LlmClient(FakeProvider(*texts), load_app_settings().llm, sessions, now=lambda: datetime(2026, 10, 8))


def test_each_job_is_asked_about_exactly_as_analysis_will(sessions):
    client = client_with(sessions, json.dumps(ANSWER))
    outcomes, model = accuracy_check.ask_about(client, [eval_job("acme/1")])
    [(sent_instructions, messages, schema_name, schema)] = client.provider.requests
    assert sent_instructions == instructions() and (schema_name, schema) == (SCHEMA_NAME, answer_schema())
    assert messages[0].content == job_message("Acme", "Graduate Data Analyst", "London", date(2026, 10, 1), "Join us.")
    assert model == "gpt-6-luna-2026-05-18"
    [outcome] = outcomes
    assert (outcome.job_id, outcome.answer["seniority"], outcome.error) == ("acme/1", "graduate", None)
    assert (outcome.input_tokens, outcome.output_tokens) == (2500, 300) and outcome.cost_usd > 0


def test_a_job_without_a_usable_answer_is_recorded_and_the_rest_go_on(sessions, capsys):
    bad = json.dumps({**ANSWER, "countries": ["US"]})
    client = client_with(sessions, bad, bad, json.dumps(ANSWER))
    outcomes, _ = accuracy_check.ask_about(client, [eval_job("acme/1"), eval_job("acme/2")], label="low")
    assert outcomes[0].answer is None and "broke the rules twice" in outcomes[0].error
    assert "acme/1 (no usable answer)" in capsys.readouterr().err
    assert outcomes[1].answer is not None


def test_the_check_stops_when_the_llm_is_unavailable(sessions):
    client = client_with(sessions, LlmUnavailable("Could not reach OpenAI"))
    with pytest.raises(LlmUnavailable):
        accuracy_check.ask_about(client, [eval_job("acme/1"), eval_job("acme/2")])


def write_labelled(folder, jobs: list[EvalJob]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data = [job.model_dump(mode="json") for job in jobs]
    (folder / "jobs.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_run_asks_once_per_effort_saves_the_answers_and_reports(tmp_path, sessions, monkeypatch, capsys):
    folder = tmp_path / "accuracy"
    write_labelled(folder, [eval_job("acme/1"), eval_job("acme/2")])
    asked = []

    def fake_make_client(reasoning_effort=None):
        asked.append(reasoning_effort)
        return client_with(sessions, json.dumps(ANSWER), json.dumps(ANSWER))

    monkeypatch.setattr(accuracy_check, "make_client", fake_make_client)
    accuracy_check.run(folder, ["low", "none", "low"])
    assert asked == ["low", "none"]  # each effort once
    for effort in ("low", "none"):
        details, outcomes = load_outcomes(folder / f"answers-{effort}.json")
        assert details == {"effort": effort, "prompt_version": PROMPT_VERSION, "model": "gpt-6-luna-2026-05-18"}
        assert [outcome.job_id for outcome in outcomes] == ["acme/1", "acme/2"]
    captured = capsys.readouterr()
    out = captured.out
    assert out.startswith("Asking about each job at effort low, none: at least 4 calls.")
    progress = captured.err.splitlines()  # one line per job, kept apart from the report
    assert len(progress) == 4 and progress[0].startswith("  low      1/2") and progress[0].endswith("acme/1")
    assert progress[3].startswith("  none     2/2") and "acme/" not in out
    assert "2 of 2 jobs right on all five" in out
    assert out.rstrip().endswith("Cheapest effort reaching the target: none")


def test_run_refuses_until_every_job_is_labelled(tmp_path):
    folder = tmp_path / "accuracy"
    unfinished = eval_job("acme/2").model_copy(update={"expected": {**LABELS, "seniority": "TODO"}})
    write_labelled(folder, [eval_job("acme/1"), unfinished])
    result = run_script("--folder", str(folder), "run")
    assert result.returncode == 1
    assert "1 of 2 jobs need their labels fixed first:\n  acme/2: not labelled yet: seniority" in result.stderr


def test_run_says_how_to_start_when_there_are_no_jobs(tmp_path):
    result = run_script("--folder", str(tmp_path / "nothing-here"), "run")
    assert result.returncode == 1 and "accuracy_check.py pick" in result.stderr


def test_run_stops_cleanly_when_the_llm_cannot_be_used(tmp_path):
    """LLM_PROVIDER=anthropic makes make_client refuse: the check stops before any request."""
    folder = tmp_path / "accuracy"
    write_labelled(folder, [eval_job("acme/1")])
    result = run_script("--folder", str(folder), "run", "--effort", "none")
    assert result.returncode == 1
    assert "Stopped during effort 'none' (nothing saved for it): " in result.stderr
    assert "LLM_PROVIDER=anthropic is not built yet" in result.stderr
    assert not (folder / "answers-none.json").exists()


# --- score and the report ---------------------------------------------------------------------------


def saved(folder, effort: str, answers: list[dict | None]) -> None:
    outcomes = [
        Outcome(f"acme/{n}", answer, None if answer else "refused", 3000, 400, 0.0005, 4.0)
        for n, answer in enumerate(answers, start=1)
    ]
    save_outcomes(folder / f"answers-{effort}.json", effort, "1", "gpt-6-luna-2026-05-18", outcomes)


def test_score_reports_every_saved_effort_and_the_cheapest_good_enough(tmp_path):
    folder = tmp_path / "accuracy"
    jobs = [eval_job(f"acme/{n}", f"Graduate Data Analyst {n}") for n in range(1, 11)]
    write_labelled(folder, jobs)
    wrong = {**ANSWER, "seniority": "mid"}
    saved(folder, "none", [ANSWER] * 8 + [wrong, None])  # 8 of 10: under the target of 9
    saved(folder, "low", [ANSWER] * 9 + [wrong])  # 9 of 10: reaches it
    saved(folder, "medium", [ANSWER] * 10)
    result = run_script("--folder", str(folder), "score")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Effort 'none' (prompt version 1, model gpt-6-luna-2026-05-18): 8 of 10 jobs right" in out
    assert "8 of 10 jobs right on all five (target: 9)" in out
    assert "  seniority     8 of 10" in out and "  no answer     1" in out
    assert "per job: 3,000 input tokens, 400 output tokens, about $0.00050, 4.0 s" in out
    assert "acme/9  Graduate Data Analyst 9\n      seniority: 'mid'  /  label: ['graduate']" in out
    assert "acme/10  Graduate Data Analyst 10\n      no answer: 'refused'" in out
    assert "  low       9 of 10 right   about $0.00050 per job" in out
    assert out.rstrip().endswith("Cheapest effort reaching the target: low")


def test_score_says_when_no_effort_is_good_enough(tmp_path):
    folder = tmp_path / "accuracy"
    write_labelled(folder, [eval_job("acme/1"), eval_job("acme/2")])
    saved(folder, "low", [ANSWER, {**ANSWER, "countries": []}])
    result = run_script("--folder", str(folder), "score")
    assert "No effort reached the target yet" in result.stdout


def test_score_needs_saved_answers(tmp_path):
    folder = tmp_path / "accuracy"
    write_labelled(folder, [eval_job("acme/1")])
    result = run_script("--folder", str(folder), "score")
    assert result.returncode == 1 and "No saved answers" in result.stderr


@pytest.mark.parametrize(("jobs", "needed"), [(30, 27), (10, 9), (25, 23), (1, 1)])
def test_the_target_is_ninety_percent_rounded_up(jobs, needed):
    assert accuracy_check.target(jobs) == needed
