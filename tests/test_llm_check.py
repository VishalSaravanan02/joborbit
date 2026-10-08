"""Tests for scripts/llm_check.py. Nothing here contacts a real LLM.

The script's parts are loaded straight from the file and run with a fake provider. The one
end-to-end test runs the real script with LLM_PROVIDER=anthropic, which stops before any request
is made, so it is safe even on a computer whose .env holds a real key.
"""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime

import pytest
from sqlalchemy.orm import sessionmaker

from joborbit.config import load_app_settings
from joborbit.db.models import Base, LlmUsage
from joborbit.db.session import create_sqlite_engine
from joborbit.llm.client import AnswerFailed, LlmClient, Provider, Reply
from joborbit.llm.schemas import answer_schema
from joborbit.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "llm_check.py"
spec = importlib.util.spec_from_file_location("llm_check", SCRIPT)
llm_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_check)

SENSIBLE = {
    "countries": ["GB"],
    "cities": ["London"],
    "work_mode": "hybrid",
    "seniority": "graduate",
    "is_graduate_scheme": True,
    "experience_years": None,
    "experience_mandatory": False,
    "required_languages": ["es"],
    "role_families": ["data_analyst", "tech_graduate_scheme"],
    "matched_custom_roles": [],
    "skills": ["SQL", "Python"],
    "min_degree": "bachelor",
    "deadline": "2026-11-30",
    "summary": "Two-year data analyst graduate programme in London; fluent Spanish required.",
}


class FakeProvider(Provider):
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[tuple] = []

    def send(self, instructions, messages, schema_name, schema):
        self.requests.append((instructions, list(messages), schema_name, schema))
        return Reply("done", self.texts.pop(0), "test-model-2026-05-18", 1500, 1000, 400)


@pytest.fixture
def client(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    provider = FakeProvider(json.dumps(SENSIBLE))
    return LlmClient(provider, load_app_settings().llm, sessions, now=lambda: datetime(2026, 10, 8, 21, 0))


def test_the_sample_posting_is_sent_with_the_real_answer_schema(client):
    answer = llm_check.check_once(client)
    [(instructions, messages, schema_name, schema)] = client.provider.requests
    assert instructions == llm_check.INSTRUCTIONS
    assert [message.content for message in messages] == [llm_check.SAMPLE_POSTING]
    assert (schema_name, schema) == ("job_analysis", answer_schema())
    assert answer.result.deadline == date(2026, 11, 30)


def test_the_report_shows_the_answer_and_what_it_cost(client):
    answer = llm_check.check_once(client)
    text = llm_check.report(answer, 2.34, LlmUsage(day=date(2026, 10, 8), calls=3, est_cost_usd=0.0012))
    assert text.startswith("The LLM answered in 2.3 s, and the answer passed every check.")
    assert "model:   test-model-2026-05-18" in text
    assert "calls:   1\n" in text
    assert "tokens:  1,500 input, 400 output" in text
    assert "role_families:         data_analyst, tech_graduate_scheme" in text
    assert "experience_years:      (not stated)" in text
    assert "matched_custom_roles:  (none)" in text
    assert "deadline:              2026-11-30" in text
    assert text.endswith("LLM use today (UTC): 3 calls, about $0.0012.")


def test_the_report_says_when_the_answer_needed_a_second_try(client):
    client.provider.texts = ['{"countries": ["US"]}', json.dumps(SENSIBLE)]
    answer = llm_check.check_once(client)
    assert "calls:   2 (the first answer broke the rules" in llm_check.report(answer, 1.0, None)


def test_no_custom_roles_are_offered_so_none_may_be_matched(client):
    invented = json.dumps({**SENSIBLE, "matched_custom_roles": ["Insights Analyst"]})
    client.provider.texts = [invented, invented]
    with pytest.raises(AnswerFailed, match="not a custom role we asked about"):
        llm_check.check_once(client)


def test_a_problem_is_explained_without_contacting_any_llm(tmp_path):
    """LLM_PROVIDER=anthropic stops before any request; real variables beat the values in .env."""
    env = {**os.environ, "LLM_PROVIDER": "anthropic", "DATABASE_PATH": str(tmp_path / "test.db")}
    result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "The check could not run: LLM_PROVIDER=anthropic is not built yet" in result.stderr


@pytest.mark.parametrize(("calls", "words"), [(1, "1 call,"), (2, "2 calls,")])
def test_the_number_of_calls_today_reads_correctly(client, calls, words):
    answer = llm_check.check_once(client)
    text = llm_check.report(answer, 1.0, LlmUsage(day=date(2026, 10, 8), calls=calls, est_cost_usd=0.0))
    assert text.endswith(f"LLM use today (UTC): {words} about $0.0000.")
