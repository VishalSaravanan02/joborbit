"""Tests for the LLM client (joborbit/llm/client.py). Nothing here contacts a real LLM.

The client's own rules (budget, usage records, retry, failures) are tested with FakeProvider,
which returns prepared replies. OpenAIProvider is tested against a fake OpenAI built with
httpx2.MockTransport: the openai library uses httpx2, which respx can't intercept.
"""

import json
import logging
from datetime import date, datetime
from typing import Literal

import httpx2
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from joborbit.config import LlmConfig, LlmPrices
from joborbit.db.models import Base, LlmUsage
from joborbit.db.session import create_sqlite_engine
from joborbit.llm import client as client_module
from joborbit.llm.client import (
    RETRY_REQUEST,
    AnswerFailed,
    BudgetReached,
    LlmClient,
    LlmUnavailable,
    Message,
    OpenAIProvider,
    Provider,
    Reply,
    estimate_cost_usd,
    make_client,
)
from joborbit.llm.schemas import AnswerError
from joborbit.settings import Settings

# Round numbers, so costs are easy to check by hand.
CONFIG = LlmConfig(
    model="test-model",
    reasoning_effort="low",
    max_output_tokens=4000,
    timeout_seconds=60,
    network_retries=3,
    daily_call_cap=10,
    warn_at_fraction=0.8,
    monthly_budget_usd=1.0,
    prices_usd_per_million=LlmPrices(input=2.0, cached_input=0.5, output=8.0),
)
NOW = datetime(2026, 10, 8, 21, 0)
TODAY = NOW.date()
# 80 uncached input tokens at $2, 20 cached at $0.50 and 50 output at $8, per million.
REPLY_COST = (80 * 2.0 + 20 * 0.5 + 50 * 8.0) / 1_000_000


class Colour(BaseModel):
    """A tiny answer, so these tests are about the client rather than the job schema."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["red", "blue"]


def check_colour(text: str) -> Colour:
    try:
        return Colour.model_validate_json(text)
    except ValidationError as error:
        raise AnswerError(f"name: must be red or blue ({error.error_count()} problem)") from error


def reply(text: str = '{"name": "red"}', stop: str = "done", detail: str = "") -> Reply:
    return Reply(stop, text, "test-model-2026-05-18", 100, 20, 50, detail)


class FakeProvider(Provider):
    """Returns the prepared replies in order (or raises a prepared error), and keeps every request."""

    def __init__(self, *replies: Reply | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[tuple] = []

    def send(self, instructions, messages, schema_name, schema):
        self.requests.append((instructions, list(messages), schema_name, schema))
        prepared = self.replies.pop(0)
        if isinstance(prepared, Exception):
            raise prepared
        return prepared


@pytest.fixture
def sessions(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def make(sessions, *replies, now: datetime = NOW, config: LlmConfig = CONFIG) -> tuple[LlmClient, FakeProvider]:
    provider = FakeProvider(*replies)
    return LlmClient(provider, config, sessions, now=lambda: now), provider


def ask(client: LlmClient):
    return client.ask(
        instructions="Name a colour.", text="The sky", schema_name="colour", schema={"type": "object"},
        check=check_colour,
    )


def usage(sessions, day: date = TODAY) -> tuple | None:
    with sessions() as session:
        row = session.get(LlmUsage, day)
        return (row.calls, row.input_tokens, row.output_tokens, row.est_cost_usd) if row else None


def add_usage(sessions, day: date, calls: int = 0, cost: float = 0.0) -> None:
    with sessions() as session, session.begin():
        session.add(LlmUsage(day=day, calls=calls, est_cost_usd=cost))


# --- A good answer ---------------------------------------------------------------------------------


def test_a_good_answer_is_returned_with_what_it_cost(sessions):
    client, _ = make(sessions, reply())
    answer = ask(client)
    assert answer.result == Colour(name="red")
    assert (answer.model, answer.calls, answer.input_tokens, answer.output_tokens) == (
        "test-model-2026-05-18", 1, 100, 50,
    )
    assert answer.cost_usd == pytest.approx(REPLY_COST)


def test_the_request_carries_the_instructions_text_and_schema(sessions):
    client, provider = make(sessions, reply())
    ask(client)
    assert provider.requests == [("Name a colour.", [Message("user", "The sky")], "colour", {"type": "object"})]


def test_the_cost_estimate_prices_cached_input_separately():
    assert estimate_cost_usd(reply(), CONFIG.prices_usd_per_million) == pytest.approx(REPLY_COST)


def test_more_cached_tokens_than_input_tokens_never_gives_a_negative_cost():
    odd = Reply("done", "", "m", input_tokens=10, cached_input_tokens=20, output_tokens=0)
    assert estimate_cost_usd(odd, CONFIG.prices_usd_per_million) == pytest.approx(20 * 0.5 / 1_000_000)


# --- Usage records ---------------------------------------------------------------------------------


def test_every_reply_is_recorded_for_its_day(sessions):
    client, _ = make(sessions, reply(), reply())
    ask(client)
    ask(client)
    calls, input_tokens, output_tokens, cost = usage(sessions)
    assert (calls, input_tokens, output_tokens) == (2, 200, 100)
    assert cost == pytest.approx(2 * REPLY_COST)


def test_a_new_day_gets_its_own_row(sessions):
    ask(make(sessions, reply())[0])
    tomorrow = datetime(2026, 10, 9, 0, 1)
    ask(make(sessions, reply(), now=tomorrow)[0])
    assert usage(sessions)[0] == 1 and usage(sessions, tomorrow.date())[0] == 1


def test_the_day_is_the_utc_day_of_the_call(sessions):
    """Just before midnight UTC still counts as that day."""
    ask(make(sessions, reply(), now=datetime(2026, 10, 8, 23, 59))[0])
    assert usage(sessions, date(2026, 10, 8))[0] == 1


def test_a_reply_that_could_not_be_sent_is_not_recorded(sessions):
    """Only replies are paid for: a request that never reached the LLM costs nothing."""
    client, _ = make(sessions, LlmUnavailable("Could not reach OpenAI"))
    with pytest.raises(LlmUnavailable):
        ask(client)
    assert usage(sessions) is None


# --- An answer that breaks the rules ----------------------------------------------------------------


def test_a_bad_answer_is_retried_once_with_the_problems_listed(sessions):
    client, provider = make(sessions, reply('{"name": "green"}'), reply())
    answer = ask(client)
    assert answer.result == Colour(name="red")
    assert (answer.calls, answer.input_tokens, answer.output_tokens) == (2, 200, 100)
    assert answer.cost_usd == pytest.approx(2 * REPLY_COST)
    _, messages, _, _ = provider.requests[1]
    assert messages == [
        Message("user", "The sky"),
        Message("assistant", '{"name": "green"}'),
        Message("user", RETRY_REQUEST.format(problems="name: must be red or blue (1 problem)")),
    ]
    assert usage(sessions)[0] == 2


def test_two_bad_answers_fail_the_request_and_both_are_recorded(sessions):
    client, provider = make(sessions, reply('{"name": "green"}'), reply("not json"))
    with pytest.raises(AnswerFailed, match="broke the rules twice:\nname: must be red or blue"):
        ask(client)
    assert len(provider.requests) == 2  # never a third try
    assert usage(sessions)[0] == 2


def test_an_empty_answer_counts_as_breaking_the_rules(sessions):
    client, _ = make(sessions, reply(""), reply())
    assert ask(client).calls == 2


# --- Replies that end badly -----------------------------------------------------------------------------


def test_a_cut_off_answer_fails_without_a_retry(sessions):
    client, provider = make(sessions, reply('{"na', stop="cut_off"))
    with pytest.raises(AnswerFailed, match=r"cut off at max_output_tokens \(4000\)"):
        ask(client)
    assert len(provider.requests) == 1
    assert usage(sessions)[0] == 1  # paid for, so recorded


def test_a_refusal_fails_without_a_retry(sessions):
    client, provider = make(sessions, reply("", stop="refused", detail="I can't help with that."))
    with pytest.raises(AnswerFailed, match="refused to answer: I can't help with that."):
        ask(client)
    assert len(provider.requests) == 1
    assert usage(sessions)[0] == 1


def test_a_failed_reply_stops_the_run_and_is_recorded(sessions):
    client, _ = make(sessions, reply("", stop="failed", detail="server overloaded"))
    with pytest.raises(LlmUnavailable, match="could not finish its reply: server overloaded"):
        ask(client)
    assert usage(sessions)[0] == 1


def test_failures_that_stop_the_run_are_not_answer_failures():
    """The caller tells them apart: stop the run, or mark one job as failed."""
    assert issubclass(BudgetReached, LlmUnavailable)
    assert not issubclass(LlmUnavailable, AnswerFailed) and not issubclass(AnswerFailed, LlmUnavailable)


# --- The budget --------------------------------------------------------------------------------------


def test_the_last_call_under_the_daily_cap_is_allowed(sessions):
    add_usage(sessions, TODAY, calls=CONFIG.daily_call_cap - 1)
    ask(make(sessions, reply())[0])
    assert usage(sessions)[0] == CONFIG.daily_call_cap


def test_nothing_is_sent_once_the_daily_cap_is_reached(sessions):
    add_usage(sessions, TODAY, calls=CONFIG.daily_call_cap)
    client, provider = make(sessions, reply())
    with pytest.raises(BudgetReached, match="Daily cap reached: 10 of 10"):
        ask(client)
    assert provider.requests == []


def test_yesterdays_calls_do_not_count_towards_todays_cap(sessions):
    add_usage(sessions, date(2026, 10, 7), calls=CONFIG.daily_call_cap)
    assert ask(make(sessions, reply())[0]).calls == 1


def test_the_retry_also_respects_the_cap(sessions):
    add_usage(sessions, TODAY, calls=CONFIG.daily_call_cap - 1)
    client, provider = make(sessions, reply('{"name": "green"}'), reply())
    with pytest.raises(BudgetReached):
        ask(client)
    assert len(provider.requests) == 1


def test_nothing_is_sent_once_the_monthly_budget_is_reached(sessions):
    add_usage(sessions, date(2026, 10, 1), cost=0.6)
    add_usage(sessions, date(2026, 10, 7), cost=0.4)
    client, provider = make(sessions, reply())
    with pytest.raises(BudgetReached, match=r"Monthly budget reached: an estimated \$1.00 of \$1.00"):
        ask(client)
    assert provider.requests == []


def test_just_under_the_monthly_budget_is_allowed(sessions):
    add_usage(sessions, date(2026, 10, 1), cost=0.99)
    assert ask(make(sessions, reply())[0]).calls == 1


def test_last_months_spending_does_not_count(sessions):
    add_usage(sessions, date(2026, 9, 30), cost=5.0)
    assert ask(make(sessions, reply())[0]).calls == 1


# --- The warning ----------------------------------------------------------------------------------------


def test_the_warning_comes_once_on_the_call_that_passes_the_line(sessions, caplog):
    """With a cap of 10 and a line at 80%, only the 8th call warns."""
    client, _ = make(sessions, *[reply() for _ in range(10)])
    warned_at = []
    for call in range(1, 11):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="joborbit.llm.client"):
            ask(client)
        if caplog.records:
            warned_at.append(call)
    assert warned_at == [8]
    assert usage(sessions)[0] == 10


def test_the_warning_says_how_much_of_the_cap_is_used(sessions, caplog):
    add_usage(sessions, TODAY, calls=7)
    with caplog.at_level(logging.WARNING, logger="joborbit.llm.client"):
        ask(make(sessions, reply())[0])
    assert caplog.messages == ["LLM calls today: 8 of the daily cap of 10 (80%)"]


def test_logs_never_include_the_text_sent_or_received(sessions, caplog):
    client, _ = make(sessions, reply('{"name": "green"}'), reply())
    with caplog.at_level(logging.DEBUG):
        client.ask(
            instructions="SECRET INSTRUCTIONS", text="SECRET POSTING", schema_name="colour", schema={},
            check=check_colour,
        )
    logged = "\n".join(caplog.messages)
    assert "SECRET" not in logged and "green" not in logged and "red" not in logged


# --- The OpenAI provider, against a fake OpenAI --------------------------------------------------------


def openai_response(status: str = "completed", content: list | None = None, **extra) -> dict:
    """A reply in the shape OpenAI's Responses API uses."""
    content = [{"type": "output_text", "text": '{"name": "red"}', "annotations": []}] if content is None else content
    return {
        "id": "resp_1", "object": "response", "created_at": 1, "status": status,
        "model": "gpt-6-luna-2026-05-18", "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant", "content": content}],
        "usage": {
            "input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1000},
            "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 250}, "total_tokens": 1500,
        },
        **extra,
    }


class FakeOpenAI:
    """Answers every request with the prepared HTTP replies, in order, and keeps each request body."""

    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        prepared = self.responses.pop(0)
        if isinstance(prepared, Exception):
            raise prepared
        return prepared

    def provider(self, config: LlmConfig = CONFIG) -> OpenAIProvider:
        client = httpx2.Client(transport=httpx2.MockTransport(self.handle))
        return OpenAIProvider("sk-test-key-123", config, http_client=client)


def send(provider: OpenAIProvider) -> Reply:
    messages = [Message("user", "The sky"), Message("assistant", "{}"), Message("user", "Again")]
    return provider.send("Name a colour.", messages, "colour", {"type": "object"})


def test_the_request_to_openai_is_exactly_as_designed():
    fake = FakeOpenAI(httpx2.Response(200, json=openai_response()))
    send(fake.provider())
    assert fake.bodies == [
        {
            "model": "test-model",
            "instructions": "Name a colour.",
            "input": [
                {"role": "user", "content": "The sky"},
                {"role": "assistant", "content": "{}"},
                {"role": "user", "content": "Again"},
            ],
            "text": {"format": {"type": "json_schema", "name": "colour", "schema": {"type": "object"}, "strict": True}},
            "reasoning": {"effort": "low"},
            "max_output_tokens": 4000,
            "store": False,
        }
    ]


def test_a_completed_reply_gives_the_answer_and_its_tokens():
    fake = FakeOpenAI(httpx2.Response(200, json=openai_response()))
    assert send(fake.provider()) == Reply("done", '{"name": "red"}', "gpt-6-luna-2026-05-18", 1200, 1000, 300)


def test_a_refusal_from_openai_is_reported():
    refusal = [{"type": "refusal", "refusal": "I can't help with that."}]
    fake = FakeOpenAI(httpx2.Response(200, json=openai_response(content=refusal)))
    result = send(fake.provider())
    assert (result.stop, result.text, result.detail) == ("refused", "", "I can't help with that.")


@pytest.mark.parametrize(
    ("reason", "stop"),
    [("max_output_tokens", "cut_off"), ("content_filter", "refused")],
)
def test_an_incomplete_reply_says_why(reason, stop):
    response = openai_response(status="incomplete", incomplete_details={"reason": reason})
    result = send(FakeOpenAI(httpx2.Response(200, json=response)).provider())
    assert result.stop == stop and reason in result.detail
    assert result.output_tokens == 300  # still paid for


def test_a_failed_reply_is_reported_with_openais_reason():
    response = openai_response(status="failed", error={"code": "server_error", "message": "Something broke"})
    result = send(FakeOpenAI(httpx2.Response(200, json=response)).provider())
    assert (result.stop, result.detail) == ("failed", "Something broke")


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_explained_without_repeating_any_of_it(status):
    body = {"error": {"message": "Incorrect API key provided: sk-tes***123", "type": "invalid_request_error"}}
    fake = FakeOpenAI(httpx2.Response(status, json=body))
    with pytest.raises(LlmUnavailable, match=f"rejected the API key \\(HTTP {status}\\)") as error:
        send(fake.provider())
    assert "sk-" not in str(error.value)


def test_other_refused_requests_stop_the_run_with_openais_reason():
    body = {"error": {"message": "Invalid schema for response_format", "type": "invalid_request_error"}}
    fake = FakeOpenAI(httpx2.Response(400, json=body))
    with pytest.raises(LlmUnavailable, match=r"refused the request \(HTTP 400\): Invalid schema"):
        send(fake.provider())


def test_a_long_reason_that_is_not_openais_is_shortened():
    """A gateway between us and OpenAI may answer with a whole web page."""
    page = "<html>" + "Bad gateway " * 100 + "</html>"
    fake = FakeOpenAI(httpx2.Response(502, text=page))
    with pytest.raises(LlmUnavailable) as error:
        send(fake.provider(CONFIG.model_copy(update={"network_retries": 0})))
    assert str(error.value) == f"OpenAI refused the request (HTTP 502): {page[:client_module.MAX_REASON_CHARS]}"


def server_error() -> httpx2.Response:
    # retry-after-ms tells the openai library to wait 1 ms before retrying, so the test is quick.
    return httpx2.Response(500, json={"error": {"message": "boom"}}, headers={"retry-after-ms": "1"})


def test_server_errors_are_retried_by_the_library_then_succeed():
    fake = FakeOpenAI(server_error(), server_error(), httpx2.Response(200, json=openai_response()))
    assert send(fake.provider()).stop == "done"
    assert len(fake.bodies) == 3


def test_after_the_retries_a_server_error_stops_the_run():
    retries = CONFIG.model_copy(update={"network_retries": 2})
    fake = FakeOpenAI(server_error(), server_error(), server_error())
    with pytest.raises(LlmUnavailable, match=r"HTTP 500"):
        send(fake.provider(retries))
    assert len(fake.bodies) == 3  # the first try and 2 retries


def test_connection_problems_stop_the_run():
    no_retries = CONFIG.model_copy(update={"network_retries": 0})
    fake = FakeOpenAI(httpx2.ConnectError("no route to host"))
    with pytest.raises(LlmUnavailable, match="Could not reach OpenAI"):
        send(fake.provider(no_retries))


def test_the_timeout_and_retries_come_from_the_settings():
    library_client = FakeOpenAI().provider()._client
    assert (library_client.timeout, library_client.max_retries) == (60, 3)


def test_the_client_and_openai_work_together(sessions):
    """End to end: the answer, the model's dated name and the cost of the cached and uncached input."""
    fake = FakeOpenAI(httpx2.Response(200, json=openai_response()))
    answer = ask(LlmClient(fake.provider(), CONFIG, sessions, now=lambda: NOW))
    assert answer.result == Colour(name="red") and answer.model == "gpt-6-luna-2026-05-18"
    assert answer.cost_usd == pytest.approx((200 * 2.0 + 1000 * 0.5 + 300 * 8.0) / 1_000_000)
    with sessions() as session:
        assert session.scalars(select(LlmUsage.calls)).one() == 1


# --- Setting the client up ----------------------------------------------------------------------------------


def settings_without_env_file(**values) -> Settings:
    """Settings that ignore .env, so these tests behave the same on any computer."""
    return Settings(_env_file=None, **values)


def test_make_client_uses_openai_and_the_settings_file(monkeypatch, sessions):
    settings = settings_without_env_file(llm_provider="openai", openai_api_key="sk-test-key-123")
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    client = make_client(sessions)
    assert isinstance(client.provider, OpenAIProvider)
    assert client.config.model == "gpt-6-luna"  # from config/settings.yaml
    assert client.provider._client.api_key == "sk-test-key-123"


def test_make_client_needs_the_api_key(monkeypatch):
    settings = settings_without_env_file(llm_provider="openai", openai_api_key=None)
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    with pytest.raises(LlmUnavailable, match="OPENAI_API_KEY is not set in .env"):
        make_client()


def test_anthropic_is_not_built_yet(monkeypatch):
    settings = settings_without_env_file(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    with pytest.raises(LlmUnavailable, match="LLM_PROVIDER=anthropic is not built yet"):
        make_client()
