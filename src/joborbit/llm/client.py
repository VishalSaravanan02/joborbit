"""Talks to the LLM: the provider, the budget guard, the usage records and the retry, in one place.

LlmClient.ask() sends one request and returns the checked answer. On the way it:

- checks the budget first: the daily call cap and the monthly spend limit in settings.yaml.
  Once either is reached it raises BudgetReached, and nothing is sent until the next day or month (UTC).
- records each reply's tokens and estimated cost in llm_usage straight away, in its own small save,
  because the money is spent the moment the provider replies, whatever happens next.
- warns once a day, on the call that passes warn_at_fraction of the cap.
- checks the answer, and if it breaks the rules, asks once more with the problems listed.

Two kinds of failure:

- LlmUnavailable: nothing can be analysed right now (no key, the provider unreachable after its own
  retries, the budget used up). The caller stops, and the jobs wait for the next run.
- AnswerFailed: this one request got no usable answer (refused, cut off, or invalid twice). The
  caller marks that job as failed and carries on.

The client knows nothing about jobs: the caller gives it the instructions, the text, the JSON schema
and a function that checks the answer. Retries after timeouts, rate limits and server errors are
done by the provider's own library. Logs never include the text sent or received.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import httpx2
import openai
from openai.types.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session, sessionmaker

from joborbit.config import LlmConfig, LlmPrices, load_app_settings
from joborbit.db.models import LlmUsage
from joborbit.db.session import get_engine
from joborbit.llm.schemas import AnswerError
from joborbit.settings import get_settings
from joborbit.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Sent after an answer that broke the rules, together with that answer, for the one retry.
RETRY_REQUEST = (
    "Your answer broke these rules:\n{problems}\n\n"
    "Answer again with the whole JSON object, fixing only these problems."
)


# --- Errors ------------------------------------------------------------------------------------


class LlmError(Exception):
    """Something went wrong while asking the LLM."""


class LlmUnavailable(LlmError):
    """Nothing can be analysed right now. Stop, and let the next run try again."""


class BudgetReached(LlmUnavailable):
    """The daily call cap or the monthly budget is used up."""


class AnswerFailed(LlmError):
    """This one request got no usable answer. Mark it as failed and carry on with the next."""


# --- What goes to and comes back from a provider ---------------------------------------------


@dataclass(frozen=True)
class Message:
    """One turn of the conversation: our request ("user") or the LLM's earlier answer ("assistant")."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Reply:
    """What the provider sent back for one request.

    stop says how the reply ended: "done" (a full answer), "cut_off" (it hit max_output_tokens),
    "refused" (the model declined, or a safety filter stopped it) or "failed" (the provider's error).
    """

    stop: Literal["done", "cut_off", "refused", "failed"]
    text: str  # the answer as JSON text; "" if there was none
    model: str  # the exact model that answered, e.g. a dated version of the one asked for
    input_tokens: int  # all input, including the cached part
    cached_input_tokens: int
    output_tokens: int  # the answer plus the model's thinking
    detail: str = ""  # the refusal, or why the reply was cut off or failed


@dataclass(frozen=True)
class Answer:
    """A checked answer, and what it took to get it (every attempt together)."""

    result: BaseModel
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_cost_usd(reply: Reply, prices: LlmPrices) -> float:
    """The reply's cost in US dollars, from the prices in settings.yaml.

    An estimate: the provider's bill is what counts. It leaves out the small premium some
    providers charge the first time a piece of input is cached.
    """
    uncached = max(reply.input_tokens - reply.cached_input_tokens, 0)
    dollars_per_million = (
        uncached * prices.input + reply.cached_input_tokens * prices.cached_input + reply.output_tokens * prices.output
    )
    return dollars_per_million / 1_000_000


# --- Providers ----------------------------------------------------------------------------------


class Provider(ABC):
    """The template for one LLM provider. Adding Anthropic later means writing one more of these."""

    @abstractmethod
    def send(
        self, instructions: str, messages: Sequence[Message], schema_name: str, schema: dict[str, Any]
    ) -> Reply:
        """Send one request and return the reply.

        Raises LlmUnavailable if the provider can't be reached (after its own retries) or rejects the
        request itself, e.g. because the key is wrong.
        """


class OpenAIProvider(Provider):
    """OpenAI's models, through its Responses API with strict structured output."""

    def __init__(self, api_key: str, config: LlmConfig, http_client: httpx2.Client | None = None) -> None:
        """`http_client` is for tests, which fake OpenAI's replies."""
        self._config = config
        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.network_retries,
            http_client=http_client,
        )

    def send(
        self, instructions: str, messages: Sequence[Message], schema_name: str, schema: dict[str, Any]
    ) -> Reply:
        try:
            response = self._client.responses.create(
                model=self._config.model,
                instructions=instructions,
                input=[{"role": message.role, "content": message.content} for message in messages],
                # strict: the answer is guaranteed to match the schema, allowed values included.
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
                reasoning={"effort": self._config.reasoning_effort},
                max_output_tokens=self._config.max_output_tokens,
                store=False,  # OpenAI would otherwise keep every reply for 30 days; we don't need that
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            # Our own words: OpenAI's message repeats part of the key, and logs never show keys.
            raise LlmUnavailable(
                f"OpenAI rejected the API key (HTTP {error.status_code}); check OPENAI_API_KEY in .env"
            ) from error
        except openai.APIStatusError as error:
            raise LlmUnavailable(f"OpenAI refused the request (HTTP {error.status_code}): {_reason(error)}") from error
        except openai.APIError as error:  # timeouts and connection problems, after the retries
            raise LlmUnavailable(f"Could not reach OpenAI: {error}") from error
        return _reply_from(response)


MAX_REASON_CHARS = 300  # longest error text kept from the provider (some send whole web pages)


def _reason(error: openai.APIStatusError) -> str:
    """OpenAI's own short reason for refusing a request, or the start of whatever it sent instead."""
    if isinstance(error.body, dict) and isinstance(error.body.get("message"), str):
        return error.body["message"]
    return str(error.message)[:MAX_REASON_CHARS]


def _reply_from(response: Response) -> Reply:
    """Turn OpenAI's response into a Reply."""
    refusals = [
        part.refusal
        for item in response.output
        if item.type == "message"
        for part in item.content
        if part.type == "refusal"
    ]
    if refusals:
        stop, detail = "refused", refusals[0]
    elif response.status == "completed":
        stop, detail = "done", ""
    elif response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        stop = "cut_off" if reason == "max_output_tokens" else "refused"  # the other reason is a safety filter
        detail = f"the reply stopped early ({reason or 'no reason given'})"
    else:  # "failed", or anything new
        stop = "failed"
        detail = response.error.message if response.error else f"status {response.status}"

    usage = response.usage
    details = usage.input_tokens_details if usage else None
    return Reply(
        stop=stop,
        text=response.output_text,
        model=response.model,
        input_tokens=usage.input_tokens if usage else 0,
        cached_input_tokens=(details.cached_tokens or 0) if details else 0,
        output_tokens=usage.output_tokens if usage else 0,
        detail=detail,
    )


# --- The client -------------------------------------------------------------------------------------


class LlmClient:
    """Asks the LLM within the budget, records every reply, and returns only checked answers."""

    def __init__(
        self,
        provider: Provider,
        config: LlmConfig,
        session_factory: Callable[[], Session] | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        """`session_factory` and `now` are for tests; normally the real database and clock are used."""
        self.provider = provider
        self.config = config
        self._sessions = session_factory or sessionmaker(bind=get_engine(), expire_on_commit=False)
        self._now = now

    def ask(
        self,
        *,
        instructions: str,
        text: str,
        schema_name: str,
        schema: dict[str, Any],
        check: Callable[[str], BaseModel],
    ) -> Answer:
        """Ask once, and once more if the answer breaks the rules. Returns the checked answer.

        `check` turns the answer's JSON text into the result, raising AnswerError if it breaks the
        rules (e.g. llm/schemas.py's parse_answer). Raises LlmUnavailable (stop for now) or
        AnswerFailed (this request failed).
        """
        messages = [Message("user", text)]
        calls = input_tokens = output_tokens = 0
        cost = 0.0
        for attempt in (1, 2):
            self._check_budget()
            reply = self.provider.send(instructions, messages, schema_name, schema)
            cost += self._record(reply)
            calls += 1
            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens

            if reply.stop == "failed":
                raise LlmUnavailable(f"The LLM could not finish its reply: {reply.detail}")
            if reply.stop == "cut_off":
                # Not retried: the same request would be cut off again.
                raise AnswerFailed(f"The answer was cut off at max_output_tokens ({self.config.max_output_tokens})")
            if reply.stop == "refused":
                raise AnswerFailed(f"The LLM refused to answer: {reply.detail}")

            try:
                result = check(reply.text)
            except AnswerError as error:
                if attempt == 2:
                    raise AnswerFailed(f"The answer broke the rules twice:\n{error}") from error
                logger.info("The LLM's answer broke the rules; asking once more")
                messages += [
                    Message("assistant", reply.text),
                    Message("user", RETRY_REQUEST.format(problems=error)),
                ]
                continue
            return Answer(result, reply.model, calls, input_tokens, output_tokens, cost)
        raise AssertionError("unreachable")  # pragma: no cover

    # -- The budget guard and the usage records --

    def _check_budget(self) -> None:
        """Refuse to send anything once today's calls or this month's spend reach their limits."""
        today = self._now().date()
        with self._sessions() as session:
            calls = session.scalar(select(LlmUsage.calls).where(LlmUsage.day == today)) or 0
            spent = session.scalar(
                select(func.coalesce(func.sum(LlmUsage.est_cost_usd), 0.0)).where(
                    LlmUsage.day >= today.replace(day=1), LlmUsage.day <= today
                )
            )
        cap, budget = self.config.daily_call_cap, self.config.monthly_budget_usd
        if calls >= cap:
            raise BudgetReached(f"Daily cap reached: {calls} of {cap} LLM calls today (UTC); analysis resumes tomorrow")
        if spent >= budget:
            raise BudgetReached(
                f"Monthly budget reached: an estimated ${spent:.2f} of ${budget:.2f} this month (UTC); "
                "analysis resumes next month"
            )

    def _record(self, reply: Reply) -> float:
        """Add the reply to today's usage, in its own save. Returns its estimated cost."""
        today = self._now().date()
        cost = estimate_cost_usd(reply, self.config.prices_usd_per_million)
        _add_usage(self._sessions, today, reply, cost)
        logger.info(
            "LLM call: %d input tokens (%d cached), %d output tokens, about $%.5f",
            reply.input_tokens, reply.cached_input_tokens, reply.output_tokens, cost,
        )
        self._warn_if_crossed(today)
        return cost

    def _warn_if_crossed(self, today: date) -> None:
        """Warn on the one call that passes warn_at_fraction of the daily cap."""
        with self._sessions() as session:
            calls = session.scalar(select(LlmUsage.calls).where(LlmUsage.day == today)) or 0
        cap = self.config.daily_call_cap
        line = cap * self.config.warn_at_fraction
        if calls - 1 < line <= calls:
            logger.warning("LLM calls today: %d of the daily cap of %d (%.0f%%)", calls, cap, 100 * calls / cap)


def _add_usage(session_factory: Callable[[], Session], day: date, reply: Reply, cost: float) -> None:
    """Add one call to the day's totals, creating the day's row if it is the first call.

    Done in one statement ("insert, or add to the row if it exists"), so it stays correct even if
    two processes record a call at the same moment.
    """
    statement = insert(LlmUsage).values(
        day=day, calls=1, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens, est_cost_usd=cost
    )
    statement = statement.on_conflict_do_update(
        index_elements=[LlmUsage.day],
        set_={
            "calls": LlmUsage.calls + 1,
            "input_tokens": LlmUsage.input_tokens + statement.excluded.input_tokens,
            "output_tokens": LlmUsage.output_tokens + statement.excluded.output_tokens,
            "est_cost_usd": LlmUsage.est_cost_usd + statement.excluded.est_cost_usd,
        },
    )
    with session_factory() as session, session.begin():
        session.execute(statement)


# --- Setting it up --------------------------------------------------------------------------------


def make_client(
    session_factory: Callable[[], Session] | None = None,
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
) -> LlmClient:
    """The client set up from .env (provider and key) and settings.yaml (everything else).

    `reasoning_effort` replaces the setting for this client only, e.g. to compare efforts in the
    accuracy check. Raises LlmUnavailable if the client can't be set up, e.g. the key is missing.
    """
    settings = get_settings()
    config = load_app_settings().llm
    if reasoning_effort is not None:
        config = config.model_copy(update={"reasoning_effort": reasoning_effort})
    if settings.llm_provider != "openai":
        raise LlmUnavailable(f"LLM_PROVIDER={settings.llm_provider} is not built yet; set LLM_PROVIDER=openai in .env")
    if settings.openai_api_key is None:
        raise LlmUnavailable("OPENAI_API_KEY is not set in .env")
    provider = OpenAIProvider(settings.openai_api_key.get_secret_value(), config)
    return LlmClient(provider, config, session_factory)
