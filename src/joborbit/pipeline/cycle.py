"""What happens to jobs after each fetch: the pre-filter, then the LLM analysis.

process_new_jobs is the one place these steps are joined, so scripts/run_cycle.py and (later)
the worker run exactly the same thing. Fetching is left to the caller, so this can be tested
without any network.

The order matters:

1. (Only if asked) failed jobs go back to "pending", e.g. after a prompt fix.
2. The pre-filter decides every waiting job, not only this cycle's: jobs left over by a run that
   crashed between steps are picked up too. Its verdicts are saved before step 3 starts, so no
   write is held open during the LLM calls.
3. The analysis takes every job waiting for it, newest first, including any left by a run that
   stopped at the daily cap. If the LLM can't be used (no key, unreachable, budget used up), the
   verdicts are kept and the jobs simply wait for the next run: nothing is lost.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from joborbit.db.session import get_engine
from joborbit.llm.client import LlmClient, LlmUnavailable, make_client
from joborbit.pipeline.analyse import AnalysisRun, pending_job_ids, retry_failed, run_analysis
from joborbit.pipeline.prefilter import PrefilterRun, run_prefilter

logger = logging.getLogger(__name__)


@dataclass
class Processing:
    """What the steps after a fetch did."""

    prefilter: PrefilterRun
    retried: int  # failed jobs sent back for another try
    analysis: AnalysisRun | None  # None when analysis was switched off
    waiting: int  # jobs still waiting for analysis at the end


def process_new_jobs(
    analyse: bool = True,
    limit: int | None = None,
    retry: bool = False,
    session_factory: Callable[[], Session] | None = None,
    client: LlmClient | None = None,
) -> Processing:
    """Pre-filter every waiting job, then (if `analyse`) ask the LLM about at most `limit` of them.

    `retry` first sends failed jobs back to "pending". `client` is for tests; normally the client
    is set up from .env and settings.yaml, and a missing key just stops the analysis.
    """
    sessions = session_factory or sessionmaker(bind=get_engine(), expire_on_commit=False)
    with sessions() as session, session.begin():
        retried = retry_failed(session) if retry else 0
        prefilter = run_prefilter(session)
    logger.info(
        "Pre-filter: %d checked, %d passed%s",
        prefilter.checked, len(prefilter.passed_job_ids), f", {retried} failed jobs retried" if retried else "",
    )

    analysis = None
    if analyse:
        try:
            analysis = run_analysis(client or make_client(sessions), sessions, limit)
        except LlmUnavailable as error:  # the client couldn't even be set up, e.g. no key
            logger.warning("Analysis not started: %s", error)
            analysis = AnalysisRun(stopped=str(error))

    with sessions() as session:
        waiting = len(pending_job_ids(session))
    return Processing(prefilter, retried, analysis, waiting)
