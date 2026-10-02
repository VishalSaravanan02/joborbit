"""The template every fetcher follows, and the job format they all produce.

Each applicant tracking system (Greenhouse, Lever, Ashby...) returns jobs in its
own shape. A fetcher's only job is to turn that shape into NormalisedJob, so
the rest of JobOrbit never needs to know where a job came from.
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from joborbit.utils.http import PoliteClient

logger = logging.getLogger(__name__)


class NormalisedJob(BaseModel):
    """One job, in the same format no matter which ATS it came from."""

    external_id: str = Field(min_length=1)  # the job's ID on the ATS
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)  # where a person applies
    location_raw: str = ""  # location text exactly as the company wrote it
    description_html: str | None = None
    posted_at: datetime | None = None  # in UTC

    @field_validator("external_id", mode="before")
    @classmethod
    def _id_as_text(cls, value: Any) -> Any:
        """Some ATSs use numbers as IDs; we always store them as text."""
        return str(value).strip() if value is not None else value

    @field_validator("title", "location_raw", mode="before")
    @classmethod
    def _tidy_spaces(cls, value: Any) -> Any:
        """Turn "  Data   Scientist\\n" into "Data Scientist"."""
        return " ".join(value.split()) if isinstance(value, str) else value


class Fetcher(ABC):
    """The template for a fetcher. Each ATS gets one subclass.

    A subclass must set `ats_type` (e.g. "greenhouse") and write two methods:
    `fetch`, which returns every job currently open at one company, and
    `parse_job`, which turns one raw job from the ATS into a NormalisedJob.
    """

    ats_type: ClassVar[str]

    @abstractmethod
    async def fetch(self, token: str, client: PoliteClient) -> list[NormalisedJob]:
        """Return all open jobs for the company identified by `token` on this ATS."""

    @abstractmethod
    def parse_job(self, raw: Any) -> NormalisedJob:
        """Turn one job, as the ATS sent it, into a NormalisedJob."""

    def parse_jobs(self, raw_jobs: list[Any], token: str) -> list[NormalisedJob]:
        """Parse every raw job, skipping (and logging) any that are malformed.

        One broken job should never stop us seeing a company's other jobs.
        """
        jobs = []
        for raw in raw_jobs:
            try:
                jobs.append(self.parse_job(raw))
            except (ValidationError, KeyError, TypeError, AttributeError) as exc:
                logger.warning("Skipped a malformed %s job for %s: %s", self.ats_type, token, exc)
        return jobs


def parse_timestamp(value: Any) -> datetime | None:
    """Turn the many date formats ATSs use into a UTC datetime, or None if unusable.

    Handles ISO text ("2026-09-30T10:15:00Z", "2026-09-30T11:15:00+01:00") and
    Unix times in seconds or milliseconds (Lever uses milliseconds).
    Times without a time zone are assumed to be UTC.
    """
    if value is None or value == "":
        return None
    try:
        if isinstance(value, int | float):
            seconds = value / 1000 if value > 10_000_000_000 else value  # milliseconds?
            moment = datetime.fromtimestamp(seconds, tz=UTC)
        else:
            moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError):
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return moment
