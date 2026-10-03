"""Database tables for JobOrbit, written as Python classes (SQLAlchemy ORM).

Each class is one table, and each attribute is one column. Phase 1 needs
three tables for fetching jobs; the rest are added in Phase 2.
All times are stored in UTC.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from joborbit.utils.timeutil import utcnow


# Predictable names for indexes and constraints, so future changes to the
# tables (migrations) can find and alter them reliably on SQLite.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """The parent class every table inherits from."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Company(Base):
    """A company whose job feed we track."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(200), unique=True)
    size_category: Mapped[str] = mapped_column(String(20))  # "mnc", "medium" or "startup"
    industry: Mapped[str | None] = mapped_column(String(50))
    countries: Mapped[list[str]] = mapped_column(JSON, default=list)  # e.g. ["GB", "IN"]
    careers_url: Mapped[str | None] = mapped_column(String(500))
    ats_type: Mapped[str | None] = mapped_column(String(30))  # e.g. "greenhouse"
    ats_token: Mapped[str | None] = mapped_column(String(200))  # the company's ID on that ATS
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    added_by_user_id: Mapped[int | None] = mapped_column(Integer)  # linked to users in Phase 2
    baseline_done: Mapped[bool] = mapped_column(Boolean, default=False)
    last_fetch_ok_at: Mapped[datetime | None] = mapped_column(DateTime)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    jobs: Mapped[list["Job"]] = relationship(back_populates="company")


class Job(Base):
    """One job posting, as seen on a company's job feed."""

    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("company_id", "external_id", name="uq_jobs_company_external_id"),
        Index("ix_jobs_first_seen_at", "first_seen_at"),
        Index("ix_jobs_fingerprint", "fingerprint"),
        Index("ix_jobs_analysis_status", "analysis_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    ats_type: Mapped[str] = mapped_column(String(30))
    external_id: Mapped[str] = mapped_column(String(200))  # the job's ID on the ATS
    url: Mapped[str] = mapped_column(String(1000))
    title: Mapped[str] = mapped_column(String(300))
    location_raw: Mapped[str | None] = mapped_column(String(500))
    country_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    description_text: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    missing_count: Mapped[int] = mapped_column(Integer, default=0)  # fetches in a row it was absent
    content_hash: Mapped[str | None] = mapped_column(String(64))
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set when this job is a re-post of another job at the same company (same fingerprint).
    # The link lets the dashboard follow a saved job to its re-post if the original closes.
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    prefilter_status: Mapped[str] = mapped_column(String(20), default="pending")
    prefilter_reason: Mapped[str | None] = mapped_column(String(200))
    analysis_status: Mapped[str] = mapped_column(String(20), default="pending")

    company: Mapped[Company] = relationship(back_populates="jobs")


class FetchRun(Base):
    """One attempt to fetch one company's jobs, used for health checks."""

    __tablename__ = "fetch_runs"
    __table_args__ = (Index("ix_fetch_runs_company_started", "company_id", "started_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(10), default="running")  # "running", "ok", "error"
    jobs_returned: Mapped[int | None] = mapped_column(Integer)
    new_jobs: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
