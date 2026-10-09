"""Database tables for JobOrbit, written as Python classes (SQLAlchemy ORM).

Each class is one table, and each attribute is one column. Tables are added
in the step that first needs them. All times are stored in UTC.

Shared tables (one copy for everyone): companies, jobs, fetch_runs, job_analysis, llm_usage.
Personal tables (one row per user): users, user_profiles, user_company_prefs, matches.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
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
    added_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
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
        Index("ix_jobs_became_new_at", "became_new_at"),
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
    # When the job last became new: when first seen, or when it came back after a long gap as a
    # fresh posting (first_seen_at stays the same then). Re-matching after a profile change looks
    # back from this, so a job that became new again is never missed.
    became_new_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
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
    # Deleting a job deletes its analysis with it.
    analysis: Mapped["JobAnalysis | None"] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )
    # Deleting a job deletes its matches with it.
    matches: Mapped[list["Match"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )


class JobAnalysis(Base):
    """The LLM's reading of one job: one row per analysed job.

    Every answer is checked by joborbit/llm/schemas.py before it is saved here.
    None means the posting doesn't say. A job analysed again has this row updated.
    """

    __tablename__ = "job_analysis"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    countries: Mapped[list[str]] = mapped_column(JSON, default=list)  # where the job can be done, e.g. ["GB"]
    cities: Mapped[list[str]] = mapped_column(JSON, default=list)
    work_mode: Mapped[str | None] = mapped_column(String(20))  # "onsite", "hybrid" or "remote"
    seniority: Mapped[str | None] = mapped_column(String(20))  # "intern", "graduate", "entry", "mid" or "senior"
    is_graduate_scheme: Mapped[bool] = mapped_column(Boolean, default=False)
    experience_years: Mapped[int | None] = mapped_column(Integer)
    experience_mandatory: Mapped[bool] = mapped_column(Boolean, default=False)  # required, not just preferred
    # Language codes, e.g. ["en", "es"]; "other" means a language not in languages.yaml.
    required_languages: Mapped[list[str]] = mapped_column(JSON, default=list)
    role_families: Mapped[list[str]] = mapped_column(JSON, default=list)  # role slugs, best first; [] = none of ours
    matched_custom_roles: Mapped[list[str]] = mapped_column(JSON, default=list)  # users' custom role names
    skills: Mapped[list[str]] = mapped_column(JSON, default=list)
    min_degree: Mapped[str | None] = mapped_column(String(20))  # "none", "bachelor", "master" or "phd"
    deadline: Mapped[date | None] = mapped_column(Date)
    summary: Mapped[str] = mapped_column(String(200))  # one line for alerts
    model: Mapped[str] = mapped_column(String(100))  # the LLM that made this analysis
    prompt_version: Mapped[str] = mapped_column(String(50))  # the prompt it was given
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    analysed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="analysis")


class LlmUsage(Base):
    """LLM use on one day (UTC), for the budget guard.

    Every request counts, retries included, because every request is paid for.
    """

    __tablename__ = "llm_usage"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    est_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)  # estimated from token prices


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


class User(Base):
    """A person who receives alerts. Their Telegram ID is the only identifier we store."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True)  # can exceed 32 bits
    display_name: Mapped[str] = mapped_column(String(100))  # Telegram first name
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)  # /pause sets this
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Deleting a user deletes their profile, company choices and matches with them.
    profile: Mapped["UserProfile | None"] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    company_prefs: Mapped[list["UserCompanyPref"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    matches: Mapped[list["Match"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class UserProfile(Base):
    """What one user wants: the answers from the profile form (or my_profile.yaml for now).

    The rules for valid values (limits, known codes...) live in joborbit/profiles.py.
    """

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list)  # default role slugs, e.g. ["data_scientist"]
    custom_roles: Mapped[list[str]] = mapped_column(JSON, default=list)  # role names the user typed
    countries: Mapped[list[str]] = mapped_column(JSON, default=list)  # ranked: first = most wanted
    languages: Mapped[list[str]] = mapped_column(JSON, default=list)  # e.g. ["en", "es"]
    preferred_industries: Mapped[list[str]] = mapped_column(JSON, default=list)
    excluded_industries: Mapped[list[str]] = mapped_column(JSON, default=list)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list)
    highest_degree: Mapped[str | None] = mapped_column(String(20))  # "bachelor", "master" or "phd"
    include_internships: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_style: Mapped[str] = mapped_column(String(20), default="balanced")  # "fewer", "balanced", "more"
    transfer_boost: Mapped[bool] = mapped_column(Boolean, default=True)
    # Personal scoring weights, set only by the weekly tuning. Empty = use the current defaults.
    weights: Mapped[dict[str, float] | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="profile")


class UserCompanyPref(Base):
    """A company one user marked as a favourite or excluded."""

    __tablename__ = "user_company_prefs"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # "favourite" or "excluded"
    never_miss: Mapped[bool] = mapped_column(Boolean, default=False)  # favourites only

    user: Mapped[User] = relationship(back_populates="company_prefs")
    company: Mapped[Company] = relationship()


class Match(Base):
    """One job that passed one user's hard filters, with its score and where it was routed.

    A job is matched at most once per user. Matching the same job again (after a profile
    change, say) updates this row instead of adding another.
    """

    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("user_id", "job_id", name="uq_matches_user_id_job_id"),
        Index("ix_matches_job_id", "job_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    score: Mapped[int] = mapped_column(Integer)  # 0 to 100
    components: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)  # points per scoring component
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)  # short "why" lines for the alert
    tier: Mapped[str] = mapped_column(String(10))  # "instant", "digest" or "silent"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)  # when the score was last worked out

    user: Mapped[User] = relationship(back_populates="matches")
    job: Mapped[Job] = relationship(back_populates="matches")
