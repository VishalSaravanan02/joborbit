"""Time helpers. JobOrbit stores every time in UTC, without a time zone attached."""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current time in UTC, as a "naive" datetime (SQLite can't store time zones)."""
    return datetime.now(UTC).replace(tzinfo=None)
