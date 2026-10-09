"""Alert routing: where a scored job goes for one user, from their alert style's thresholds.

There are three tiers:

- instant: a Telegram alert straight away;
- digest: a line in the next morning's digest;
- silent: shown on the dashboard only.

The rules, in order:

1. A favourite company with "never miss" on: always instant (it passed the hard filters, so the
   level, country and languages are right).
2. A score at or above the user's instant threshold: instant, unless the ad is evergreen (first
   posted long before we saw it). Instant alerts are for applying before everyone else, and an ad
   that has been up for months has no such race, so an evergreen ad goes to the digest instead.
3. A score at or above the digest threshold: digest.
4. Anything lower: silent.

The thresholds for each alert style ("fewer", "balanced", "more") are in settings.yaml
(scoring.alert_styles). A paused user's matches are routed as usual: pausing stops sending,
which the notifier checks, so nothing is missing from the dashboard when they resume.
"""

from joborbit.config import ScoringConfig, load_app_settings
from joborbit.matching.filters import Verdict
from joborbit.matching.scoring import Score

INSTANT, DIGEST, SILENT = "instant", "digest", "silent"


def route(score: Score, verdict: Verdict, alert_style: str, config: ScoringConfig | None = None) -> str:
    """The tier for a job that passed the user's hard filters (`verdict`) with this `score`."""
    config = config or load_app_settings().scoring
    styles = config.alert_styles
    thresholds = {"fewer": styles.fewer, "balanced": styles.balanced, "more": styles.more}.get(alert_style)
    if thresholds is None:
        raise ValueError(f"unknown alert style {alert_style!r}; expected fewer, balanced or more")

    if verdict.never_miss:
        return INSTANT
    if score.score >= thresholds.instant:
        return DIGEST if score.evergreen else INSTANT
    if score.score >= thresholds.digest:
        return DIGEST
    return SILENT
