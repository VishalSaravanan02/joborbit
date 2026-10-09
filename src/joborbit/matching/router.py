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
5. A mid-level job that requires experience without saying how much goes no higher than
   scoring.mid_required_max_tier (silent by default): a long shot for a graduate, kept on the
   dashboard but not pushed. Its role and country alone would otherwise put it in every digest.

The thresholds for each alert style ("fewer", "balanced", "more") are in settings.yaml
(scoring.alert_styles). A paused user's matches are routed as usual: pausing stops sending,
which the notifier checks, so nothing is missing from the dashboard when they resume.
"""

from joborbit.config import ScoringConfig, load_app_settings
from joborbit.matching.filters import Verdict
from joborbit.matching.scoring import Score

INSTANT, DIGEST, SILENT = "instant", "digest", "silent"
_ORDER = (SILENT, DIGEST, INSTANT)  # lowest first


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
        tier = DIGEST if score.evergreen else INSTANT
    elif score.score >= thresholds.digest:
        tier = DIGEST
    else:
        tier = SILENT
    if score.mid_required:
        tier = min(tier, config.mid_required_max_tier, key=_ORDER.index)
    return tier
