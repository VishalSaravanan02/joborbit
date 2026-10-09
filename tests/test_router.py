"""Tests for alert routing (joborbit/matching/router.py), with the real settings.yaml thresholds."""

import pytest

from joborbit.config import AlertThresholds, load_app_settings
from joborbit.matching.filters import Verdict
from joborbit.matching.router import DIGEST, INSTANT, SILENT, route
from joborbit.matching.scoring import Score

CONFIG = load_app_settings().scoring
PASSED = Verdict(passed=True, reason="passed")
NEVER_MISS = Verdict(passed=True, reason="passed", never_miss=True)


def scored(value: int, evergreen: bool = False, mid_required: bool = False) -> Score:
    return Score(score=value, points={}, evergreen=evergreen, mid_required=mid_required)


def with_cap(tier: str):
    return CONFIG.model_copy(update={"mid_required_max_tier": tier})


@pytest.mark.parametrize(
    ("style", "instant", "digest"),
    [("fewer", 80, 60), ("balanced", 75, 50), ("more", 65, 40)],
)
def test_each_alert_style_has_its_own_thresholds(style, instant, digest):
    assert route(scored(100), PASSED, style, CONFIG) == INSTANT
    assert route(scored(instant), PASSED, style, CONFIG) == INSTANT
    assert route(scored(instant - 1), PASSED, style, CONFIG) == DIGEST
    assert route(scored(digest), PASSED, style, CONFIG) == DIGEST
    assert route(scored(digest - 1), PASSED, style, CONFIG) == SILENT
    assert route(scored(0), PASSED, style, CONFIG) == SILENT


def test_an_evergreen_ad_that_would_be_instant_goes_to_the_digest():
    assert route(scored(90, evergreen=True), PASSED, "balanced", CONFIG) == DIGEST


@pytest.mark.parametrize(("value", "tier"), [(60, DIGEST), (30, SILENT)])
def test_an_evergreen_ad_below_the_instant_threshold_is_routed_as_usual(value, tier):
    assert route(scored(value, evergreen=True), PASSED, "balanced", CONFIG) == tier


@pytest.mark.parametrize("evergreen", [False, True])
def test_a_never_miss_favourite_is_always_instant(evergreen):
    assert route(scored(10, evergreen=evergreen), NEVER_MISS, "fewer", CONFIG) == INSTANT


def test_the_settings_given_are_the_ones_used():
    styles = CONFIG.alert_styles.model_copy(update={"balanced": AlertThresholds(instant=95, digest=90)})
    strict = CONFIG.model_copy(update={"alert_styles": styles})
    assert route(scored(92), PASSED, "balanced", strict) == DIGEST
    assert route(scored(92), PASSED, "balanced", CONFIG) == INSTANT


def test_the_settings_file_is_used_by_default():
    assert route(scored(75), PASSED, "balanced") == INSTANT
    assert route(scored(74), PASSED, "balanced") == DIGEST


def test_an_unknown_alert_style_is_refused():
    with pytest.raises(ValueError, match="unknown alert style 'loud'"):
        route(scored(90), PASSED, "loud", CONFIG)


# --- Mid-level jobs that require experience -----------------------------------------------------------


@pytest.mark.parametrize("value", [90, 60, 30])
def test_a_mid_level_job_requiring_experience_goes_no_higher_than_silent(value):
    assert route(scored(value, mid_required=True), PASSED, "balanced", CONFIG) == SILENT


@pytest.mark.parametrize(("value", "tier"), [(90, DIGEST), (60, DIGEST), (30, SILENT)])
def test_the_cap_can_be_raised_to_the_digest(value, tier):
    assert route(scored(value, mid_required=True), PASSED, "balanced", with_cap("digest")) == tier


@pytest.mark.parametrize(("value", "tier"), [(90, INSTANT), (60, DIGEST), (30, SILENT)])
def test_a_cap_of_instant_changes_nothing(value, tier):
    assert route(scored(value, mid_required=True), PASSED, "balanced", with_cap("instant")) == tier


def test_the_cap_and_the_evergreen_rule_together_give_the_lower_tier():
    assert route(scored(90, evergreen=True, mid_required=True), PASSED, "balanced", with_cap("digest")) == DIGEST
    assert route(scored(90, evergreen=True, mid_required=True), PASSED, "balanced", CONFIG) == SILENT


def test_a_never_miss_favourite_is_instant_even_when_capped():
    assert route(scored(30, mid_required=True), NEVER_MISS, "balanced", CONFIG) == INSTANT
