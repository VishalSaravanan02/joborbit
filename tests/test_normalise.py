"""Tests for turning descriptions into plain text and for job fingerprints."""

import json
from pathlib import Path

from joborbit.fetchers.ashby import AshbyFetcher
from joborbit.fetchers.greenhouse import GreenhouseFetcher
from joborbit.fetchers.lever import LeverFetcher
from joborbit.pipeline.normalise import content_hash, html_to_text, job_fingerprint

FIXTURES = Path(__file__).parent / "fixtures"


def test_html_becomes_readable_text():
    html = (
        "<h2>About the role</h2><p>Join our <strong>data</strong>&nbsp;team.</p>"
        "<ul><li>Python</li><li>SQL &amp; statistics</li></ul>"
        "<script>alert('x')</script><p>Line one<br>Line two</p>"
    )
    assert html_to_text(html) == (
        "About the role\nJoin our data team.\n• Python\n• SQL & statistics\nLine one\nLine two"
    )


def test_empty_descriptions_give_none():
    assert html_to_text(None) is None
    assert html_to_text("") is None
    assert html_to_text("   ") is None
    assert html_to_text("<div></div>") is None


def test_long_descriptions_are_cut_at_a_word_boundary():
    text = html_to_text("<p>" + "word " * 2000 + "</p>", max_chars=50)
    assert len(text) <= 52
    assert text.endswith(" …")
    assert "wor …" not in text  # no half-words


def test_real_descriptions_from_all_three_sites_convert_cleanly():
    greenhouse = json.loads((FIXTURES / "greenhouse.json").read_text())["jobs"]
    lever = json.loads((FIXTURES / "lever.json").read_text())
    ashby = json.loads((FIXTURES / "ashby.json").read_text())["jobs"]
    jobs = [
        *GreenhouseFetcher().parse_jobs(greenhouse, "monzo"),
        *LeverFetcher().parse_jobs(lever, "palantir"),
        *AshbyFetcher().parse_jobs(ashby, "elevenlabs"),
    ]
    assert len(jobs) == 9
    for job in jobs:
        text = html_to_text(job.description_html)
        assert text and len(text) > 200
        assert "<" not in text and "&lt;" not in text and "&amp;" not in text


def test_fingerprint_ignores_capitals_spacing_and_punctuation():
    assert job_fingerprint(1, "Data Scientist", "London, UK") == job_fingerprint(1, "  data   scientist ", "london uk")


def test_fingerprint_differs_by_company_title_or_location():
    base = job_fingerprint(1, "Android Engineer", "London")
    assert job_fingerprint(2, "Android Engineer", "London") != base
    assert job_fingerprint(1, "iOS Engineer", "London") != base
    assert job_fingerprint(1, "Android Engineer", "Barcelona") != base


def test_content_hash_changes_when_the_ad_is_edited():
    original = content_hash("Data Analyst", "London", "Requires SQL.")
    assert content_hash("Data Analyst", "London", "Requires SQL.") == original
    assert content_hash("Data Analyst", "London", "Requires SQL and Python.") != original
