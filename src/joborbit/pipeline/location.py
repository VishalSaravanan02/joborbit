"""Works out which countries a job's location text refers to.

Companies write locations in endless ways: "Cardiff, London or Remote (UK)",
"Singapore, Singapore", "Remote", "EMEA", "Washington, D.C.". This module turns
that text into country codes for the countries in config/countries.yaml.

The answer has two parts:
- countries: the countries clearly mentioned, e.g. ["GB"].
- ambiguous: True when the text is too vague to tell ("Remote", "EMEA", blank).
  Ambiguous jobs are kept so the LLM can read the full ad and decide.

If the text names a specific place that isn't one of our countries
(e.g. "Washington, D.C."), countries is empty and ambiguous is False,
so the job can be safely ignored.

Known limitation: a city name shared with another country ("Cambridge, MA" in
the USA) is matched to our country. The LLM step later reads the full job and
corrects this, so the cost is one extra LLM call, not a wrong alert.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from joborbit.config import load_countries

# Words that, on their own, say nothing about which country a job is in.
VAGUE_WORDS = frozenset(
    """
    remote remotely worldwide anywhere global globally international distributed
    emea europe european eu apac asia pacific southeast south east north west
    hybrid flexible onsite on site office home work from first fully based
    multiple various locations location region regional time zone zones timezone timezones
    and or the in any of within
    """.split()
)


# Place names containing one of our aliases that belong to another country.
# They are removed before matching, so "New South Wales" (Australia) isn't read as Wales.
EXCLUDED_PHRASES = ("new south wales",)


@dataclass(frozen=True)
class LocationResult:
    countries: list[str] = field(default_factory=list)  # e.g. ["GB", "SG"]
    ambiguous: bool = False


@dataclass(frozen=True)
class _Pattern:
    code: str
    regex: re.Pattern[str]


def _whole_words(term: str) -> re.Pattern[str]:
    """A pattern matching `term` only as whole words: "uk" matches "Remote (UK)" but not "ukulele"."""
    return re.compile(rf"(?<![\w]){re.escape(term.lower())}(?![\w])")


@lru_cache
def _patterns() -> tuple[_Pattern, ...]:
    """Every alias and city for every country, longest first ("new delhi" before "delhi")."""
    entries: list[tuple[str, str]] = []
    for country in load_countries():
        for term in [*country.aliases, *country.cities]:
            entries.append((country.code, term))
    entries.sort(key=lambda entry: len(entry[1]), reverse=True)
    return tuple(_Pattern(code, _whole_words(term)) for code, term in entries)


def parse_location(text: str | None) -> LocationResult:
    """Turn location text into country codes, or mark it as ambiguous."""
    if not text or not text.strip():
        return LocationResult(ambiguous=True)

    lowered = text.lower()
    for phrase in EXCLUDED_PHRASES:
        lowered = lowered.replace(phrase, " ")
    found: list[str] = []
    for pattern in _patterns():
        if pattern.code not in found and pattern.regex.search(lowered):
            found.append(pattern.code)

    if found:
        order = [country.code for country in load_countries()]  # same order as the config file
        return LocationResult(countries=sorted(found, key=order.index))

    words = re.findall(r"[a-z]+", lowered)
    if all(word in VAGUE_WORDS for word in words):
        return LocationResult(ambiguous=True)
    return LocationResult()  # a specific place that isn't one of our countries
