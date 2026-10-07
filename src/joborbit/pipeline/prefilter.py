"""The pre-filter: cheap keyword rules that decide which new jobs are worth an LLM call.

A job passes only if every check passes, in this order (cheapest and clearest first):

1. Country: the location names a switched-on country, or is too vague to tell
   ("Remote", "EMEA", blank), in which case the LLM decides later.
2. Not senior: the title contains none of the drop words in seniority.yaml.
3. Not an internship: unless at least one user wants internships.
4. Role: the title matches a keyword of a role that at least one user has.

Every decision comes with a short reason, stored with the job so it can be audited.

How titles are matched. Titles and keywords are both reduced to lower-case words without
punctuation, so "Data-Scientist (London)" is "data scientist london". A keyword matches
exactly when its words appear together, in order. It also matches fuzzily when a run of
the same number of title words is close to it word by word: every word at least
FUZZY_WORD_SCORE similar. That catches typos and plurals ("scientst", "analysts").
Comparing word by word matters: comparing whole phrases would accept "QA Engineer" as
"AI Engineer", because only two letters differ. Drop, internship and exclude words only
ever match exactly, so "lead" never drops "Leading".
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from joborbit.config import load_countries, load_roles, load_seniority
from joborbit.pipeline.location import parse_location

FUZZY_WORD_SCORE = 85  # out of 100: how similar each word must be for a fuzzy match

Words = tuple[str, ...]


def words(text: str | None) -> Words:
    """Lower-case words without punctuation: "Sr. Data-Scientist (UK)" -> ("sr", "data", "scientist", "uk")."""
    return tuple(re.findall(r"[^\W_]+", (text or "").lower()))


def contains(title: Words, phrase: Words) -> bool:
    """True if the phrase's words appear in the title together and in order."""
    size = len(phrase)
    return any(title[start : start + size] == phrase for start in range(len(title) - size + 1))


def fuzzy_contains(title: Words, phrase: Words) -> bool:
    """True if some run of title words is close to the phrase, word by word."""
    size = len(phrase)
    for start in range(len(title) - size + 1):
        run = title[start : start + size]
        if all(fuzz.ratio(word, wanted) >= FUZZY_WORD_SCORE for word, wanted in zip(run, phrase, strict=True)):
            return True
    return False


def _first_found(title: Words, phrases: tuple[Words, ...]) -> str | None:
    """The first phrase that appears exactly in the title, as text, or None."""
    for phrase in phrases:
        if contains(title, phrase):
            return " ".join(phrase)
    return None


@dataclass(frozen=True)
class RoleRule:
    """One role to look for: a default role (by slug) or a custom role (by its name)."""

    role: str  # the slug, e.g. "data_scientist", or the custom role's name, e.g. "Insights Analyst"
    keywords: tuple[Words, ...]
    exclude: tuple[Words, ...] = ()


@dataclass(frozen=True)
class RoleMatch:
    """A role a title matched, and whether the match was exact (scoring values exact matches more)."""

    role: str
    exact: bool


@dataclass(frozen=True)
class Decision:
    """The pre-filter's verdict on one job."""

    passed: bool
    reason: str  # why it was rejected, or which roles it matched
    roles: tuple[RoleMatch, ...] = ()


@dataclass(frozen=True)
class PrefilterRules:
    """Everything the pre-filter needs, built once and reused for every job in a run."""

    enabled_countries: frozenset[str]
    roles: tuple[RoleRule, ...]
    drop_words: tuple[Words, ...]
    internship_words: tuple[Words, ...]
    allow_internships: bool

    @classmethod
    def build(cls, role_slugs: set[str], custom_roles: set[str], allow_internships: bool) -> "PrefilterRules":
        """Rules for the given roles (every active user's, combined), using the config files.

        `role_slugs` are default roles; `custom_roles` are role names users typed, matched as
        written. Unknown slugs are ignored, since a role removed from roles.yaml simply stops matching.
        """
        rules = [
            RoleRule(
                role.slug,
                keywords=tuple(words(keyword) for keyword in role.keywords),
                exclude=tuple(words(word) for word in role.exclude),
            )
            for role in load_roles()
            if role.slug in role_slugs
        ]
        rules += [RoleRule(name, (words(name),)) for name in sorted(custom_roles) if words(name)]
        seniority = load_seniority()
        return cls(
            enabled_countries=frozenset(country.code for country in load_countries() if country.enabled),
            roles=tuple(rules),
            drop_words=tuple(words(word) for word in seniority.drop_words),
            internship_words=tuple(words(word) for word in seniority.internship_words),
            allow_internships=allow_internships,
        )


def match_roles(title: Words, roles: tuple[RoleRule, ...]) -> tuple[RoleMatch, ...]:
    """Every role the title matches (exact matches preferred), skipping roles it excludes."""
    matches = []
    for rule in roles:
        if _first_found(title, rule.exclude):
            continue
        if any(contains(title, keyword) for keyword in rule.keywords):
            matches.append(RoleMatch(rule.role, exact=True))
        elif any(fuzzy_contains(title, keyword) for keyword in rule.keywords):
            matches.append(RoleMatch(rule.role, exact=False))
    return tuple(matches)


def check_job(title: str, location_raw: str | None, rules: PrefilterRules) -> Decision:
    """Run every check on one job and return the verdict with its reason."""
    if not rules.roles:
        return Decision(False, "no active users")

    location = parse_location(location_raw)
    ours = [code for code in location.countries if code in rules.enabled_countries]
    if not ours and not location.ambiguous:
        if location.countries:
            return Decision(False, f"country: {', '.join(location.countries)} (not switched on)")
        return Decision(False, "country: not one we cover")

    title_words = words(title)
    senior = _first_found(title_words, rules.drop_words)
    if senior:
        return Decision(False, f"senior title: {senior}")

    internship = _first_found(title_words, rules.internship_words)
    if internship and not rules.allow_internships:
        return Decision(False, f"internship: {internship}")

    roles = match_roles(title_words, rules.roles)
    if not roles:
        return Decision(False, "no matching role")
    described = ", ".join(match.role if match.exact else f"{match.role} (fuzzy)" for match in roles)
    return Decision(True, f"roles: {described}", roles)
