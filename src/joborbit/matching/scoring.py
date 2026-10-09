"""The score out of 100: how well a job fits one user, for a job that passed their hard filters.

Five parts each give a value from 0 to 1, multiplied by the user's weight for that part
(settings.yaml, scoring.weights, unless the weekly tuning has set personal weights):

- role fit: 1.0 if the title contains a keyword of one of the user's roles (or a custom role's
  name); 0.7 if the role came from a near match or only from the LLM.
- entry fit: from the level. Graduate scheme or graduate 1.0, entry (or internship) 0.9, no clue
  0.6, mid 0.25; at most 0.5 when experience is required with no number given.
- skills: the share of the job's skills the user has, counting at most 6 of the job's skills,
  so 3 of 6 or more is 0.5. Neutral (0.5) when either side lists none.
- country: the job's best country in the user's ranked list: 1.0 for the first, falling evenly
  to 0.5 for the last. 0.5 when the posting doesn't say where (the "country unclear" pass).
- transfer: 1.0 if the user wants the transfer boost and the company works in one of their top
  two countries and in another of their countries; otherwise 0.

Bonuses are added on top (a favourite company, a preferred industry) and the total is capped at
100. The exact values live in settings.yaml, so they can be tuned without changing code.

With the score come up to MAX_REASONS short reasons (the parts that gave the most points, in
plain words, for the alert) and notes the user should always see: the location isn't stated,
experience is required with no number, or the ad is old (first posted long before we saw it).
"""

import math
import re
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache

from joborbit.config import ScoringConfig, load_app_settings, load_countries, load_industries, load_roles
from joborbit.matching.facts import JobFacts, UserFacts
from joborbit.matching.filters import Verdict
from joborbit.pipeline.prefilter import PrefilterRules, RoleRule, contains, match_roles, words

PARTS = ("role_fit", "entry_fit", "skills", "country", "transfer")
MAX_REASONS = 3
MAX_SKILLS_NAMED = 3  # skills named in the skills reason

NOTE_COUNTRY_UNCLEAR = "Location not stated: check it's open in your country"
NOTE_EXPERIENCE_REQUIRED = "Experience required (no number given)"


@dataclass(frozen=True)
class Score:
    """A job's score for one user, with the reasons and notes shown in the alert."""

    score: int  # 0 to 100
    points: dict[str, float]  # points per part (and per bonus earned)
    reasons: list[str] = field(default_factory=list)  # the parts that gave the most points
    notes: list[str] = field(default_factory=list)  # things the user should always be told
    evergreen: bool = False  # first posted long before we saw it


# --- Skills ----------------------------------------------------------------------------------

# UK spellings and their US forms, applied to the end of each word, so both sides compare the same.
_SPELLING_ENDINGS = (
    ("isation", "ization"),
    ("ising", "izing"),
    ("ised", "ized"),
    ("ise", "ize"),
    ("yse", "yze"),
    ("elling", "eling"),
    ("elled", "eled"),
    ("programme", "program"),
)


def _word(word: str) -> str:
    """One word in a standard form: "LLMs" -> "llm", "visualisation" -> "visualization"."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]  # a plain plural
    for uk, us in _SPELLING_ENDINGS:
        if word.endswith(uk):
            return word[: -len(uk)] + us
    return word


def skill_words(skill: str) -> tuple[str, ...]:
    """A skill as standard words, keeping + and # ("C++", "C#"): "Data Visualisation" -> ("data", "visualization")."""
    return tuple(_word(word) for word in re.findall(r"[a-z0-9+#]+", skill.lower()))


def skills_match(job_skill: str, user_skill: str) -> bool:
    """True if either skill's words appear, in order, inside the other's.

    So "PyTorch" matches "deep learning in PyTorch", and "LLMs" matches "LLM".
    """
    job, user = skill_words(job_skill), skill_words(user_skill)
    if not job or not user:
        return False
    return contains(user, job) or contains(job, user)


def matched_skills(job_skills: tuple[str, ...], user_skills: tuple[str, ...]) -> list[str]:
    """The job's skills the user has, as the job wrote them, each once."""
    found: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for skill in job_skills:
        key = skill_words(skill)
        if key and key not in seen and any(skills_match(skill, mine) for mine in user_skills):
            seen.add(key)
            found.append(skill)
    return found


# --- Names for the reasons -----------------------------------------------------------------------


def _role_names() -> dict[str, str]:
    return {role.slug: role.name for role in load_roles()}


def _country_names() -> dict[str, str]:
    return {country.code: country.name for country in load_countries()}


def _industry_names() -> dict[str, str]:
    return {industry.slug: industry.name for industry in load_industries()}


@lru_cache
def _role_rules(roles: frozenset[str], custom_roles: tuple[str, ...]) -> tuple[RoleRule, ...]:
    """The pre-filter's title rules for one user's roles, built once per set of roles."""
    return PrefilterRules.build(set(roles), set(custom_roles), allow_internships=True).roles


# --- The parts ------------------------------------------------------------------------------------


def _role_fit(job: JobFacts, user: UserFacts, config: ScoringConfig) -> tuple[float, str | None]:
    """The role-fit value, and the name of the role it came from."""
    title_matches = match_roles(words(job.title), _role_rules(user.roles, user.custom_roles))
    names = _role_names()
    for title_match in title_matches:
        if title_match.exact:
            return config.role_fit.exact, names.get(title_match.role, title_match.role)

    llm_roles = [slug for slug in job.role_families if slug in user.roles]
    custom = [name for name in user.custom_roles if name.lower() in job.matched_custom_roles]
    if llm_roles:
        return config.role_fit.other, names[llm_roles[0]]
    if custom:
        return config.role_fit.other, custom[0]
    if title_matches:  # a near match only
        return config.role_fit.other, names.get(title_matches[0].role, title_matches[0].role)
    return 0.0, None  # a "never miss" job that isn't one of the user's roles


def _entry_fit(job: JobFacts, config: ScoringConfig) -> tuple[float, str | None]:
    """The entry-fit value, and the level in words when it is worth telling the user."""
    values = config.entry_fit
    if job.is_graduate_scheme:
        value, words_ = values.graduate, "Graduate scheme"
    elif job.seniority == "graduate":
        value, words_ = values.graduate, "Graduate role"
    elif job.seniority == "entry":
        value, words_ = values.entry, "Entry level"
    elif job.seniority == "intern":
        value, words_ = values.entry, "Internship"
    elif job.seniority is None:
        value, words_ = values.unknown, None
    elif job.seniority == "mid":
        value, words_ = values.mid, None
    else:  # senior jobs never pass the hard filters
        value, words_ = 0.0, None

    if job.experience_mandatory and job.experience_years is None and value > values.required_without_years:
        return values.required_without_years, None  # capped: the note explains it
    return value, words_


def _skills(job: JobFacts, user: UserFacts, config: ScoringConfig) -> tuple[float, list[str]]:
    """The skills value, and the job's skills the user has."""
    job_skills = {skill_words(skill): skill for skill in job.skills if skill_words(skill)}
    if not job_skills or not user.skills:
        return config.skills.unknown, []
    found = matched_skills(tuple(job_skills.values()), user.skills)
    counted = min(len(job_skills), config.skills.max_counted)
    return min(1.0, len(found) / counted), found


def _country(job: JobFacts, user: UserFacts, verdict: Verdict, config: ScoringConfig) -> tuple[float, int | None]:
    """The country value, and the place of the job's best country in the user's list (0 = first)."""
    if verdict.country_unclear:
        return config.country.unclear, None
    places = [place for place, code in enumerate(user.countries) if code in job.countries]
    if not places:
        return 0.0, None  # can't happen after the hard filters
    best, last = min(places), len(user.countries) - 1
    if last == 0:
        return 1.0, best
    return 1.0 - (1.0 - config.country.last_choice) * best / last, best


def _transfer(job: JobFacts, user: UserFacts) -> tuple[float, list[str]]:
    """1.0 if the company works in one of the user's top two countries and another of theirs."""
    shared = [code for code in user.countries if code in job.company_countries]
    if user.transfer_boost and len(shared) >= 2 and set(shared) & set(user.countries[:2]):
        return 1.0, shared
    return 0.0, []


def is_evergreen(job: JobFacts, config: ScoringConfig) -> bool:
    """True if the ad was first posted long before it became new to us: an old ad, re-listed."""
    if job.posted_at is None:
        return False
    return job.became_new_at - job.posted_at > timedelta(days=config.evergreen_after_days)


def _weights(user: UserFacts, config: ScoringConfig) -> dict[str, float]:
    """The default weights, with the user's personal weights (from tuning) in their place."""
    return {**config.weights.model_dump(), **(user.weights or {})}


# --- The score ---------------------------------------------------------------------------------------


def score_job(job: JobFacts, user: UserFacts, verdict: Verdict, config: ScoringConfig | None = None) -> Score:
    """Score a job that passed the user's hard filters (`verdict`), with reasons and notes."""
    config = config or load_app_settings().scoring
    weights = _weights(user, config)

    role_value, role_name = _role_fit(job, user, config)
    entry_value, level_words = _entry_fit(job, config)
    skills_value, skills_found = _skills(job, user, config)
    country_value, country_place = _country(job, user, verdict, config)
    transfer_value, transfer_countries = _transfer(job, user)
    values = {
        "role_fit": role_value, "entry_fit": entry_value, "skills": skills_value,
        "country": country_value, "transfer": transfer_value,
    }
    points = {part: weights[part] * values[part] for part in PARTS}

    # Each part that is worth telling the user, with the points it gave.
    countries, industries = _country_names(), _industry_names()
    told: list[tuple[float, str]] = []
    if role_name:
        told.append((points["role_fit"], f"Role: {role_name}"))
    if level_words:
        told.append((points["entry_fit"], level_words))
    if skills_found:
        named = ", ".join(skills_found[:MAX_SKILLS_NAMED])
        told.append((points["skills"], f"{len(skills_found)} of your skills: {named}"))
    if country_place is not None:
        code = user.countries[country_place]
        place = job.cities[0] if job.cities and len(job.countries) == 1 else countries.get(code, code)
        told.append((points["country"], f"{place}, your #{country_place + 1} country"))
    if transfer_countries:
        others = [countries.get(code, code) for code in transfer_countries[1:]]
        told.append((points["transfer"], f"Company also in {', '.join(others)}"))

    bonuses = config.bonuses
    if job.company_id in user.favourite_company_ids:
        points["favourite_company"] = bonuses.favourite_company
        told.append((bonuses.favourite_company, "Favourite company"))
    if job.company_industry in user.preferred_industries:
        points["preferred_industry"] = bonuses.preferred_industry
        name = industries.get(job.company_industry, job.company_industry)
        told.append((bonuses.preferred_industry, f"Preferred industry: {name}"))

    total = min(100, math.floor(sum(points.values()) + 0.5))  # rounded half up, never above 100
    told.sort(key=lambda item: item[0], reverse=True)  # stable: equal points keep the order above
    reasons = [text for gained, text in told if gained > 0][:MAX_REASONS]

    evergreen = is_evergreen(job, config)
    notes = []
    if verdict.country_unclear:
        notes.append(NOTE_COUNTRY_UNCLEAR)
    if job.experience_mandatory and job.experience_years is None:
        notes.append(NOTE_EXPERIENCE_REQUIRED)
    if evergreen:
        notes.append(f"Originally posted {job.posted_at:%b %Y}")

    return Score(
        score=total,
        points={part: round(value, 2) for part, value in points.items()},
        reasons=reasons,
        notes=notes,
        evergreen=evergreen,
    )
