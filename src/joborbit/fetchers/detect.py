"""Works out which applicant tracking system (ATS) a company uses, from its careers page.

Most companies don't build their own job pages; they use an ATS, and links to
it give it away: "jobs.lever.co/palantir" means Lever, with the token "palantir".
Detection checks the URL first, then the careers page's HTML, then makes one
test fetch to prove the token really works.

If that finds nothing that works, detection "probes": it asks every supported ATS
for a board under a few likely tokens made from the company's name. A token is
only a name, so a probe can find a different company with the same name; probe
results are therefore marked for a person to confirm before they are used.

Some ATSs are recognised but not supported yet (e.g. Workday). Recognising them
still helps: it tells us which fetcher would be most useful to build next.
"""

import re
from dataclasses import dataclass

from joborbit.fetchers.registry import FETCHERS, get_fetcher
from joborbit.utils.http import FetchError, PoliteClient

# (ats_type, pattern). The pattern's groups build the token (see _token_from_match).
# Order matters: API addresses first, because they're the most precise.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("greenhouse", re.compile(r"boards-api\.greenhouse\.io/v1/boards/([\w-]+)", re.I)),
    ("greenhouse", re.compile(r"greenhouse\.io/embed/job_board(?:/js)?\?for=([\w-]+)", re.I)),
    ("greenhouse", re.compile(r"(?:job-)?boards(?:\.eu)?\.greenhouse\.io/(?!embed)([\w-]+)", re.I)),
    ("lever", re.compile(r"api\.(eu\.)?lever\.co/v0/postings/([\w.-]+)", re.I)),
    ("lever", re.compile(r"jobs\.(eu\.)?lever\.co/([\w.-]+)", re.I)),
    ("ashby", re.compile(r"api\.ashbyhq\.com/posting-api/job-board/([\w.-]+)", re.I)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([\w.%-]+)", re.I)),
    # Recognised, but no fetcher yet:
    ("workday", re.compile(r"([\w-]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[a-z]{2}-[A-Z]{2}/)?([\w-]+)")),
    ("workable", re.compile(r"apply\.workable\.com/([\w-]+)", re.I)),
    ("smartrecruiters", re.compile(r"(?:careers|jobs)\.smartrecruiters\.com/([\w-]+)", re.I)),
    ("recruitee", re.compile(r"([\w-]+)\.recruitee\.com", re.I)),
    ("personio", re.compile(r"([\w-]+)\.jobs\.personio\.(?:de|com)", re.I)),
]


@dataclass(frozen=True)
class ProbeMatch:
    """A board that answered a probe: which ATS, under which token, with how many jobs."""

    ats_type: str
    token: str
    job_count: int
    sample_url: str  # one job's link, so a person can check it is the right company


@dataclass(frozen=True)
class Detection:
    """The outcome of detecting a company's ATS.

    status is one of:
      "ready"       - supported ATS, and a test fetch worked
      "probed"      - found by probing; works, but a person should confirm it is the right company
      "ambiguous"   - probing found several boards; a person must choose (see `matches`)
      "unsupported" - ATS recognised, but we have no fetcher for it yet
      "not_found"   - no ATS could be identified
      "failed"      - something went wrong (page unreachable, token rejected...)
    """

    status: str
    ats_type: str | None = None
    token: str | None = None
    job_count: int | None = None
    message: str = ""
    sample_url: str | None = None  # set for "probed"
    matches: tuple[ProbeMatch, ...] = ()  # set for "ambiguous"


def _token_from_match(ats_type: str, match: re.Match[str]) -> str:
    if ats_type == "lever":
        eu, name = match.groups()
        return f"eu:{name}" if eu else name
    if ats_type == "workday":
        tenant, server, site = match.groups()
        return f"{tenant}|{server}|{site}"
    return match.group(1)


def find_ats(text: str) -> tuple[str, str] | None:
    """Look for a known ATS link in a URL or page HTML. Returns (ats_type, token) or None."""
    for ats_type, pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return ats_type, _token_from_match(ats_type, match)
    return None


def name_tokens(name: str | None) -> list[str]:
    """Likely ATS tokens for a company name: "Google DeepMind" -> google-deepmind, googledeepmind."""
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return list(dict.fromkeys(token for token in ("-".join(words), "".join(words)) if token))


async def probe_ats(tokens: list[str], client: PoliteClient) -> list[ProbeMatch]:
    """Ask every supported ATS for a board under each token. Returns the boards that have jobs.

    A board with no jobs doesn't count: there is no job link to confirm it is the right
    company, and some sites may answer "no jobs" for any name at all.
    """
    matches = []
    for ats_type, fetcher in FETCHERS.items():
        for token in dict.fromkeys(tokens):
            try:
                jobs = await fetcher.fetch(token, client)
            except FetchError:
                continue  # no board under this token on this ATS
            if jobs:
                matches.append(ProbeMatch(ats_type, token, len(jobs), jobs[0].url))
    return matches


async def detect_ats(careers_url: str | None, client: PoliteClient, name: str | None = None) -> Detection:
    """Identify a company's ATS from its careers URL and prove it with a test fetch.

    If that doesn't give a working board, probe every supported ATS using the token
    from the URL (if any) and tokens made from the company's `name`.
    """
    result = await _detect_from_url(careers_url, client) if careers_url else None
    if result is not None and result.status in ("ready", "unsupported"):
        return result

    guessed = result.token if result is not None and result.token else None
    # The guessed token as written (an "eu:" token only works on Lever's EU servers) and without
    # any "eu:" prefix (for the other ATSs), then tokens made from the company's name.
    tokens = [guessed, guessed.removeprefix("eu:")] if guessed else []
    tokens += name_tokens(name)
    matches = await probe_ats(tokens, client) if tokens else []

    if len(matches) == 1:
        [match] = matches
        return Detection(
            "probed",
            match.ats_type,
            match.token,
            job_count=match.job_count,
            message="Found by probing: please confirm it is the right company",
            sample_url=match.sample_url,
        )
    if len(matches) > 1:
        found = ", ".join(f"{m.ats_type}/{m.token} ({m.job_count} jobs)" for m in matches)
        return Detection("ambiguous", message=f"Probing found several boards: {found}", matches=tuple(matches))
    if result is not None:
        return result
    return Detection("not_found", message="No careers URL, and probing found no board")


async def _detect_from_url(careers_url: str, client: PoliteClient) -> Detection:
    """Find the ATS from the URL or the careers page's HTML, then make one test fetch."""
    found = find_ats(careers_url)
    if found is None:
        try:
            page = await client.get_text(careers_url)
        except FetchError as exc:
            return Detection("failed", message=f"Could not open the careers page: {exc}")
        found = find_ats(page)
    if found is None:
        return Detection("not_found", message="No known ATS link on the careers page")

    ats_type, token = found
    fetcher = get_fetcher(ats_type)
    if fetcher is None:
        return Detection("unsupported", ats_type, token, message=f"{ats_type} is not supported yet")

    try:
        jobs = await fetcher.fetch(token, client)
    except FetchError as exc:
        return Detection("failed", ats_type, token, message=f"Test fetch failed: {exc}")
    return Detection("ready", ats_type, token, job_count=len(jobs))
