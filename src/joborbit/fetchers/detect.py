"""Works out which applicant tracking system (ATS) a company uses, from its careers page.

Most companies don't build their own job pages; they use an ATS, and links to
it give it away: "jobs.lever.co/palantir" means Lever, with the token "palantir".
Detection checks the URL first, then the careers page's HTML, then makes one
test fetch to prove the token really works.

Some ATSs are recognised but not supported yet (e.g. Workday). Recognising them
still helps: it tells us which fetcher would be most useful to build next.
"""

import re
from dataclasses import dataclass

from joborbit.fetchers.registry import get_fetcher
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
class Detection:
    """The outcome of detecting a company's ATS.

    status is one of:
      "ready"       - supported ATS, and a test fetch worked
      "unsupported" - ATS recognised, but we have no fetcher for it yet
      "not_found"   - no ATS could be identified
      "failed"      - something went wrong (page unreachable, token rejected...)
    """

    status: str
    ats_type: str | None = None
    token: str | None = None
    job_count: int | None = None
    message: str = ""


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


async def detect_ats(careers_url: str, client: PoliteClient) -> Detection:
    """Identify a company's ATS from its careers URL, then prove it with one test fetch."""
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
