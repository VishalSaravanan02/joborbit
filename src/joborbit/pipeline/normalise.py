"""Cleans up a fetched job before it is saved.

- html_to_text: turns the description's HTML into readable plain text, trimmed
  to a sensible length (the LLM reads this later, and shorter means cheaper).
- content_hash: a fingerprint of the job's text, to notice when a company edits an ad.
- job_fingerprint: identifies "the same job" even when it reappears under a new ID,
  so a re-posted job doesn't trigger a second alert.
"""

import hashlib
import re

from bs4 import BeautifulSoup

MAX_DESCRIPTION_CHARS = 6000

# Tags that start a new line of text when the HTML is turned into plain text.
BLOCK_TAGS = ["p", "div", "br", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"]


def html_to_text(html: str | None, max_chars: int = MAX_DESCRIPTION_CHARS) -> str | None:
    """Turn description HTML into tidy plain text, at most `max_chars` long."""
    if not html or not html.strip():
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()  # never part of the visible text
    for tag in soup.find_all("li"):
        tag.insert_before("\n• ")  # keep bullet points readable
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")

    lines = (" ".join(line.split()) for line in soup.get_text().splitlines())
    text = "\n".join(line for line in lines if line and line != "•")

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"  # cut at a word boundary
    return text or None


def _simplify(value: str | None) -> str:
    """Lower-case letters and digits only: "Data Scientist (London)" -> "data scientist london"."""
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def content_hash(title: str, location: str | None, description_text: str | None) -> str:
    """Changes whenever the title, location or description text changes."""
    combined = "\n".join([title, location or "", description_text or ""])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def job_fingerprint(company_id: int, title: str, location: str | None) -> str:
    """The same for a job at the same company with the same title and location,
    regardless of spacing, capitals or punctuation."""
    combined = f"{company_id}|{_simplify(title)}|{_simplify(location)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
