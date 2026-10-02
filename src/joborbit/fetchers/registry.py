"""The list of job sites JobOrbit can fetch from, looked up by name.

Adding a new site later means writing its fetcher and adding one line here.
"""

from joborbit.fetchers.ashby import AshbyFetcher
from joborbit.fetchers.base import Fetcher
from joborbit.fetchers.greenhouse import GreenhouseFetcher
from joborbit.fetchers.lever import LeverFetcher

FETCHERS: dict[str, Fetcher] = {
    fetcher.ats_type: fetcher
    for fetcher in (
        GreenhouseFetcher(),
        LeverFetcher(),
        AshbyFetcher(),
    )
}


def get_fetcher(ats_type: str | None) -> Fetcher | None:
    """The fetcher for an ATS name such as "greenhouse", or None if we don't support it yet."""
    return FETCHERS.get(ats_type or "")


def is_supported(ats_type: str | None) -> bool:
    return get_fetcher(ats_type) is not None
