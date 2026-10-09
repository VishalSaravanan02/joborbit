"""Shared test set-up: a check that the tests never change JobOrbit's real files.

var/ holds the real database, the dry run's paid-for answers and the review sheet with your
answers, the accuracy-check labels, and the logs. A test that reaches any of them (say, by calling a
script's main() without replacing what it writes) could quietly overwrite real work. Every file
under var/ is recorded before the tests and compared after them; any change fails the run.
"""

import pytest

from joborbit.settings import PROJECT_ROOT

VAR = PROJECT_ROOT / "var"


def _snapshot() -> dict[str, tuple[int, int]]:
    """Every file under var/, with its size and modification time."""
    if not VAR.exists():
        return {}
    return {str(path.relative_to(PROJECT_ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in VAR.rglob("*") if path.is_file()}


@pytest.fixture(autouse=True, scope="session")
def _real_files_are_never_touched():
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    assert not changed, f"the tests changed real files under var/ (created, edited or deleted): {changed}"
