"""Save you as JobOrbit's admin user, with the profile in my_profile.yaml.

A stand-in for the dashboard's profile page (Phase 4), which will replace this script.

Usage:
    python scripts/seed_my_profile.py [PROFILE_FILE]

PROFILE_FILE defaults to my_profile.yaml in the project folder; copy
my_profile.example.yaml to start one. Safe to run again after every change: the
database is updated to match the file exactly, and nothing is saved if the file has
a mistake. Your Telegram ID comes from ADMIN_TELEGRAM_ID in .env.
"""

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select

from joborbit.db.models import User
from joborbit.db.session import session_scope
from joborbit.profiles import ProfileInput, UnknownCompaniesError, save_profile
from joborbit.settings import PROJECT_ROOT, get_settings

DEFAULT_FILE = PROJECT_ROOT / "my_profile.yaml"


class ProfileFile(BaseModel):
    """What my_profile.yaml contains."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=100)
    profile: ProfileInput


def describe(error: ValidationError) -> str:
    """Pydantic's error list as short plain lines, e.g. "profile > skills: at most 40 skills"."""
    lines = []
    for problem in error.errors():
        where = " > ".join(str(part) for part in problem["loc"]) or "file"
        if problem["type"] == "extra_forbidden":
            message = "not a field JobOrbit knows (a typo?)"
        else:
            message = problem["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {message}")
    return "\n".join(lines)


def read_profile_file(path: Path) -> ProfileFile:
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Copy my_profile.example.yaml to {path.name} and fill it in.")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    try:
        return ProfileFile.model_validate(data)
    except ValidationError as error:
        raise SystemExit(f"Problems in {path.name} (nothing was saved):\n{describe(error)}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile_file", nargs="?", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()

    telegram_id = get_settings().admin_telegram_id
    if telegram_id is None:
        raise SystemExit("ADMIN_TELEGRAM_ID is not set in .env (get your ID from @userinfobot).")
    file = read_profile_file(args.profile_file)

    try:
        with session_scope() as session:
            user = session.scalars(select(User).where(User.telegram_id == telegram_id)).one_or_none()
            created = user is None
            if created:
                user = User(telegram_id=telegram_id)
                session.add(user)
            user.display_name = file.display_name
            user.is_admin = True
            save_profile(session, user, file.profile)
    except UnknownCompaniesError as error:
        raise SystemExit(f"Problem in {args.profile_file.name} (nothing was saved): companies {error}") from error

    profile = file.profile
    never_miss = sum(company.never_miss for company in profile.favourite_companies)
    print(f"{'Created' if created else 'Updated'} {file.display_name} (admin) with this profile:")
    print(f"  roles: {len(profile.roles)} default + {len(profile.custom_roles)} custom")
    print(f"  countries: {', '.join(profile.countries)}    languages: {', '.join(profile.languages)}")
    print(f"  industries: {len(profile.preferred_industries)} preferred, {len(profile.excluded_industries)} excluded")
    print(f"  skills: {len(profile.skills)}    alert style: {profile.alert_style}")
    print(
        f"  companies: {len(profile.favourite_companies)} favourite ({never_miss} never miss), "
        f"{len(profile.excluded_companies)} excluded"
    )


if __name__ == "__main__":
    main()
