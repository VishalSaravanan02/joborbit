"""Application settings loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project's top folder: src/joborbit/settings.py -> up two levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every setting JobOrbit reads from the environment.

    Values come from real environment variables first, then from the .env
    file in the project root. Secrets use SecretStr, so they print as
    '**********' instead of their real value if ever logged by accident.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # treat "KEY=" (empty) as "not set"
        extra="ignore",  # ignore unknown variables instead of crashing
    )

    # --- Secrets ---
    anthropic_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    session_secret: SecretStr | None = None

    # --- Telegram ---
    telegram_bot_username: str | None = None
    admin_telegram_id: int | None = None

    # --- Backups ---
    backup_public_key: str | None = None

    # --- App ---
    database_path: Path = Path("var/joborbit.db")
    base_url: str = "http://localhost:8000"
    contact_email: str | None = None
    env: Literal["development", "production"] = "development"
    dry_run: bool = True

    @property
    def database_file(self) -> Path:
        """Absolute path to the SQLite database file."""
        if self.database_path.is_absolute():
            return self.database_path
        return PROJECT_ROOT / self.database_path

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, reading .env only once."""
    return Settings()