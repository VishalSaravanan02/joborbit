"""Loads JobOrbit's YAML configuration files from the config/ folder.

Each file is checked against a Pydantic model when it's loaded, so a typo
in a config file gives a clear error straight away instead of odd behaviour later.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from joborbit.settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"


class CountryConfig(BaseModel):
    code: str  # ISO code, e.g. "GB"
    name: str
    enabled: bool = False
    timezone: str
    local_languages: list[str] = []
    aliases: list[str] = []
    cities: list[str] = []


class CountriesFile(BaseModel):
    countries: list[CountryConfig]


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache
def load_countries(path: Path = CONFIG_DIR / "countries.yaml") -> list[CountryConfig]:
    """Every country in countries.yaml, enabled or not."""
    return CountriesFile.model_validate(_read_yaml(path)).countries
