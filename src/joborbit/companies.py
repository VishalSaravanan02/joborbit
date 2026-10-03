"""Reading, checking and saving the company list (data/companies_seed.csv).

The company list is kept as a CSV file you can edit in any spreadsheet app.
These helpers read it, check every row, and copy it into the database.

CSV columns:
    name, size_category, industry, countries, careers_url, ats_type, ats_token, notes

- size_category: mnc, medium or startup
- countries: ISO codes separated by |, e.g. GB|IN|SG
- ats_type / ats_token: may be left blank; scripts/detect_ats_bulk.py fills them in
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from joborbit.db.models import Company
from joborbit.fetchers.detect import Detection
from joborbit.fetchers.registry import is_supported

CSV_COLUMNS = ["name", "size_category", "industry", "countries", "careers_url", "ats_type", "ats_token", "notes"]
SIZE_CATEGORIES = {"mnc", "medium", "startup"}


class CompanyRow(BaseModel):
    """One row of the company CSV, checked and tidied."""

    name: str = Field(min_length=1)
    size_category: str
    industry: str | None = None
    countries: list[str] = []
    careers_url: str | None = None
    ats_type: str | None = None
    ats_token: str | None = None
    notes: str | None = None

    @field_validator("name", "industry", "careers_url", "ats_type", "ats_token", "notes", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("size_category", mode="before")
    @classmethod
    def _check_size(cls, value: object) -> object:
        size = str(value or "").strip().lower()
        if size not in SIZE_CATEGORIES:
            raise ValueError(f"size_category must be one of {sorted(SIZE_CATEGORIES)}, not {value!r}")
        return size

    @field_validator("countries", mode="before")
    @classmethod
    def _split_countries(cls, value: object) -> object:
        if isinstance(value, str):
            codes = [code.strip().upper() for code in value.split("|") if code.strip()]
            for code in codes:
                if not re.fullmatch(r"[A-Z]{2}", code):
                    raise ValueError(f"country codes must be two letters, e.g. GB; got {code!r}")
            return codes
        return value or []

    @field_validator("ats_type", mode="after")
    @classmethod
    def _lowercase_ats(cls, value: str | None) -> str | None:
        return value.lower() if value else value

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass
class ImportSummary:
    added: int = 0
    updated: int = 0
    active: int = 0
    inactive: int = 0


def slugify(name: str) -> str:
    """"Standard Chartered" -> "standard-chartered"."""
    return "-".join(re.findall(r"[a-z0-9]+", name.lower()))


def read_company_csv(path: Path) -> list[CompanyRow]:
    """Read and check every row. Raises ValueError listing every bad row at once."""
    rows: list[CompanyRow] = []
    problems: list[str] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"The first line of {path.name} must be exactly:\n  {','.join(CSV_COLUMNS)}\n"
                f"but it is:\n  {','.join(reader.fieldnames or [])}"
            )
        for line_number, raw in enumerate(reader, start=2):  # line 1 is the header
            # A wrong number of commas would silently shift or drop values, so refuse the row.
            if None in raw:  # the csv module puts any extra values under the key None
                count = len(CSV_COLUMNS) + len(raw[None])
                problems.append(
                    f"line {line_number}: {count} columns instead of {len(CSV_COLUMNS)} "
                    '(an extra comma? Put text that contains commas in "double quotes")'
                )
                continue
            if None in raw.values():  # ...and missing values come back as None
                problems.append(f"line {line_number}: fewer than {len(CSV_COLUMNS)} columns")
                continue
            try:
                rows.append(CompanyRow.model_validate(raw))
            except ValueError as exc:
                problems.append(f"line {line_number}: {exc}")

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.slug in seen:
            problems.append(f"duplicate company {row.name!r} (rows {seen[row.slug] + 2} and {index + 2})")
        seen[row.slug] = index

    if problems:
        raise ValueError("Problems in the company list:\n" + "\n".join(problems))
    return rows


def write_company_csv(path: Path, rows: list[CompanyRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            data = row.model_dump()
            data["countries"] = "|".join(row.countries)
            writer.writerow({column: data[column] or "" for column in CSV_COLUMNS})


def upsert_companies(session: Session, rows: list[CompanyRow]) -> ImportSummary:
    """Add new companies and update existing ones (matched by slug). Never deletes.

    A company is active (fetched) only if it has a token for a supported ATS.
    """
    summary = ImportSummary()
    existing = {company.slug: company for company in session.scalars(select(Company))}
    for row in rows:
        company = existing.get(row.slug)
        if company is None:
            company = Company(slug=row.slug)
            session.add(company)
            summary.added += 1
        else:
            summary.updated += 1

        if company.ats_type != row.ats_type or company.ats_token != row.ats_token:
            company.baseline_done = False  # a new source starts with a fresh baseline

        company.name = row.name
        company.size_category = row.size_category
        company.industry = row.industry
        company.countries = row.countries
        company.careers_url = row.careers_url
        company.ats_type = row.ats_type
        company.ats_token = row.ats_token
        company.active = is_supported(row.ats_type) and bool(row.ats_token)
        if company.active:
            summary.active += 1
        else:
            summary.inactive += 1
    return summary


def apply_detections(rows: list[CompanyRow], detections: dict[str, Detection]) -> list[CompanyRow]:
    """Copy detected ATS details into the rows they belong to (matched by slug).

    "ready" and "unsupported" results are applied: in both cases we know the ATS.
    "probed" results are applied too, so they can be reviewed in the CSV before
    importing. Ambiguous, failed and not-found rows are left untouched to fix by hand.
    """
    updated = []
    for row in rows:
        detection = detections.get(row.slug)
        if detection is not None and detection.status in ("ready", "unsupported", "probed"):
            row = row.model_copy(update={"ats_type": detection.ats_type, "ats_token": detection.token})
        updated.append(row)
    return updated
