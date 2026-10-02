"""Tests for reading the company CSV and copying it into the database."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from joborbit.companies import (
    CompanyRow,
    apply_detections,
    read_company_csv,
    slugify,
    upsert_companies,
    write_company_csv,
)
from joborbit.db.models import Base, Company
from joborbit.fetchers.detect import Detection

HEADER = "name,size_category,industry,countries,careers_url,ats_type,ats_token,notes\n"


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def write_csv(tmp_path, body: str):
    path = tmp_path / "companies.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_slugify():
    assert slugify("Standard Chartered") == "standard-chartered"
    assert slugify("  Monzo Bank Ltd. ") == "monzo-bank-ltd"


def test_reads_and_tidies_rows(tmp_path):
    path = write_csv(tmp_path, "Monzo, Medium ,fintech, gb|es ,https://monzo.com/careers,Greenhouse,monzo,\n")
    [row] = read_company_csv(path)
    assert row.name == "Monzo"
    assert row.size_category == "medium"
    assert row.countries == ["GB", "ES"]
    assert row.ats_type == "greenhouse"
    assert row.notes is None  # blank becomes None


def test_blank_ats_columns_are_allowed(tmp_path):
    [row] = read_company_csv(write_csv(tmp_path, "Acme,startup,technology,GB,https://acme.com/careers,,,\n"))
    assert row.ats_type is None and row.ats_token is None


def test_every_bad_row_is_reported_at_once(tmp_path):
    path = write_csv(
        tmp_path,
        "Good Ltd,startup,technology,GB,,,,\n"
        "Bad Size,huge,technology,GB,,,,\n"
        "Bad Country,startup,technology,United Kingdom,,,,\n"
        ",startup,technology,GB,,,,\n",
    )
    with pytest.raises(ValueError) as error:
        read_company_csv(path)
    message = str(error.value)
    assert "line 3" in message and "line 4" in message and "line 5" in message
    assert "line 2" not in message


def test_duplicate_companies_are_rejected(tmp_path):
    path = write_csv(tmp_path, "Monzo,medium,fintech,GB,,,,\nmonzo,medium,fintech,GB,,,,\n")
    with pytest.raises(ValueError, match="duplicate company"):
        read_company_csv(path)


def test_writing_then_reading_gives_the_same_rows(tmp_path):
    rows = [CompanyRow(name="Monzo", size_category="medium", countries="GB|ES", ats_type="greenhouse", ats_token="monzo")]
    path = tmp_path / "out.csv"
    write_company_csv(path, rows)
    assert read_company_csv(path) == rows


def test_import_adds_companies_and_sets_active_correctly(session):
    rows = [
        CompanyRow(name="Monzo", size_category="medium", countries="GB", ats_type="greenhouse", ats_token="monzo"),
        CompanyRow(name="Big Bank", size_category="mnc", countries="GB|SG", ats_type="workday", ats_token="bb|wd3|x"),
        CompanyRow(name="Mystery Ltd", size_category="startup"),
    ]
    summary = upsert_companies(session, rows)
    session.commit()
    assert (summary.added, summary.updated, summary.active, summary.inactive) == (3, 0, 1, 2)
    active = {c.name: c.active for c in session.scalars(select(Company))}
    assert active == {"Monzo": True, "Big Bank": False, "Mystery Ltd": False}


def test_importing_again_updates_instead_of_duplicating(session):
    upsert_companies(session, [CompanyRow(name="Monzo", size_category="medium", countries="GB")])
    session.commit()
    summary = upsert_companies(
        session, [CompanyRow(name="Monzo", size_category="medium", countries="GB|ES", ats_type="greenhouse", ats_token="monzo")]
    )
    session.commit()
    assert (summary.added, summary.updated) == (0, 1)
    [company] = session.scalars(select(Company)).all()
    assert company.countries == ["GB", "ES"]
    assert company.active is True


def test_changing_a_company_source_resets_its_baseline(session):
    upsert_companies(session, [CompanyRow(name="Acme", size_category="startup", ats_type="lever", ats_token="acme")])
    session.commit()
    company = session.scalars(select(Company)).one()
    company.baseline_done = True
    session.commit()

    upsert_companies(session, [CompanyRow(name="Acme", size_category="startup", ats_type="ashby", ats_token="acme")])
    session.commit()
    assert company.baseline_done is False


def test_detections_fill_in_only_certain_results():
    rows = [
        CompanyRow(name="Ready Co", size_category="startup", careers_url="https://ready.example"),
        CompanyRow(name="Workday Co", size_category="mnc", careers_url="https://wd.example"),
        CompanyRow(name="Broken Co", size_category="startup", careers_url="https://broken.example"),
        CompanyRow(name="Untouched Co", size_category="startup", ats_type="lever", ats_token="untouched"),
    ]
    detections = {
        "ready-co": Detection("ready", "greenhouse", "readyco", job_count=12),
        "workday-co": Detection("unsupported", "workday", "wdco|wd3|careers"),
        "broken-co": Detection("failed", "lever", "brokenco", message="HTTP 404"),
    }
    result = {row.name: (row.ats_type, row.ats_token) for row in apply_detections(rows, detections)}
    assert result == {
        "Ready Co": ("greenhouse", "readyco"),
        "Workday Co": ("workday", "wdco|wd3|careers"),
        "Broken Co": (None, None),  # failed: left for you to fix by hand
        "Untouched Co": ("lever", "untouched"),
    }
