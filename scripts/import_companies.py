"""Copy the company CSV into the database: adds new companies and updates existing ones.

Usage:
    python scripts/import_companies.py [CSV_FILE]

CSV_FILE defaults to data/companies_seed.csv. Safe to run again whenever the CSV
changes; companies are matched by name, and nothing is ever deleted.
"""

import argparse
from pathlib import Path

from joborbit.companies import read_company_csv, upsert_companies
from joborbit.db.session import session_scope
from joborbit.settings import PROJECT_ROOT

DEFAULT_CSV = PROJECT_ROOT / "data" / "companies_seed.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", nargs="?", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    try:
        rows = read_company_csv(args.csv_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with session_scope() as session:
        summary = upsert_companies(session, rows)

    print(f"Imported {len(rows)} companies: {summary.added} added, {summary.updated} updated.")
    print(f"{summary.active} active (will be fetched), {summary.inactive} inactive (unsupported or no ATS yet).")


if __name__ == "__main__":
    main()
