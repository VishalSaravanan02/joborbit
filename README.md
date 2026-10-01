# JobOrbit

A personal job-alert tool that watches company careers pages and sends instant Telegram alerts for new graduate and entry-level data roles, so you can be among the first to apply.

> **Status:** in active development.

## How it works

1. Every 20 minutes, JobOrbit checks the public job feeds of hundreds of companies.
2. New jobs pass through cheap keyword filters, then a small LLM (Claude Haiku) reads each one and extracts the country, seniority, required experience, languages and skills.
3. Each user's own profile scores every job out of 100.
4. Strong matches are pushed to Telegram instantly; medium matches arrive in a daily digest.
5. Tapping "Relevant" saves a job to the dashboard, and the scoring learns from that feedback each week.

## Tech stack

Python 3.12 · httpx · SQLAlchemy + SQLite · Alembic · Anthropic API · python-telegram-bot · FastAPI + Jinja2 + HTMX · APScheduler · pytest

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pre-commit install
cp .env.example .env   # then fill in your own values
```

## Data and privacy

JobOrbit only reads public job feeds and never scrapes LinkedIn or Indeed. Users' data is limited to their Telegram ID, first name and the preferences they choose to enter. No CVs, passwords or contact details are stored.
