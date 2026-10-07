"""The pre-filter on real jobs from the database, with the verdict each one must always get.

These titles and locations come from the first pre-filter report (October 2026), run on the
UK companies' open jobs. They include the tricky cases found while tuning the keywords:
fellowships, graduate analysts, "Software Engineer - Machine Learning", quant developers and
Cambridge, Massachusetts. If a change to roles.yaml, seniority.yaml or countries.yaml breaks
one of them, this test says which, so the change can be checked on purpose.

The rules use all 15 default roles with internships off, as in Vishal's profile.
"""

import pytest

from joborbit.config import load_roles
from joborbit.pipeline.prefilter import PrefilterRules, check_job

RULES = PrefilterRules.build({role.slug for role in load_roles()}, set(), allow_internships=False)

# (company, title, location, expected reason)
REAL_JOBS = [
    # --- Should pass, with the roles they match ---
    ("Monzo", "Data Scientist", "Cardiff, London or Remote (UK)", "roles: data_scientist"),
    ("Marshmallow", "Pricing Data Scientist", "London", "roles: data_scientist"),
    ("Trainline", "Junior Machine Learning Engineer", "London", "roles: ml_engineer"),
    ("Deliveroo", "Machine Learning Engineer II - Growth & Personalisation", "London", "roles: ml_engineer"),
    ("Figma", "Software Engineer - Machine Learning (London, United Kingdom)", "London, England / London, UK",
     "roles: ml_engineer"),
    ("Isomorphic Labs", "Research Scientist (Machine Learning), London", "London",
     "roles: ml_engineer, applied_scientist"),
    ("Synthesia", "Research Engineer in Data", "Europe", "roles: applied_scientist"),  # vague location
    ("Palantir", "Forward Deployed AI Engineer", "London, United Kingdom", "roles: ai_engineer"),
    ("Lendable", "Junior Data Engineer", "London", "roles: data_engineer"),
    ("Lendable", "Data Platform Engineer", "London", "roles: data_engineer"),
    ("Lendable", "Python Analytics Engineer", "London", "roles: analytics_engineer"),
    ("Lendable", "Growth Analyst", "London", "roles: product_analyst"),
    ("Stripe", "People Analytics Analyst & Business Partner", "Dublin, London", "roles: data_analyst"),
    ("Squarepoint Capital", "Graduate Quant Developer", "London, Montreal, Singapore", "roles: quant_analyst"),
    ("Squarepoint Capital", "Junior Quant Researcher - ML Alpha Research", "London, New York, Singapore",
     "roles: quant_analyst"),
    ("Lendable", "Graduate Analyst - £50,000 + Share Options", "London", "roles: tech_graduate_scheme"),
    ("Zopa", "2027 Graduate Analyst", "London", "roles: tech_graduate_scheme"),
    ("Anthropic", "Anthropic Fellows Program, AI Safety & Security", "London, UK", "roles: tech_graduate_scheme"),
    # --- Too senior ---
    ("Trainline", "Senior Data Scientist", "London", "senior title: senior"),
    ("Faculty", "Lead Machine Learning Engineer", "UK - Remote", "senior title: lead"),
    ("Lendable", "Strategy Analytics Manager", "London", "senior title: manager"),
    ("Coinbase", "Concierge Specialist IV", "Hybrid - London, UK", "senior title: iv"),
    # --- Internships (switched off in the profile) ---
    ("Monzo", "Associate Data Scientist - Intern", "London", "internship: intern"),
    ("Lendable", "Credit Analyst Intern", "London", "internship: intern"),
    # --- Not in a switched-on country ---
    ("Isomorphic Labs", "Research Leader (Toxicology), Cambridge, MA", "Cambridge, MA", "country: not one we cover"),
    ("OKX", "Senior Manager, Risk Operations Strategy", "Singapore, Singapore", "country: SG (not switched on)"),
    ("Databricks", "Specialist Solutions Architect - Data Warehousing", "Seoul, South Korea",
     "country: not one we cover"),
    # --- Not one of the roles (some deliberately left out when tuning) ---
    ("Squarepoint Capital", "Junior Software Developer (C++)", "London, Montreal, Singapore", "no matching role"),
    ("Anthropic", "Applied AI Architect, Digital Natives Business", "London, UK", "no matching role"),
    ("Lendable", "Complaints Officer - Graduate", "London", "no matching role"),
    ("Spotify", "Fraud Analyst (Revenue Protection) - 12-Month Fixed-Term", "London", "no matching role"),
    ("Databricks", "AI Forward Deployed Engineer - London", "London, United Kingdom", "no matching role"),
]


@pytest.mark.parametrize(
    ("company", "title", "location", "expected"),
    REAL_JOBS,
    ids=[f"{company}: {title}" for company, title, _, _ in REAL_JOBS],
)
def test_real_job_gets_its_verdict(company, title, location, expected):
    assert check_job(title, location, RULES).reason == expected


def test_there_are_about_thirty_real_jobs_with_both_verdicts():
    passed = [job for job in REAL_JOBS if job[3].startswith("roles:")]
    assert len(REAL_JOBS) >= 30
    assert 10 <= len(passed) <= len(REAL_JOBS) - 10  # plenty of each kind
