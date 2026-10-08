"""Tests for the accuracy check's labels, jobs file and scoring (joborbit/llm/accuracy.py)."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from joborbit.llm.accuracy import (
    LABELS,
    SCORED,
    TODO,
    EvalJob,
    Expected,
    Outcome,
    load_outcomes,
    read_jobs,
    save_outcomes,
    score,
    summarise,
    write_jobs,
)
from joborbit.llm.schemas import JobAnalysisResult

LABELLED = {
    "countries": ["GB"],
    "seniority": ["graduate", "entry"],
    "experience_mandatory": False,
    "experience_years": None,
    "required_languages": [],
    "role_families": ["data_analyst", "tech_graduate_scheme"],
}

# An answer that matches LABELLED on all five scored things.
ANSWER = {
    "countries": ["GB"],
    "cities": ["London"],
    "work_mode": "hybrid",
    "seniority": "graduate",
    "is_graduate_scheme": True,
    "experience_years": None,
    "experience_mandatory": False,
    "required_languages": [],
    "role_families": ["data_analyst"],
    "matched_custom_roles": [],
    "skills": ["SQL"],
    "min_degree": "bachelor",
    "deadline": None,
    "summary": "A data analyst graduate programme.",
}


def job(job_id: str = "acme/1", expected: dict | None = None, **changes) -> EvalJob:
    data = {
        "id": job_id, "company": "Acme", "title": "Graduate Data Analyst", "location": "London",
        "posted": date(2026, 10, 1), "url": "https://example.com/1", "description": "Join our data team.",
        "expected": LABELLED if expected is None else expected,
    }
    return EvalJob(**{**data, **changes})


def labels(**changes) -> Expected:
    return Expected.model_validate({**LABELLED, **changes})


def answer(**changes) -> JobAnalysisResult:
    return JobAnalysisResult.model_validate({**ANSWER, **changes})


# --- Labels ------------------------------------------------------------------------------------


def test_complete_labels_are_accepted():
    assert labels().role_families == ["data_analyst", "tech_graduate_scheme"]


def test_one_seniority_can_be_given_without_a_list():
    assert labels(seniority="graduate").seniority == ["graduate"]
    assert labels(seniority=None).seniority == [None]  # "the posting gives no hint"


def test_codes_are_tidied():
    tidied = labels(countries=["gb"], required_languages=["ES"], role_families=["Data_Analyst"])
    assert (tidied.countries, tidied.required_languages, tidied.role_families) == (["GB"], ["es"], ["data_analyst"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"countries": ["US"]}, "unknown country"),
        ({"required_languages": ["spanish"]}, "unknown language"),
        ({"role_families": ["data_wizard"]}, "unknown role"),
        ({"seniority": ["junior"]}, "seniority"),
        ({"seniority": []}, "seniority"),
        ({"experience_years": -1}, "experience_years"),
        ({"experience_mandatory": "sometimes"}, "experience_mandatory"),
        ({"langauges": ["es"]}, "langauges"),
    ],
)
def test_wrong_labels_are_refused(change, message):
    with pytest.raises(ValidationError, match=message):
        labels(**change)


def test_a_new_job_has_every_label_missing():
    assert job(expected={}).missing_labels() == list(LABELS)


def test_labels_still_todo_are_listed():
    partly = {**LABELLED, "seniority": TODO, "role_families": TODO}
    assert job(expected=partly).missing_labels() == ["seniority", "role_families"]


def test_reading_labels_names_the_job_and_the_problem():
    with pytest.raises(ValueError, match="acme/7: not labelled yet: seniority"):
        job("acme/7", {**LABELLED, "seniority": TODO}).labels()
    with pytest.raises(ValueError, match=r"acme/8: role_families: unknown role \['data_wizard'\]; choose from"):
        job("acme/8", {**LABELLED, "role_families": ["data_wizard"]}).labels()


# --- The jobs file -----------------------------------------------------------------------------


AWKWARD = '   Leading spaces\n\nKey: value # not a comment\n- "quoted" | pipe\n• Python & SQL — £50,000\n\n\nEnd'


def test_jobs_are_written_with_todo_labels_and_read_back_exactly(tmp_path):
    path = tmp_path / "accuracy" / "jobs.yaml"
    jobs = [
        job("acme/1", {}, company='Acme "Ltd": #1', title="Data: Analyst", description=AWKWARD),
        job("acme/2", {}, location=None, posted=None, description=None),
    ]
    write_jobs(path, jobs)
    back = read_jobs(path)
    assert [item.model_dump(exclude={"expected"}) for item in back] == [
        item.model_dump(exclude={"expected"}) for item in jobs
    ]
    assert all(item.missing_labels() == list(LABELS) for item in back)


def test_the_labels_come_after_each_description(tmp_path):
    """So you read the posting, then label it right below."""
    path = tmp_path / "jobs.yaml"
    write_jobs(path, [job("acme/1", {}), job("acme/2", {})])
    text = path.read_text(encoding="utf-8")
    for block in text.split("# --- Job ")[1:]:
        assert block.index("  description:") < block.index("  expected:")


def test_the_file_explains_how_to_label(tmp_path):
    path = tmp_path / "jobs.yaml"
    write_jobs(path, [job(expected={})])
    text = path.read_text(encoding="utf-8")
    assert "src/joborbit/llm/prompts.py" in text and "Label before looking at any LLM answer" in text


def test_labels_typed_into_the_file_are_read(tmp_path):
    path = tmp_path / "jobs.yaml"
    write_jobs(path, [job(expected={})])
    text = path.read_text(encoding="utf-8")
    for name, value in [
        ("countries", "[GB]"), ("seniority", "graduate"), ("experience_mandatory", "false"),
        ("experience_years", "null"), ("required_languages", "[]"), ("role_families", "[data_analyst]"),
    ]:
        text = text.replace(f"    {name}: TODO", f"    {name}: {value}")
    path.write_text(text, encoding="utf-8")
    [labelled] = read_jobs(path)
    assert labelled.labels() == labels(seniority=["graduate"], role_families=["data_analyst"])


def test_an_existing_jobs_file_is_never_replaced(tmp_path):
    path = tmp_path / "jobs.yaml"
    path.write_text("my labels", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_jobs(path, [job()])
    assert path.read_text(encoding="utf-8") == "my labels"


# --- Scoring -----------------------------------------------------------------------------------


def test_a_matching_answer_is_right_on_all_five():
    assert score(labels(), answer()) == dict.fromkeys(SCORED, True)


@pytest.mark.parametrize(
    ("field", "label_change", "answer_change", "right"),
    [
        ("countries", {"countries": ["GB", "ES"]}, {"countries": ["ES", "GB"]}, True),  # order doesn't matter
        ("countries", {}, {"countries": ["GB", "ES"]}, False),
        ("countries", {"countries": []}, {"countries": []}, True),
        ("seniority", {}, {"seniority": "entry"}, True),  # one of the accepted levels
        ("seniority", {}, {"seniority": "mid"}, False),
        ("seniority", {"seniority": [None]}, {"seniority": None}, True),
        ("experience", {}, {"experience_mandatory": True}, False),
        ("experience", {"experience_years": 2}, {"experience_years": 2}, True),
        ("experience", {"experience_years": 2}, {"experience_years": 3}, False),
        ("experience", {"experience_years": 0}, {"experience_years": None}, False),  # "none needed" isn't "not said"
        ("languages", {"required_languages": ["es", "en"]}, {"required_languages": ["en", "es"]}, True),
        ("languages", {}, {"required_languages": ["en"]}, False),
        ("languages", {"required_languages": ["es"]}, {"required_languages": []}, False),  # a language missed
        ("roles", {}, {"role_families": ["tech_graduate_scheme", "data_analyst"]}, True),
        ("roles", {}, {"role_families": ["data_analyst", "data_scientist"]}, False),  # one role not accepted
        ("roles", {}, {"role_families": []}, False),  # a role was expected
        ("roles", {"role_families": []}, {"role_families": []}, True),  # none fits, none given
        ("roles", {"role_families": []}, {"role_families": ["data_analyst"]}, False),
    ],
)
def test_each_scored_thing(field, label_change, answer_change, right):
    assert score(labels(**label_change), answer(**answer_change))[field] is right


def outcome(job_id: str, given: dict | None = None, error: str | None = None) -> Outcome:
    data = None if given is None and error else {**ANSWER, **(given or {})}
    return Outcome(job_id, data, error, input_tokens=2000, output_tokens=300, cost_usd=0.0004, seconds=3.0)


def test_a_run_is_summarised_job_by_job():
    jobs = [job("acme/1"), job("acme/2"), job("acme/3")]
    summary = summarise(
        jobs,
        [
            outcome("acme/1"),
            outcome("acme/2", {"seniority": "mid", "required_languages": ["en"]}),
            outcome("acme/3", error="The answer broke the rules twice"),
        ],
    )
    assert (summary.jobs, summary.right, summary.failed) == (3, 1, 1)
    assert summary.by_field == {"countries": 2, "seniority": 1, "experience": 2, "languages": 1, "roles": 2}
    assert summary.wrong == [
        ("acme/2", "seniority", "mid", ["graduate", "entry"]),
        ("acme/2", "languages", ["en"], []),
        ("acme/3", "no answer", "The answer broke the rules twice", None),
    ]
    assert (summary.input_tokens, summary.output_tokens) == (6000, 900)
    assert summary.cost_usd == pytest.approx(0.0012) and summary.seconds == pytest.approx(9.0)


def test_answers_are_scored_against_the_labels_as_they_are_now():
    """Correcting a label changes the score of the same saved answer."""
    saved = [outcome("acme/1", {"seniority": "mid"})]
    assert summarise([job("acme/1")], saved).right == 0
    assert summarise([job("acme/1", {**LABELLED, "seniority": ["mid"]})], saved).right == 1


def test_saved_answers_read_back_exactly(tmp_path):
    outcomes = [outcome("acme/1"), outcome("acme/2", error="refused")]
    path = tmp_path / "accuracy" / "answers-low.json"
    save_outcomes(path, "low", "1", "gpt-6-luna-2026-05-18", outcomes)
    details, back = load_outcomes(path)
    assert details == {"effort": "low", "prompt_version": "1", "model": "gpt-6-luna-2026-05-18"}
    assert back == outcomes
    assert json.loads(path.read_text(encoding="utf-8"))["outcomes"][0]["answer"]["seniority"] == "graduate"
