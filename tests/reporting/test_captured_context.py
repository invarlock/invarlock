"""Captured context describes complete recorded provenance without guessing identity."""

import hashlib
import json
from copy import deepcopy

import pytest

from invarlock import captured_reporting, record_reporting
from invarlock.captured_contracts import load_payloads
from invarlock.evaluation_records.templates import example_project
from invarlock.report_presentation import render_html, render_markdown
from tests._evaluation_support import build_pack, pack_json


def inputs(system="Follow the task exactly."):
    runs = {}
    for side in ("baseline", "subject"):
        records = []
        for index in range(3):
            messages = [{"role": "user", "content": f"Private user case {index}"}]
            if side == "subject":
                messages.insert(0, {"role": "system", "content": system})
            records.append(
                {
                    "id": f"case-{index}",
                    "input": "This is deliberately not the effective prompt.",
                    "context": {
                        "model_key": "model-32b",
                        "role": side + " prompt",
                        "effective_messages": messages,
                    },
                }
            )
        runs[side] = {"run_id": side, "records": records}
    return runs


def test_all_cases_are_paired_by_id_and_only_added_system_text_is_expanded():
    runs = inputs()
    runs["subject"]["records"].reverse()
    before = deepcopy(runs)
    subjects, context, changes, details = record_reporting._captured_context(runs)
    summary = " ".join(changes)
    assert "Both runs record model key model-32b" in summary
    assert "does not establish identical model weights or revisions" in summary
    assert "all 3 paired cases" in summary
    assert "all other effective messages are unchanged" in summary
    assert dict(context)["Subject model revision"].startswith("Unavailable")
    assert "baseline prompt" in str(context) and "subject prompt" in str(context)
    assert dict(context)["Baseline effective message roles"] == "user"
    assert dict(context)["Subject effective message roles"] == "system → user"
    assert details[0][1]["system_instruction"] == "Follow the task exactly."
    assert details[0][1]["truncated"] is False
    assert "Private user" not in str(details)
    assert runs == before


@pytest.mark.parametrize(
    "kind",
    ["missing", "mixed", "incomplete", "invalid", "empty", "different", "available"],
)
def test_identity_values_require_consistency_across_every_record(kind):
    runs = inputs()
    records = runs["subject"]["records"]
    if kind == "missing":
        for row in records:
            row.pop("context")
    elif kind == "mixed":
        records[-1]["context"]["model_key"] = "other-model"
    elif kind == "incomplete":
        records[-1]["context"].pop("model_key")
    elif kind == "invalid":
        for row in records:
            row["context"]["model_key"] = 42
    elif kind == "empty":
        records.clear()
    elif kind == "different":
        for row in records:
            row["context"]["model_key"] = "other-model"
    else:
        for row in records:
            row["context"].update(
                model_id="publisher/model", model_revision="revision-digest"
            )
    subjects, context, changes, _ = record_reporting._captured_context(runs)
    summary = " ".join(changes)
    if kind == "available":
        assert "publisher/model" in str(subjects) and "revision-digest" in str(context)
    elif kind == "different":
        assert "different model keys" in summary
    else:
        assert "missing or mixed" in summary
        assert "Both runs record model key" not in summary


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        "bad",
        [None],
        [{"role": 1, "content": "x"}],
        [{"role": " ", "content": "x"}],
        [{"role": "user", "content": []}],
        [{"role": "user", "content": "x", "tool_calls": []}],
    ],
)
def test_malformed_or_unrecognized_effective_messages_are_unavailable(messages):
    runs = inputs()
    runs["subject"]["records"][-1]["context"]["effective_messages"] = messages
    subjects, context, changes, details = record_reporting._captured_context(runs)
    summary = " ".join(changes)
    assert "Prompt comparison unavailable" in summary
    assert dict(context)["Subject effective message roles"].startswith("Unavailable")
    assert details == ()


@pytest.mark.parametrize(
    "change",
    [
        "identical",
        "different_content",
        "different_role",
        "nonuniform_system",
        "missing_id",
        "duplicate_id",
        "extra_message",
    ],
)
def test_prompt_change_claims_check_the_last_case_and_complete_id_set(change):
    runs = inputs()
    records = runs["subject"]["records"]
    if change == "identical":
        for row in records:
            row["context"]["effective_messages"].pop(0)
    elif change == "different_content":
        records[-1]["context"]["effective_messages"][-1]["content"] = (
            "changed user prompt"
        )
    elif change == "different_role":
        records[-1]["context"]["effective_messages"][0]["role"] = "developer"
    elif change == "nonuniform_system":
        records[-1]["context"]["effective_messages"][0]["content"] = (
            "Different instruction"
        )
    elif change == "missing_id":
        records.pop()
    elif change == "duplicate_id":
        records[-1]["id"] = records[0]["id"]
    else:
        records[-1]["context"]["effective_messages"].append(
            {"role": "assistant", "content": "extra"}
        )
    subjects, context, changes, details = record_reporting._captured_context(runs)
    summary = " ".join(changes)
    assert details == ()
    if change == "identical":
        assert "unchanged across all 3 paired cases" in summary
    elif change in ("missing_id", "duplicate_id"):
        assert "Prompt comparison unavailable" in summary
    else:
        assert "not one uniform added system instruction" in summary
    if change in ("different_role", "extra_message"):
        assert (
            dict(context)["Subject effective message roles"] == "Mixed across records"
        )


def test_large_context_has_bounded_expansion_and_explicit_truncation():
    system = "<script>" + "x" * 100000
    runs = inputs(system)
    for row in runs["baseline"]["records"]:
        row["context"]["model_id"] = "model" * 10000
    subjects, context, changes, details = record_reporting._captured_context(runs)
    summary = " ".join(changes)
    preview = details[0][1]
    assert preview["system_instruction"] == system[:4096]
    assert preview["characters"] == len(system) and preview["truncated"] is True
    assert preview["sha256"] == hashlib.sha256(system.encode()).hexdigest()
    assert len(json.dumps((summary, subjects, context, details))) < 9000
    assert "truncated" in subjects[0][1]
    for run in runs.values():
        for row in run["records"]:
            row["context"]["effective_messages"] = [
                {"role": "tool" * 100, "content": "x"}
            ] * 100
    subjects, context, _, _ = record_reporting._captured_context(runs)
    assert len(str(subjects)) < 1000
    assert len(str(context)) < 2500
    assert "truncated" in str(context)


def test_both_captured_adapters_show_escaped_context_without_mutating_evidence():
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for side, run in (("baseline", baseline), ("subject", subject)):
        for row in run["records"]:
            messages = [{"role": "user", "content": "USER_TEXT_NOT_FOR_REPORT"}]
            if side == "subject":
                messages.insert(
                    0, {"role": "system", "content": "<script>alert('prompt')</script>"}
                )
            row["context"] = {
                "model_key": "<img src=x onerror=alert(1)>",
                "role": side,
                "effective_messages": messages,
            }
    snapshot = build_pack(baseline, subject, policy)
    original = dict(snapshot.files)
    comparison = pack_json(snapshot, "report")
    manifest, payloads, signer = load_payloads(snapshot)
    for view in (
        record_reporting._view(comparison, snapshot),
        captured_reporting._view(manifest, payloads, signer),
    ):
        html = render_html(view)
        markdown = render_markdown(view, include_details=True)
        for rendered in (html, markdown):
            assert "&lt;img" in rendered
            assert "<img src=x" not in rendered
            assert "USER_TEXT_NOT_FOR_REPORT" not in rendered
            assert "evaluator-recorded provenance" in rendered
        assert "<script>alert('prompt')</script>" not in html
        assert "&lt;script&gt;" in html
        assert "Recorded added system instruction" in html
        assert view.technical["decision"] == comparison["decision"]
    assert snapshot.files == original


def test_recorded_workflow_and_dataset_are_complete_row_facts_not_case_inference():
    runs = inputs()
    for run in runs.values():
        for row in run["records"]:
            row["metadata"] = {"dataset": "public-dialogues", "workflow": "routing"}
    _, facts, _, _ = record_reporting._captured_context(runs)
    assert dict(facts)["Baseline dataset"] == "public-dialogues"
    assert dict(facts)["Subject workflow"] == "routing"
    assert dict(facts)["Subject records"] == "3"
    runs["subject"]["records"][-1]["metadata"]["dataset"] = "other-dataset"
    _, facts, _, _ = record_reporting._captured_context(runs)
    assert dict(facts)["Subject dataset"] == "Mixed or incomplete across records"


@pytest.mark.parametrize(
    ("example", "basis"),
    [
        ("classification", "expected and output values, scored"),
        ("judge", "external measurements or judgments, aggregated"),
    ],
)
def test_scoring_notes_describe_stored_comparison_not_report_execution(example, basis):
    baseline, subject, policy = example_project(example)
    policy["metrics"] = policy["metrics"][:1]
    snapshot = build_pack(baseline, subject, policy)
    manifest, payloads, signer = load_payloads(snapshot)
    for view in (
        record_reporting._view(pack_json(snapshot, "report"), snapshot),
        captured_reporting._view(manifest, payloads, signer),
    ):
        for metric in view.metrics:
            assert (
                f"Recorded scoring basis: {basis} when the comparison was created."
                in metric.notes
            )
            assert "Scoring and replay were not performed by report." in metric.notes
