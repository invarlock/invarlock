"""Hosted reports identify retained service declarations without weight claims."""

import copy
import json

import pytest

from invarlock import captured_reporting, record_reporting
from invarlock.captured_contracts import load_payloads
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evaluation_records.templates import example_project
from invarlock.report_presentation import render_html, render_markdown
from tests._evaluation_support import build_pack, pack_json


def service_identity(**changes):
    configuration = {"temperature": 0, "instructions": "<script>unsafe()</script>"}
    return {
        "kind": "hosted_service",
        "provider": "example-provider",
        "service": "text-service",
        "deployment": "production",
        "requested_model": "example-8b",
        "observed_model": None,
        "exposed_revision": None,
        "configuration": configuration,
        "configuration_digest": digest(configuration),
        "harness": {
            "name": "existing-harness",
            "version": "1",
            "source_digest": digest({"code": 1}),
        },
        "observation_window": {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
        },
        **changes,
    }


def runs():
    baseline, subject, policy = example_project("classification")
    for run in (baseline, subject):
        run["artifact_digest"] = None
        run["service_identity"] = service_identity()
    subject["service_identity"]["observation_window"] = {
        "started_at": "2026-01-02T00:00:00Z",
        "ended_at": "2026-01-02T00:01:00Z",
    }
    return baseline, subject, policy


def test_hosted_report_forms_show_windows_configuration_and_unknown_weights():
    baseline, subject, policy = runs()
    snapshot = build_pack(baseline, subject, policy)
    original = dict(snapshot.files)
    manifest, payloads, signer = load_payloads(snapshot)
    for view in (
        record_reporting._view(pack_json(snapshot, "report"), snapshot),
        captured_reporting._view(manifest, payloads, signer),
    ):
        assert "example-8b" in str(view.subjects)
        assert "service identity" in str(view.identity)
        assert "attributed artifact" not in str(view.identity)
        for rendered in (
            render_html(view),
            render_markdown(view, include_details=True),
        ):
            assert "2026-01-02T00:00:00Z" in rendered
            assert "Model weights are not identified" in rendered
            assert "does not remeasure" in rendered
            assert "example-provider" in rendered
            assert "temperature" in rendered
        assert "<script>unsafe()" not in render_html(view)
        assert "unchanged" in " ".join(view.changes)
    assert snapshot.files == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment", "canary"),
        ("observed_model", "resolved-8b"),
        ("exposed_revision", "revision-2"),
        ("requested_model", "other-8b"),
        ("configuration", {"temperature": 1}),
        (
            "harness",
            {"name": "other", "version": "1", "source_digest": "sha256:" + "1" * 64},
        ),
    ],
)
def test_reports_identify_declared_changes_without_assigning_a_cause(field, value):
    baseline, subject, _ = runs()
    subject["service_identity"][field] = value
    if field == "configuration":
        subject["service_identity"]["configuration_digest"] = digest(value)
    before = copy.deepcopy((baseline, subject))
    _, context, changes, _ = record_reporting._captured_context(
        {"baseline": baseline, "subject": subject}
    )
    assert field.replace("_", " ") in " ".join(changes)
    assert "does not identify the cause" in " ".join(changes)
    assert (baseline, subject) == before
    if field == "observed_model":
        assert "resolved-8b" in str(context)


def test_mixed_artifact_and_service_do_not_share_a_weight_identity():
    baseline, subject, _ = runs()
    baseline.pop("service_identity")
    baseline["artifact_digest"] = "sha256:" + "a" * 64
    material = {"baseline": baseline, "subject": subject}
    assert "different identity profiles" in " ".join(
        record_reporting._captured_context(material)[2]
    )
    identities = dict(record_reporting._captured_identities(material))
    assert identities["Baseline attributed artifact"] == baseline["artifact_digest"]
    assert identities["Subject service identity"] == digest(subject["service_identity"])


def test_hosted_configuration_preview_is_bounded_and_not_a_policy_claim():
    baseline, subject, _ = runs()
    subject["service_identity"]["configuration"] = {"large": "x" * 50000}
    subject["service_identity"]["configuration_digest"] = digest(
        subject["service_identity"]["configuration"]
    )
    _, _, _, details = record_reporting._captured_context(
        {"baseline": baseline, "subject": subject}
    )
    text = json.dumps(details)
    assert "x" * 1000 not in text
    assert "policy binding" not in text
    assert "captured run" in text


def test_same_alias_distinct_observations_are_visible_without_empty_artifact_fields():
    baseline, subject, _ = runs()
    for run, model in ((baseline, "model-a"), (subject, "model-b")):
        run["service_identity"]["observed_model"] = model
        run["source"]["name"] = "existing-harness"
        run["source"]["version"] = "1"
    subjects, context, _, _ = record_reporting._captured_context(
        {"baseline": baseline, "subject": subject}
    )
    assert (
        dict(subjects)["Baseline"] == "Observed model: model-a · Deployment: production"
    )
    assert (
        dict(subjects)["Subject"] == "Observed model: model-b · Deployment: production"
    )
    assert "Unavailable in recorded context" not in str(context)
    assert "Baseline evaluator" not in dict(context)
    assert dict(context)["Baseline service harness"] == "existing-harness 1"
    assert dict(context)["Baseline exposed revision"] == "Not exposed"


def test_hosted_context_retains_distinct_evaluator_and_mixed_recorded_metadata():
    baseline, subject, _ = runs()
    baseline["records"][0]["metadata"] = {"dataset": "tasks-a"}
    baseline["records"][1]["metadata"] = {"dataset": "tasks-b"}
    subjects, context, _, _ = record_reporting._captured_context(
        {"baseline": baseline, "subject": subject}
    )
    assert dict(subjects)["Baseline"].startswith("Requested model: example-8b")
    assert "Baseline evaluator" in dict(context)
    assert dict(context)["Baseline dataset"] == "Mixed or incomplete across records"
