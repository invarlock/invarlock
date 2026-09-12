from __future__ import annotations

import base64
from xml.etree.ElementTree import fromstring

import pytest

from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES
from invarlock.judge_measurements import acceptance, evidence
from invarlock.judge_measurements.analysis import ANALYSIS_POLICY_MAX_BYTES
from invarlock.judge_measurements.contracts import (
    MEASUREMENTS_MAX_BYTES,
    PLAN_MAX_BYTES,
)
from invarlock.judge_measurements.reporting import render_judge_evidence
from tests.judge_measurements.test_evidence_acceptance import _json, _publish, _write


@pytest.mark.parametrize(
    "failure", ["unsigned", "signer", "signature", "subject", "plan", "scope"]
)
def test_untrusted_or_unpinned_envelopes_do_not_start_artifact_replay(
    tmp_path, monkeypatch, failure
):
    publication, policy_path = _publish(tmp_path, signed=failure != "unsigned")
    policy = _json(policy_path)
    if failure == "signer":
        policy["trusted_signer"]["identity"] = "unapproved-signer"
    elif failure == "subject":
        policy["intended_subject"] = "sha256:" + "0" * 64
    elif failure == "plan":
        policy["bindings"]["plan_sha256"] = "0" * 64
    elif failure == "scope":
        policy["decision_scope"] = "native-inference-v1"
    _write(policy_path, policy)
    if failure == "signature":
        envelope = _json(publication.path / "envelope.json")
        envelope["signature"] = base64.b64encode(b"x" * 64).decode()
        _write(publication.path / "envelope.json", envelope)
    # Artifact parsing would fail if it were reached before the cheap rejection.
    (publication.path / "measurements.json").write_text("not JSON")
    monkeypatch.setattr(
        acceptance,
        "replay_judge_evidence",
        lambda *args, **kwargs: pytest.fail(
            "unapproved envelope triggered expensive artifact replay"
        ),
    )
    result = acceptance.verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not result.accepted and not result.replayed
    assert result.authenticated is (failure in {"subject", "plan"})


def test_envelope_cannot_change_between_authentication_and_replay(
    tmp_path, monkeypatch
):
    publication, policy_path = _publish(tmp_path)
    replay = evidence.replay_judge_evidence

    def substitute(path, *, expected_envelope_sha256):
        assert expected_envelope_sha256 == evidence.object_sha256(publication.envelope)
        changed = _json(path / "envelope.json")
        changed["signature"] = base64.b64encode(b"x" * 64).decode()
        _write(path / "envelope.json", changed)
        (path / "measurements.json").write_text("not JSON")
        return replay(path, expected_envelope_sha256=expected_envelope_sha256)

    monkeypatch.setattr(acceptance, "replay_judge_evidence", substitute)
    receipt = acceptance.verify_judge_evidence_with_policy(
        publication.path, policy_path
    )
    assert receipt.authenticated and not receipt.replayed and not receipt.accepted
    assert "envelope changed before replay" in receipt.errors[0]


@pytest.mark.parametrize(
    "filename,native_limit",
    [
        ("plan.json", PLAN_MAX_BYTES),
        ("measurements.json", MEASUREMENTS_MAX_BYTES),
        ("baseline_run.json", MAX_INPUT_BYTES),
        ("subject_run.json", MAX_INPUT_BYTES),
        ("case_set.json", MAX_INPUT_BYTES),
        ("analysis_policy.json", ANALYSIS_POLICY_MAX_BYTES),
        ("analysis_result.json", 1024 * 1024),
    ],
)
def test_replay_rejects_each_artifact_at_its_native_read_limit(
    tmp_path, filename, native_limit
):
    publication, policy_path = _publish(tmp_path)
    # A sparse file exercises the pre-read size check without allocating its contents.
    with (publication.path / filename).open("wb") as stream:
        stream.truncate(native_limit + 1)
    receipt = acceptance.verify_judge_evidence_with_policy(
        publication.path, policy_path
    )
    assert receipt.authenticated and not receipt.replayed and not receipt.accepted
    assert filename in receipt.errors[0]
    assert f"{native_limit}-byte size limit" in receipt.errors[0]


@pytest.mark.parametrize("role", ["required", "advisory"])
@pytest.mark.parametrize("outcome", ["pass", "regression", "insufficient_evidence"])
def test_reports_preserve_decision_and_role_without_gating_advisory_metrics(
    tmp_path, role, outcome
):
    publication, _ = _publish(
        tmp_path,
        role=role,
        baseline=1 if outcome == "regression" else 0,
        subject=0 if outcome == "regression" else 1,
        incomplete=outcome == "insufficient_evidence",
    )
    html_path = tmp_path / "report.html"
    junit_path = tmp_path / "report.xml"
    result = render_judge_evidence(
        publication.path, html_path=html_path, junit_path=junit_path
    )
    assert result.facts["assurance"]["policy_decision"] == outcome
    assert result.facts["assurance"]["decision_role"] == role
    assert f"Decision role: {role}" in result.text
    assert "16 cases; 16 independent units;" in result.text
    assert "completed trials" in result.text
    assert f"Decision role: {role}" in html_path.read_text()
    suite = fromstring(junit_path.read_bytes())
    assert suite.find("properties/property[@name='decision_role']").get("value") == role
    assert (
        suite.find("properties/property[@name='policy_decision']").get("value")
        == outcome
    )
    case = suite.find("testcase")
    if role == "advisory":
        assert suite.get("failures") == suite.get("errors") == "0"
        assert suite.get("skipped") == "1" and case.find("skipped") is not None
        assert case.find("failure") is None and case.find("error") is None
        assert "does not gate required decisions" in result.text
    else:
        assert suite.get("failures") == str(int(outcome == "regression"))
        assert suite.get("errors") == str(int(outcome == "insufficient_evidence"))
        assert suite.get("skipped") == "0" and case.find("skipped") is None
    if outcome == "regression":
        assert f"At least one declared {role} bound is violated." in result.text
        assert "bounds are satisfied" not in result.text
        if role == "required":
            assert "bound is violated" in case.find("failure").get("message")


@pytest.mark.parametrize("signed", [True, False])
def test_reports_show_descriptive_baseline_and_correct_signing_next_step(
    tmp_path, signed
):
    publication, _ = _publish(tmp_path, baseline=1, subject=0, signed=signed)
    result = render_judge_evidence(publication.path)
    assert "| 1.000000000000000 | 0" in result.text
    assert "equal independent-unit weights" in result.text
    assert result.facts["descriptive_means"]["baseline"] == "1.000000000000000"
    signing_step = "republish the same retained inputs"
    assert (signing_step in result.text) is not signed
    if not signed:
        assert result.text.index(signing_step) < result.text.index("Use verify")
        assert "new evidence destination" in result.text


def test_baseline_display_uses_equal_unit_weights_and_fixed_rounding():
    from decimal import localcontext

    from invarlock.judge_measurements.reporting import _baseline_mean

    plan = {
        "sampling": {
            "case_units": [
                {"case_id": "a", "unit_id": "one"},
                {"case_id": "b", "unit_id": "two"},
                {"case_id": "c", "unit_id": "two"},
            ]
        }
    }
    measurements = {
        "trials": [
            {"case_id": case, "side": "baseline", "parse": {"value": score}}
            for case, score in (
                ("a", "1"),
                ("a", "0"),
                ("b", "0"),
                ("b", "0"),
                ("c", "0"),
                ("c", "0"),
            )
        ]
    }
    with localcontext() as context:
        context.prec = 2
        assert _baseline_mean(plan, measurements) == "0.250000000000000"


def test_incomplete_report_does_not_infer_a_baseline_from_complete_cases(tmp_path):
    publication, _ = _publish(tmp_path, incomplete=True)
    result = render_judge_evidence(publication.path)
    assert "| Unavailable | Unavailable | Unavailable |" in result.text
    assert result.facts["descriptive_means"]["baseline"] is None


def test_report_can_select_any_retained_case_without_expanding_default_details(
    tmp_path,
):
    publication, _ = _publish(tmp_path)
    html = tmp_path / "selected.html"
    result = render_judge_evidence(
        publication.path, html_path=html, case_ids=("case-15",)
    )
    assert result.facts["detail_limits"] == {
        "shown_cases": 1,
        "total_cases": 16,
        "text_excerpt_characters": 2000,
        "selection": "requested",
        "case_ids": ["case-15"],
    }
    rendered = html.read_text()
    assert "<summary>case-15</summary>" in rendered
    assert "<summary>case-0</summary>" not in rendered


@pytest.mark.parametrize(
    "case_ids,message",
    [
        (("case-0", "case-0"), "must be unique"),
        (("not-retained",), "not in the judge plan"),
        (tuple(f"case-{index}" for index in range(51)), "at most 50"),
    ],
)
def test_report_rejects_ambiguous_or_unbounded_case_selection(
    tmp_path, case_ids, message
):
    publication, _ = _publish(tmp_path)
    with pytest.raises(ValueError, match=message):
        render_judge_evidence(publication.path, case_ids=case_ids)
