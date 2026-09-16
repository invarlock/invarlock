"""Focused coverage for the captured evaluation transaction."""

import json
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock import captured_evaluation, captured_evidence_publication
from invarlock.captured_evaluation import (
    CapturedEvaluationError,
    evaluate_captured_request,
    preflight_captured_request,
)
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes


def _request(tmp_path, baseline, subject, policy, *, pin=False):
    (tmp_path / "baseline.json").write_bytes(canonical_json_bytes(baseline))
    (tmp_path / "subject.json").write_bytes(canonical_json_bytes(subject))
    (tmp_path / "policy.json").write_bytes(canonical_json_bytes(policy))
    baseline_pin = f"\n    expected_run_digest: {digest(baseline)}" if pin else ""
    subject_pin = f"\n    expected_run_digest: {digest(subject)}" if pin else ""
    request = tmp_path / "request.yaml"
    request.write_text(
        "format_version: invarlock/evaluation-request-v2\n"
        "execution:\n  mode: captured\n"
        "comparison:\n"
        "  baseline:\n"
        "    path: baseline.json\n"
        "    adapter: invarlock"
        f"{baseline_pin}\n"
        "  subject:\n"
        "    path: subject.json\n"
        "    adapter: invarlock"
        f"{subject_pin}\n"
        "  policy: policy.json\n"
        "output:\n  evidence: evidence\n",
        encoding="utf-8",
    )
    return load_evaluation_request(request)


def _key(tmp_path):
    path = tmp_path / "signing-key.pem"
    path.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return path


def test_signed_and_unsigned_captured_evaluation_publish_fixed_inventory(tmp_path):
    baseline, subject, policy = example_project("classification")
    signed_root = tmp_path / "signed"
    signed_root.mkdir()
    signed = _request(signed_root, baseline, subject, policy)
    signed_result = evaluate_captured_request(
        signed, signing_key_path=_key(signed.root)
    )
    assert signed_result.authentication == "signed"
    assert (signed.evidence / "manifest.signature.json").is_file()
    assert (signed.evidence / "manifest.json").is_file()
    assert (signed.evidence / "checksums.sha256").is_file()

    unsigned_root = tmp_path / "unsigned"
    unsigned_root.mkdir()
    unsigned = _request(unsigned_root, baseline, subject, policy)
    unsigned_result = evaluate_captured_request(unsigned, unsigned=True)
    assert unsigned_result.authentication == "unsigned_local"
    assert not (unsigned.evidence / "signature.json").exists()


def test_captured_run_pin_mismatch_rejects_before_publication(tmp_path):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / "pinned"
    root.mkdir()
    request = _request(root, baseline, subject, policy, pin=True)
    (root / "subject.json").write_bytes(
        canonical_json_bytes({**subject, "run_id": "replaced"})
    )
    with pytest.raises(CapturedEvaluationError, match="subject run digest"):
        evaluate_captured_request(request, unsigned=True)
    assert not request.evidence.exists()


@pytest.mark.parametrize(
    "unsigned, expected", [(False, "signed"), (True, "unsigned_local")]
)
def test_captured_preflight_is_structural_and_does_not_publish(
    tmp_path, monkeypatch, unsigned, expected
):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / expected
    root.mkdir()
    request = _request(root, baseline, subject, policy)
    signing_key = None if unsigned else _key(root)
    validate_key = Mock(wraps=captured_evidence_publication._private_key)
    monkeypatch.setattr(captured_evidence_publication, "_private_key", validate_key)
    forbidden = Mock(
        side_effect=AssertionError("preflight must not score, sign or publish")
    )
    monkeypatch.setattr(captured_evaluation, "compare_runs", forbidden)
    monkeypatch.setattr(captured_evaluation, "publish_captured_evidence", forbidden)
    monkeypatch.setattr(captured_evidence_publication, "manifest_signature", forbidden)

    result = preflight_captured_request(
        request, signing_key_path=signing_key, unsigned=unsigned
    )

    forbidden.assert_not_called()
    if unsigned:
        validate_key.assert_not_called()
    else:
        validate_key.assert_called_once_with(signing_key)
    assert result.requested_authentication == expected
    assert result.execution_mode == "captured"
    assert result.record_count == len(baseline["records"])
    assert result.required_bootstrap_draws > 0
    assert not request.evidence.exists()
    payload = json.loads(result.as_json())
    assert payload["format_version"] == "invarlock/evaluation-preflight-v3"
    assert "decision" not in payload
    assert "policy_verdict" not in payload


def test_captured_publication_does_not_clobber_existing_output(tmp_path):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / "collision"
    root.mkdir()
    request = _request(root, baseline, subject, policy)
    request.evidence.mkdir()
    marker = request.evidence / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(CapturedEvaluationError, match="already exists"):
        evaluate_captured_request(request, unsigned=True)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "100"])
def test_captured_evaluation_budget_precedes_scoring_and_publication(
    tmp_path, monkeypatch, budget
):
    from invarlock.evaluation_comparison import comparison

    baseline, subject, policy = example_project("judge")
    request = _request(tmp_path, baseline, subject, policy)

    def forbidden(*args, **kwargs):
        pytest.fail("budget refusal must precede scoring")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(CapturedEvaluationError, match="budget|non-negative integer"):
        evaluate_captured_request(request, unsigned=True, max_bootstrap_draws=budget)
    assert not request.evidence.exists()


def test_captured_preflight_allows_explicit_unlimited_work(tmp_path):
    baseline, subject, policy = example_project("judge")
    request = _request(tmp_path, baseline, subject, policy)
    result = preflight_captured_request(
        request, unsigned=True, max_bootstrap_draws=None
    )
    assert result.required_bootstrap_draws == 2048 * 2 * (40 + 20)
    assert not request.evidence.exists()


def test_captured_preflight_checks_planned_membership(tmp_path):
    from invarlock.evaluation_records.cases import case_set_digest

    baseline, subject, policy = example_project("judge")
    cases = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {key: row[key] for key in ("id", "input", "expected", "metadata")}
            for row in baseline["records"]
        ],
    }
    policy["expected_case_set_digest"] = case_set_digest(cases)
    baseline["records"].pop()
    subject["records"].pop()
    request = _request(tmp_path, baseline, subject, policy)
    with pytest.raises(CapturedEvaluationError, match="planned case set"):
        preflight_captured_request(request, unsigned=True)
    assert not request.evidence.exists()
