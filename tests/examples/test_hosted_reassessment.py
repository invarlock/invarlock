"""Two bounded assessments retain distinct signed results and original bytes."""

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

from invarlock.engine import (
    EvidenceVerificationError,
    verify_signed_verification_receipt,
)

SCRIPT = Path(__file__).resolve().parents[2] / "examples/hosted-service/reassessment.py"


def module():
    spec = importlib.util.spec_from_file_location("hosted_reassessment", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_later_assessment_preserves_original_bytes_and_authenticated_verdict(tmp_path):
    helper = module()
    root = tmp_path / "reassessment"
    result = helper.rehearse(root)
    first, later = (result["assessments"][name] for name in ("a", "b"))
    assert first["decision"] == "pass"
    assert later["decision"] == "regression"
    assert (
        first["anchors"]["baseline_run_digest"]
        == later["anchors"]["baseline_run_digest"]
    )
    for field in ("subject_run_digest", "request_digest"):
        assert first["anchors"][field] != later["anchors"][field]
    assert first["receipt_sha256"] != later["receipt_sha256"]
    assert first["observation_window"]["ended_at"] < result["trigger"]["declared_at"]
    assert result["trigger"]["declared_at"] < later["observation_window"]["started_at"]
    assert helper.snapshot(root / "a") == result["historical_files_sha256"]
    for name, assessment in result["assessments"].items():
        directory = root / name
        anchors = assessment["anchors"]
        verified = verify_signed_verification_receipt(
            directory / "verification.receipt.json",
            directory / "evidence",
            policy_path=directory / "comparison-policy.json",
            expected_run_digests={
                role: anchors[f"{role}_run_digest"] for role in ("baseline", "subject")
            },
            expected_request_digest=anchors["request_digest"],
            expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
            expected_verifier_identity="reassessment-example-recipient",
            expected_verifier_fingerprint=result["verifier_fingerprint"],
        )
        assert verified.ok, verified.errors
        receipt = json.loads((directory / "verification.receipt.json").read_bytes())
        assert receipt["statement"]["verdict"]["decision"] == assessment["decision"]
        assert receipt["statement"]["verification_scope"] == "captured_comparison"
    assert result["qualification"] == "synthetic_integration_fixture"


def test_rehearsal_refuses_to_replace_an_existing_campaign(tmp_path):
    helper = module()
    root = tmp_path / "reassessment"
    helper.rehearse(root)
    before = helper.snapshot(root)
    with pytest.raises(FileExistsError):
        helper.rehearse(root)
    assert helper.snapshot(root) == before


def test_rehearsal_detects_changes_to_the_first_assessment(tmp_path, monkeypatch):
    helper = module()
    original = helper.assess

    def change_history(directory, **kwargs):
        result = original(directory, **kwargs)
        if directory.name == "b":
            receipt = directory.parent / "a/verification.receipt.json"
            receipt.chmod(0o600)
            receipt.write_bytes(receipt.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(helper, "assess", change_history)
    with pytest.raises(ValueError, match="historical assessment changed"):
        helper.rehearse(tmp_path / "reassessment")


def test_documented_command_runs_offline_and_reports_both_results(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "command"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(root)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["a"] == "pass" and result["b"] == "regression"
    assert result["historical_bytes_unchanged"] is True
    assert (root / "scenario.json").is_file()


def test_incomplete_verification_does_not_publish_a_successful_scenario(
    tmp_path, monkeypatch
):
    helper = module()

    def incomplete(*args, **kwargs):
        raise EvidenceVerificationError("verification unavailable", exit_code=2)

    monkeypatch.setattr(helper, "verify_evidence", incomplete)
    root = tmp_path / "incomplete"
    with pytest.raises(EvidenceVerificationError, match="verification unavailable"):
        helper.rehearse(root)
    assert not (root / "scenario.json").exists()
    assert not (root / "b").exists()
