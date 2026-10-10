"""A known source error gets linked replacement evidence without rewriting history."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from invarlock.engine import verify_signed_verification_receipt

SCRIPT = Path(__file__).resolve().parents[2] / "examples/hosted-service/reassessment.py"


def module():
    spec = importlib.util.spec_from_file_location("correction_fixture", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_correction_retains_error_and_authenticates_replacement(tmp_path):
    helper = module()
    root = tmp_path / "correction"
    result = helper.rehearse_correction(root)
    assert helper.snapshot(root / "a") == result["historical_files_sha256"]
    assert result["assessments"]["a"]["decision"] == "pass"
    assert result["assessments"]["replacement"]["decision"] == "regression"
    assert result["correction"]["requires_review"] == [
        "deployment-approval",
        "release-review",
    ]
    assert result["correction"]["unassessed_consumers"] == "unknown"
    assert "unrelated-review" not in result["correction"]["requires_review"]
    first = result["assessments"]["a"]
    replacement = result["assessments"]["replacement"]
    assert first["observation_window"] == replacement["observation_window"]
    assert first["receipt_sha256"] != replacement["receipt_sha256"]
    raw = json.loads((root / "source-export.json").read_text())
    old = json.loads((root / "a/subject.json").read_text())["records"]
    corrected = json.loads((root / "replacement/subject.json").read_text())["records"]
    assert all(r["output"] == "no" for r in raw)
    assert all(r["output"] == "yes" for r in old)
    assert all(r["output"] == "no" for r in corrected)
    for name, assessment in result["assessments"].items():
        anchors = assessment["anchors"]
        checked = verify_signed_verification_receipt(
            root / name / "verification.receipt.json",
            root / name / "evidence",
            policy_path=root / name / "comparison-policy.json",
            expected_run_digests={
                role: anchors[f"{role}_run_digest"] for role in ("baseline", "subject")
            },
            expected_request_digest=anchors["request_digest"],
            expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
            expected_verifier_identity="reassessment-example-recipient",
            expected_verifier_fingerprint=result["verifier_fingerprint"],
        )
        assert checked.ok, checked.errors
    with pytest.raises(FileExistsError):
        helper.rehearse_correction(root)


def test_dependency_impact_follows_transitive_edges_only():
    helper = module()
    graph = {"a": [], "b": ["a"], "c": ["b"], "d": []}
    assert helper.correction_impact(graph, "a") == ["b", "c"]
    assert helper.correction_impact(graph, "d") == []
    with pytest.raises(ValueError, match="unknown"):
        helper.correction_impact(graph, "missing")
    with pytest.raises(ValueError, match="unknown"):
        helper.correction_impact({"a": ["absent"]}, "a")
    with pytest.raises(ValueError, match="cycle"):
        helper.correction_impact({"a": ["b"], "b": ["a"]}, "a")


def test_correction_detects_historical_mutation(tmp_path, monkeypatch):
    helper = module()
    original = helper.assess

    def alter(directory, **kwargs):
        result = original(directory, **kwargs)
        if directory.name == "replacement":
            (directory.parent / "a/verification.json").write_text("changed")
        return result

    monkeypatch.setattr(helper, "assess", alter)
    with pytest.raises(ValueError, match="historical"):
        helper.rehearse_correction(tmp_path / "mutation")


def test_replacement_must_match_retained_source(tmp_path, monkeypatch):
    helper = module()
    original = helper.captured_run

    def wrong(name, day, answer):
        return original(name, day, "yes" if name == "corrected-import" else answer)

    monkeypatch.setattr(helper, "captured_run", wrong)
    root = tmp_path / "wrong-source"
    with pytest.raises(ValueError, match="replacement does not match"):
        helper.rehearse_correction(root)
    assert not (root / "correction.json").exists()


def test_misconfigured_policy_cannot_claim_demonstrated_correction(
    tmp_path, monkeypatch
):
    helper = module()
    policy = helper.example_policy()
    policy["metrics"][0]["maximum_regression"] = 1.0
    monkeypatch.setattr(helper, "example_policy", lambda: policy)
    root = tmp_path / "wrong-policy"
    with pytest.raises(ValueError, match="unexpected correction decisions"):
        helper.rehearse_correction(root)
    assert not (root / "correction.json").exists()
    assert (
        json.loads((root / "replacement/verification.json").read_text())["decision"]
        == "pass"
    )


def test_correction_command_runs_and_records_known_impact(
    tmp_path, monkeypatch, capsys
):
    import runpy
    import sys

    root = tmp_path / "command"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--scenario", "correction", "--output", str(root)]
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    output = json.loads(capsys.readouterr().out)
    assert output["original"] == "pass"
    assert output["replacement"] == "regression"
    assert output["requires_review"] == ["deployment-approval", "release-review"]
    assert (root / "correction.json").is_file()
