"""Signed CLI replay accounts for unavailable answers without dropping rows."""

import json

import pytest
from typer.testing import CliRunner

from invarlock import engine
from invarlock.cli.app import app
from tests.core.test_captured_sdk_omissions import _bytes, _digest, _key, _profile
from tests.evaluation_comparison.test_answer_availability import policy, run

RUNNER = CliRunner()


def inputs(root, output, error, selector):
    baseline = run(output, error=error)
    subject = {**baseline, "run_id": "subject"}
    selected = policy()
    selected["slices"] = [{"name": "first", "where": {"group": "0"}}]
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "path": "baseline.json",
                "adapter": "invarlock",
                "expected_run_digest": _digest(baseline),
            },
            "subject": {
                "path": "subject.json",
                "adapter": "invarlock",
                "expected_run_digest": _digest(subject),
            },
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    if selector:
        request["comparison"]["metric"] = selector
    for name, value in (
        ("request", request),
        ("baseline", baseline),
        ("subject", subject),
        ("policy", selected),
    ):
        (root / f"{name}.json").write_bytes(_bytes(value))
    signer = _key(root / "signer.pem")
    # Recipient pins come from authored inputs before any evidence is produced.
    profile, verifier = _profile(root, request, baseline, subject, selected, signer)
    return baseline, subject, profile, verifier


def evaluate(root):
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "request.json"),
            "--signing-key",
            str(root / "signer.pem"),
            "--json",
        ],
    )
    (root / "evaluate-output.json").write_text(result.stdout)
    return result


def verify(root, profile, verifier):
    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(root / "evidence"),
            "--trust-profile",
            str(root / "trust.json"),
            "--receipt",
            str(root / "receipt.json"),
            "--json",
        ],
    )
    (root / "verify-output.json").write_text(result.stdout)
    trust = engine.load_trust_inputs(root / "trust.json")
    authenticated = engine.verify_signed_verification_receipt(
        root / "receipt.json",
        root / "evidence",
        policy_path=root / "policy.json",
        expected_run_digests=dict(trust.expected_run_digests),
        expected_request_digest=trust.expected_request_digest,
        expected_pack_signer_fingerprint=profile["anchors"][
            "evidence_signer_fingerprint"
        ],
        expected_verifier_identity="test-verifier",
        expected_verifier_fingerprint=verifier,
        expected_trust_profile_digest=trust.profile_digest,
    )
    assert authenticated.ok, authenticated.errors
    receipt = json.loads((root / "receipt.json").read_text())["statement"]
    assert receipt["format"] == "invarlock/evidence-verification-receipt-v3"
    return result, receipt


@pytest.mark.parametrize("selector", [None, "exact_match"])
@pytest.mark.parametrize(
    "output,error,missing",
    [
        (None, None, True),
        ("", None, False),
        ("wrong", None, False),
        ("answer", "timeout", True),
    ],
)
def test_signed_cli_missingness_and_zero_scores(
    tmp_path, selector, output, error, missing
):
    baseline, subject, profile, verifier = inputs(tmp_path, output, error, selector)
    evaluated = evaluate(tmp_path)
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.stdout)["decision"] == (
        "insufficient_evidence" if missing else "pass"
    )
    verified, receipt = verify(tmp_path, profile, verifier)
    assert verified.exit_code == (7 if missing else 0), verified.output
    result = json.loads(verified.stdout)
    decision = "insufficient_evidence" if missing else "pass"
    assert result["decision"] == decision
    assert result["integrity_ok"] is True
    assert result["replay_status"] == "completed"
    assert receipt["verdict"]["decision"] == decision
    assert receipt["verdict"]["ok"] is (not missing)
    for summary, count in zip(result["metric_summaries"], [2, 1], strict=True):
        assert summary["usable_count"] == (0 if missing else count)
        assert summary["missing_count"] == (count if missing else 0)
    for name, captured in (("baseline", baseline), ("subject", subject)):
        retained = json.loads((tmp_path / f"evidence/records/{name}.json").read_text())
        assert retained == captured
    comparison = json.loads(
        (tmp_path / "evidence/reports/evaluation.report.json").read_text()
    )
    for metric in comparison["metrics"]:
        assert metric["subject_mean"] == (None if missing else 0)


def test_verifier_rejects_old_zero_scored_null_comparison(tmp_path, monkeypatch):
    _, _, profile, verifier = inputs(tmp_path, None, None, None)
    # Recreate the previous comparison rule while retaining genuine signed bytes.
    # Restore current replay before the recipient reads the immutable pack.
    with monkeypatch.context() as legacy:
        legacy.setattr(
            "invarlock.evaluation_comparison.comparison.missing_pair",
            lambda left, right, metric: (
                left["error"] is not None or right["error"] is not None
            ),
        )
        evaluated = evaluate(tmp_path)
        assert evaluated.exit_code == 0, evaluated.output
    report = tmp_path / "evidence/reports/evaluation.report.json"
    original = report.read_bytes()
    verified, receipt = verify(tmp_path, profile, verifier)
    assert verified.exit_code == 6, verified.output
    assert "replay disagrees" in verified.stdout
    assert receipt["verdict"]["ok"] is False
    assert receipt["verdict"]["integrity_ok"] is False
    assert report.read_bytes() == original
