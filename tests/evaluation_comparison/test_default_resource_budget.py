"""Omitted local settings must bound a complete, highly overlapping policy."""

import copy
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.cli import app
from invarlock.evidence_verification import EvidenceVerificationError
from tests._evaluation_support import (
    EvaluationRecordsError,
    build_pack,
    captured_request_digest,
    comparison,
    digest,
    example_project,
    materialize_captured_request,
    pack_json,
    replay_pack,
)


def test_default_supports_one_scalar_over_the_complete_50000_case_schedule():
    baseline, candidate, policy = example_project("judge")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for run in (baseline, candidate):
        original = run["records"][0]
        run["records"] = [
            {**copy.deepcopy(original), "id": f"case-{i}"} for i in range(50000)
        ]
    key = Ed25519PrivateKey.generate()
    evidence = build_pack(baseline, candidate, policy, key)
    replay = replay_pack(
        evidence,
        public_key=key.public_key(),
        expected_baseline_run=digest(baseline),
        expected_subject_run=digest(candidate),
        policy=policy,
    )
    assert replay == pack_json(evidence, "report")
    assert replay["metrics"][0]["count"] == 50000
    assert replay["metrics"][0]["interval"]["replicates"] == 2048
    with pytest.raises(EvaluationRecordsError, match="102400000 bootstrap draws"):
        comparison.compare_runs(
            baseline, candidate, policy, max_bootstrap_draws=102399999
        )


def overlapping_project():
    baseline, candidate, policy = example_project("judge")
    # Just 184 pairs can exceed the measured work envelope when every one of
    # sixteen scalar metrics applies to all seventeen overlapping scopes.
    for run in (baseline, candidate):
        original = run["records"][0]
        run["records"] = []
        for i in range(184):
            row = copy.deepcopy(original)
            row["id"] = f"case-{i}"
            row["metadata"]["included"] = "yes"
            run["records"].append(row)
    policy["metrics"] = [
        {**copy.deepcopy(policy["metrics"][0]), "name": f"quality-{i}"}
        for i in range(16)
    ]
    policy["slices"] = [
        {"name": f"slice-{i}", "where": {"included": "yes"}} for i in range(16)
    ]
    return baseline, candidate, policy


@pytest.mark.parametrize("check", [comparison.compare_runs, build_pack])
def test_omitted_budget_rejects_before_scoring(check, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the default must reject before computing a partial verdict")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvaluationRecordsError, match="local budget is 102400000"):
        check(*overlapping_project())


def test_explicit_none_preserves_complete_replay_but_not_recipient_default(monkeypatch):
    baseline, candidate, policy = overlapping_project()
    key = Ed25519PrivateKey.generate()
    evidence = build_pack(baseline, candidate, policy, key, max_bootstrap_draws=None)
    options = {
        "public_key": key.public_key(),
        "expected_baseline_run": digest(baseline),
        "expected_subject_run": digest(candidate),
        "policy": policy,
    }
    replay = replay_pack(evidence, **options, max_bootstrap_draws=None)
    assert replay == pack_json(evidence, "report")
    assert len(replay["metrics"]) == 272
    assert all(m["count"] == 184 for m in replay["metrics"])

    def forbidden(*args, **kwargs):
        pytest.fail("an evidence author's override cannot raise the recipient budget")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvidenceVerificationError, match="local_work_budget_exceeded"):
        replay_pack(evidence, **options)


def test_cli_omitted_budget_rejects_without_publishing(tmp_path):
    baseline, candidate, policy = overlapping_project()
    runner = CliRunner()
    root = tmp_path / "project"
    request = materialize_captured_request(root, baseline, candidate, policy)
    result = runner.invoke(app, ["evaluate", str(request), "--unsigned", "--json"])
    assert result.exit_code == 2
    assert "local budget is 102400000" in " ".join(
        json.loads(result.stdout).get("errors", [])
    )
    assert not (root / "evidence").exists()


def test_cli_recipient_requires_its_own_larger_budget(tmp_path):
    baseline, candidate, policy = example_project("judge")
    root = tmp_path / "project"
    request = materialize_captured_request(root, baseline, candidate, policy)
    signer = Ed25519PrivateKey.generate()
    signer_path = tmp_path / "signer.pem"
    signer_path.write_bytes(
        signer.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    runner = CliRunner()
    evaluated = runner.invoke(
        app,
        ["evaluate", str(request), "--signing-key", str(signer_path), "--json"],
    )
    assert evaluated.exit_code == 0, evaluated.output
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    verifier = Ed25519PrivateKey.generate()
    verifier_path = tmp_path / "verifier.pem"
    verifier_path.write_bytes(
        verifier.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    args = [
        "verify",
        str(root / "evidence"),
        "--policy",
        str(root / "policy.json"),
        "--expected-baseline-run",
        digest(baseline),
        "--expected-subject-run",
        digest(candidate),
        "--expected-request-digest",
        captured_request_digest(baseline, candidate, policy),
        "--expected-signer",
        manifest["signing_key_fingerprint"],
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--verifier-signing-key",
        str(verifier_path),
        "--verifier-identity",
        "budget-test",
        "--json",
    ]
    rejected = runner.invoke(app, [*args, "--max-bootstrap-draws", "0"])
    assert rejected.exit_code == 2
    assert "local_work_budget_exceeded" in json.loads(rejected.stdout)["errors"]
    assert not (tmp_path / "receipt.json").exists()
    args[args.index("--receipt") + 1] = str(tmp_path / "accepted-receipt.json")
    # Two scalar metrics cover the overall 40 pairs and the 20 exception pairs.
    required = 2048 * 2 * (40 + 20)
    accepted = runner.invoke(app, [*args, "--max-bootstrap-draws", str(required)])
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["ok"] is True


def test_cli_recipient_default_cannot_inherit_evaluation_allowance(tmp_path):
    baseline, subject, policy = overlapping_project()
    root = tmp_path / "approved"
    request = materialize_captured_request(root, baseline, subject, policy)
    runner = CliRunner()
    keys = tmp_path / "keys"
    generated = runner.invoke(app, ["evaluate", "--keygen", str(keys), "--json"])
    assert generated.exit_code == 0, generated.output
    signer = json.loads(generated.stdout)["details"]["public_key_fingerprint"]
    required = 2048 * 16 * 184 * 17
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(keys / "private.pem"),
            "--max-bootstrap-draws",
            str(required),
            "--json",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    args = [
        "verify",
        str(root / "evidence"),
        "--policy",
        str(root / "policy.json"),
        "--expected-baseline-run",
        digest(baseline),
        "--expected-subject-run",
        digest(subject),
        "--expected-request-digest",
        captured_request_digest(baseline, subject, policy),
        "--expected-signer",
        signer,
        "--verifier-signing-key",
        str(keys / "private.pem"),
        "--verifier-identity",
        "recipient",
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--json",
    ]
    refused = runner.invoke(app, args)
    assert refused.exit_code == 2, refused.output
    result = json.loads(refused.stdout)
    assert result["integrity_ok"] is None
    assert result["errors"] == ["local_work_budget_exceeded"]
    assert not (tmp_path / "receipt.json").exists()
    accepted = runner.invoke(app, [*args, "--max-bootstrap-draws", str(required)])
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["ok"] is True
