"""Omitted local settings must bound a complete, highly overlapping policy."""

import copy
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.pipeline import comparison, create_evidence, verify_evidence
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import PipelineError, digest
from invarlock.pipeline.templates import example_project


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
    evidence = create_evidence(baseline, candidate, policy, key)
    replay = verify_evidence(
        evidence,
        public_key=key.public_key(),
        expected_baseline=digest(baseline),
        expected_candidate=digest(candidate),
        policy=policy,
    )
    assert replay == evidence["comparison"]
    assert replay["metrics"][0]["count"] == 50000
    assert replay["metrics"][0]["interval"]["replicates"] == 2048
    with pytest.raises(PipelineError, match="102400000 bootstrap draws"):
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


@pytest.mark.parametrize("check", [comparison.compare_runs, create_evidence])
def test_omitted_budget_rejects_before_scoring(check, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the default must reject before computing a partial verdict")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match="local budget is 102400000"):
        check(*overlapping_project())


def test_explicit_none_preserves_complete_replay_but_not_recipient_default(monkeypatch):
    baseline, candidate, policy = overlapping_project()
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(
        baseline, candidate, policy, key, max_bootstrap_draws=None
    )
    options = {
        "public_key": key.public_key(),
        "expected_baseline": digest(baseline),
        "expected_candidate": digest(candidate),
        "policy": policy,
    }
    replay = verify_evidence(evidence, **options, max_bootstrap_draws=None)
    assert replay == evidence["comparison"]
    assert len(replay["metrics"]) == 272
    assert all(m["count"] == 184 for m in replay["metrics"])

    def forbidden(*args, **kwargs):
        pytest.fail("an evidence author's override cannot raise the recipient budget")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match="local budget is 102400000"):
        verify_evidence(evidence, **options)


def test_cli_omitted_budget_rejects_without_publishing(tmp_path):
    baseline, candidate, policy = overlapping_project()
    runner = CliRunner()
    root = tmp_path / "project"
    assert runner.invoke(app, ["init", str(root), "--example", "judge"]).exit_code == 0
    project = json.loads((root / "pipeline.json").read_text())
    for side, run in (("baseline", baseline), ("candidate", candidate)):
        (root / project[side]["path"]).write_text(json.dumps(run))
    (root / project["policy"]).write_text(json.dumps(policy))
    result = runner.invoke(
        app, ["compare", str(root / "pipeline.json"), "--output", str(root / "result")]
    )
    assert result.exit_code == 2
    assert "local budget is 102400000" in json.loads(result.stdout)["message"]
    assert not (root / "result").exists()


def test_cli_recipient_requires_its_own_larger_budget(tmp_path):
    baseline, candidate, policy = overlapping_project()
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(
        baseline, candidate, policy, key, max_bootstrap_draws=None
    )
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    (tmp_path / "public.pem").write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    args = [
        "verify",
        str(tmp_path / "evidence.json"),
        "--public-key",
        str(tmp_path / "public.pem"),
        "--policy",
        str(tmp_path / "policy.json"),
        "--expected-baseline",
        digest(baseline),
        "--expected-candidate",
        digest(candidate),
    ]
    runner = CliRunner()
    rejected = runner.invoke(app, args)
    assert rejected.exit_code == 2
    assert json.loads(rejected.stdout)["status"] == "integration_error"
    assert "local budget is 102400000" in json.loads(rejected.stdout)["message"]
    accepted = runner.invoke(
        app, [*args, "--max-bootstrap-draws", str(184 * 16 * 17 * 2048)]
    )
    assert accepted.exit_code == 0
    assert json.loads(accepted.stdout) == {
        "authenticated": True,
        "decision": "pass",
        "exit_code": 0,
    }
