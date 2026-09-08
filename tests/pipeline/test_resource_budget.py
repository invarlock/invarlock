"""A recipient's resource budget cannot change the signed quality decision."""

import copy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.pipeline import comparison, create_evidence, verify_evidence
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import PipelineError, digest
from invarlock.pipeline.templates import example_project


def test_overlapping_scopes_rejected_before_any_scoring(monkeypatch):
    baseline, candidate, policy = example_project("judge")
    for run in (baseline, candidate):
        for row in run["records"]:
            row["metadata"]["included"] = "yes"
    policy["slices"] = [
        {"name": f"slice-{i}", "where": {"included": "yes"}} for i in range(16)
    ]
    required = len(baseline["records"]) * len(policy["metrics"]) * 17 * 2048

    def forbidden(*args, **kwargs):
        pytest.fail("capacity rejection must precede all scoring and intervals")

    monkeypatch.setattr(comparison, "score", forbidden)
    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match=f"{required} bootstrap draws"):
        comparison.compare_runs(
            baseline, candidate, policy, max_bootstrap_draws=required - 1
        )


def test_exact_budget_preserves_all_results_and_replicates():
    baseline, candidate, policy = example_project("judge")
    policy["slices"] = []
    for i, row in enumerate(candidate["records"]):
        row["scores"]["quality"] += (i % 3 - 1) / 100
    expected = comparison.compare_runs(baseline, candidate, policy)
    required = len(baseline["records"]) * len(policy["metrics"]) * 2048
    assert (
        comparison.compare_runs(
            baseline, candidate, policy, max_bootstrap_draws=required
        )
        == expected
    )
    assert all(m["interval"]["replicates"] == 2048 for m in expected["metrics"])


def test_binary_policy_accepts_zero_bootstrap_budget():
    baseline, candidate, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    assert comparison.compare_runs(
        baseline, candidate, policy, max_bootstrap_draws=0
    ) == comparison.compare_runs(baseline, candidate, policy)


@pytest.mark.parametrize("budget", [-1, True, 1.5, "100"])
def test_invalid_budget_is_an_integration_error(budget):
    with pytest.raises(PipelineError, match="non-negative integer"):
        comparison.compare_runs(*example_project("judge"), max_bootstrap_draws=budget)


def test_signed_recipient_owns_budget_and_never_returns_partial_verdict(monkeypatch):
    baseline, candidate, policy = example_project("judge")
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(baseline, candidate, policy, key)
    original = copy.deepcopy(evidence)

    def forbidden(*args, **kwargs):
        pytest.fail("recipient must reject before interval replay")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match="bootstrap draws"):
        verify_evidence(
            evidence,
            public_key=key.public_key(),
            expected_baseline=digest(baseline),
            expected_candidate=digest(candidate),
            policy=policy,
            max_bootstrap_draws=0,
        )
    assert evidence == original


def test_producer_budget_prevents_signing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("evidence creation must not calculate a partial comparison")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match="bootstrap draws"):
        create_evidence(*example_project("judge"), max_bootstrap_draws=0)


def test_cli_capacity_rejection_publishes_no_evidence(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    assert (
        runner.invoke(app, ["init", str(project), "--example", "judge"]).exit_code == 0
    )
    result = runner.invoke(
        app,
        [
            "compare",
            str(project / "pipeline.json"),
            "--output",
            str(project / "result"),
            "--max-bootstrap-draws",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "integration_error"
    assert "bootstrap draws" in json.loads(result.stdout)["message"]
    assert not (project / "result").exists()
