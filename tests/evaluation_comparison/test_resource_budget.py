"""A recipient's resource budget cannot change the signed quality decision."""

import copy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.cli import app
from invarlock.evidence_verification import EvidenceVerificationError
from tests._evaluation_support import (
    EvaluationRecordsError,
    build_pack,
    comparison,
    digest,
    example_project,
    materialize_captured_request,
    replay_pack,
)


def test_overlapping_scopes_rejected_before_any_scoring(monkeypatch):
    baseline, candidate, policy = example_project("judge")
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
    policy["metrics"] = [
        {**copy.deepcopy(policy["metrics"][0]), "name": f"quality-{i}"}
        for i in range(16)
    ]
    required = len(baseline["records"]) * len(policy["metrics"]) * 17 * 2048

    def forbidden(*args, **kwargs):
        pytest.fail("capacity rejection must precede all scoring and intervals")

    monkeypatch.setattr(comparison, "score", forbidden)
    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvaluationRecordsError, match=f"{required} bootstrap draws"):
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
    with pytest.raises(EvaluationRecordsError, match="non-negative integer"):
        comparison.compare_runs(*example_project("judge"), max_bootstrap_draws=budget)


def test_signed_recipient_owns_budget_and_never_returns_partial_verdict(monkeypatch):
    baseline, candidate, policy = example_project("judge")
    key = Ed25519PrivateKey.generate()
    evidence = build_pack(baseline, candidate, policy, key)
    original = dict(evidence.files)

    def forbidden(*args, **kwargs):
        pytest.fail("recipient must reject before interval replay")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvidenceVerificationError, match="local_work_budget_exceeded"):
        replay_pack(
            evidence,
            public_key=key.public_key(),
            expected_baseline_run=digest(baseline),
            expected_subject_run=digest(candidate),
            policy=policy,
            max_bootstrap_draws=0,
        )
    assert evidence.files == original


def test_producer_budget_prevents_signing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("evidence creation must not calculate a partial comparison")

    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvaluationRecordsError, match="bootstrap draws"):
        build_pack(*example_project("judge"), max_bootstrap_draws=0)


@pytest.mark.parametrize("budget", [0, 2048 * 2 * 40 * 17 - 1])
@pytest.mark.parametrize("preflight", [False, True])
def test_cli_capacity_rejection_publishes_no_evidence(tmp_path, budget, preflight):
    baseline, candidate, policy = example_project("judge")
    for run in (baseline, candidate):
        for row in run["records"]:
            row["metadata"]["included"] = "yes"
    policy["slices"] = [
        {"name": f"slice-{i}", "where": {"included": "yes"}} for i in range(16)
    ]
    project = tmp_path / "project"
    request = materialize_captured_request(project, baseline, candidate, policy)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(request),
            "--unsigned",
            "--max-bootstrap-draws",
            str(budget),
            "--json",
            *(["--preflight"] if preflight else []),
        ],
    )
    assert result.exit_code == 2
    assert "bootstrap draws" in " ".join(json.loads(result.stdout)["errors"])
    assert not (project / "evidence").exists()
