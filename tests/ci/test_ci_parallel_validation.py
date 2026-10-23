"""Parallel CI must retain its required gates and exact replay workload."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci import coverage_runner


def test_capacity_partition_preserves_fast_and_supplemental_cases():
    paths = {
        "tests/evaluation_comparison/test_capacity.py",
        "tests/evaluation_comparison/test_default_resource_budget.py",
    }
    capacity = coverage_runner.selection("core-capacity")
    assert set(capacity[:-2]) == paths
    assert capacity[-2:] == ["-m", coverage_runner.FAST_MARKERS]
    for path in paths:
        assert f"--ignore={path}" in coverage_runner.selection("core")
        assert f"--ignore={path}" not in coverage_runner.test_selection("supplement")


def test_comparison_and_journey_partitions_preserve_all_moved_files():
    partitions = coverage_runner.EXAMPLE_PARTITIONS
    assert partitions["examples-parity"] == (
        "tests/integration/test_evaluator_parity.py",
    )
    assert set(partitions["examples-comparisons"]) == {
        "tests/examples/test_priority_workflow_reference.py",
        "tests/examples/test_captured_reference.py",
    }
    assert set(partitions["examples-journeys"]) == {
        "tests/examples/test_modelkit_real_journey.py",
        "tests/examples/test_evaluator_real_judge_journey.py",
        "tests/examples/test_judge_heldout_reference.py",
        "tests/examples/test_live_evaluator_reference.py",
    }
    moved = [path for paths in partitions.values() for path in paths]
    assert len(moved) == len(set(moved))
    assert all(
        f"--ignore={path}" in coverage_runner.selection("examples") for path in moved
    )


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
@pytest.mark.parametrize(
    "dependency", ["TEST_RESULT", "PACKAGE_RESULT", "PARITY_RESULT"]
)
def test_minimum_python_gate_requires_every_independent_check(result, dependency):
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    gate = jobs["minimum-python"]
    assert set(gate["needs"]) == {
        "minimum-python-tests",
        "minimum-python-packages",
        "minimum-python-parity",
    }
    assert "always()" in gate["if"]
    step = next(
        s for s in gate["steps"] if s["name"] == "Require every minimum-Python check"
    )
    assert step["env"] == {
        "TEST_RESULT": "${{ needs.minimum-python-tests.result }}",
        "PACKAGE_RESULT": "${{ needs.minimum-python-packages.result }}",
        "PARITY_RESULT": "${{ needs.minimum-python-parity.result }}",
    }
    env = {**os.environ, **dict.fromkeys(step["env"], "success")}
    env[dependency] = result
    completed = subprocess.run(["bash", "-c", step["run"]], env=env, check=False)
    assert (completed.returncode == 0) == (result == "success")


def test_judge_replays_have_independent_disjoint_partitions():
    expected = {
        "tests/examples/test_judge_measurements_reference.py",
        "tests/examples/test_judge_measurements_review.py",
        "tests/examples/test_priority_judge_reference.py",
    }
    partitions = {
        name: paths
        for name, paths in coverage_runner.EXAMPLE_PARTITIONS.items()
        if name.startswith("examples-judge")
    }
    assert len(partitions) == 3
    assert all(len(paths) == 1 for paths in partitions.values())
    assert {path for paths in partitions.values() for path in paths} == expected
    for path in expected:
        assert f"--ignore={path}" in coverage_runner.selection("examples")


@pytest.mark.parametrize(
    ("workflow", "job"),
    [
        ("ci.yml", "coverage-tests"),
        ("ci.yml", "minimum-python-tests"),
        ("release.yml", "coverage_shards"),
    ],
)
def test_every_workflow_collects_all_required_shards(workflow, job):
    jobs = yaml.safe_load((Path(".github/workflows") / workflow).read_text())["jobs"]
    expected = (
        coverage_runner.MINIMUM_PYTHON_SHARDS
        if job == "minimum-python-tests"
        else coverage_runner.SHARDS
    )
    assert jobs[job]["strategy"]["matrix"]["shard"] == list(expected)
    makefile = Path("Makefile").read_text()
    shards = re.search(r"(?m)^COVERAGE_SHARDS := (.+)$", makefile)
    assert shards is not None
    assert shards.group(1).split() == list(coverage_runner.SHARDS)


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
@pytest.mark.parametrize(
    "dependency", ["CHECK_RESULT", "PACKAGE_RESULT", "PARITY_RESULT"]
)
def test_required_verification_gate_rejects_every_unsuccessful_dependency(
    result, dependency
):
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    gate = jobs["verify-fast"]
    assert set(gate["needs"]) == {
        "verify-checks",
        "verify-packages",
        "evaluator-parity",
    }
    assert "always()" in gate["if"]
    assert "inputs.full" in gate["if"]
    step = next(
        step
        for step in gate["steps"]
        if step["name"] == "Require every verification check"
    )
    assert step["env"] == {
        "CHECK_RESULT": "${{ needs.verify-checks.result }}",
        "PACKAGE_RESULT": "${{ needs.verify-packages.result }}",
        "PARITY_RESULT": "${{ needs.evaluator-parity.result }}",
    }
    env = {**os.environ, **dict.fromkeys(step["env"], "success")}
    env[dependency] = result
    completed = subprocess.run(["bash", "-c", step["run"]], env=env, check=False)
    assert (completed.returncode == 0) == (result == "success")
    assert set(jobs["coverage"]["needs"]) == {"coverage-tests", "verify-fast"}


def test_parallel_parity_retains_the_isolated_installed_recipient_command():
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    parity = jobs["evaluator-parity"]
    checks = jobs["verify-checks"]
    assert "needs" not in parity and "needs" not in checks
    assert parity["if"] == checks["if"]
    commands = [step.get("run", "") for step in parity["steps"]]
    assert commands[-1] == (
        "INVARLOCK_REPLAY_RETAINED_CAMPAIGNS=0 "
        "PYTHON=python bash scripts/evaluator_parity_gate.sh"
    )
    assert any("--no-deps --force-reinstall dist/*.whl" in run for run in commands)


def test_reporting_lock_preserves_versions_without_restoring_the_runtime_cache():
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    report = jobs["coverage"]
    setup = next(step for step in report["steps"] if step["name"] == "Set up Python")
    assert "cache" not in setup["with"]
    commands = "\n".join(step.get("run", "") for step in report["steps"])
    assert (
        "--require-hashes -r requirements/workflows/coverage-report-py313.txt"
        in commands
    )
    assert "make coverage-report" in commands
    locks = Path("requirements/workflows")

    def versions(name):
        return dict(
            re.findall(r"(?m)^([\w.-]+)==([^ \\\n]+)", (locks / name).read_text())
        )

    reporting = versions("coverage-report-py313.txt")
    measured = versions("ci-hf-py313.txt")
    assert {
        "coverage",
        "cryptography",
        "build",
        "setuptools",
        "wheel",
    } <= reporting.keys()
    assert not {"torch", "transformers", "accelerate"} & reporting.keys()
    assert all(measured[name] == version for name, version in reporting.items())


def test_main_ci_does_not_transfer_the_shared_model_wheel_cache():
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    for name, job in jobs.items():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/setup-python@"):
                # Restoring the multi-GB cache exceeded account egress limits
                # and delayed setup by 12 minutes before any checks could run.
                assert "cache" not in step.get("with", {}), name
                assert "cache-dependency-path" not in step.get("with", {}), name
