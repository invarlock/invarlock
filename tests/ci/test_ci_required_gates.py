from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


def _load(path: str) -> dict[str, Any]:
    workflow = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def _assert_core_wheel_install(job: dict[str, Any]) -> None:
    install = _step(job, "Install dependencies")["run"]
    assert "--require-hashes" in install
    assert "python -m build --wheel --no-isolation" in install
    assert "--no-deps --force-reinstall dist/*.whl" in install


def test_ci_runs_the_repository_gates() -> None:
    workflow = _load(".github/workflows/ci.yml")
    jobs = workflow["jobs"]

    assert set(jobs) == {
        "policy-engine-interop",
        "verify-fast",
        "minimum-python",
        "coverage",
        "coverage-tests",
        "verify-full",
        "supply-chain",
    }
    assert workflow["on"]["push"]["branches"] == [
        "main",
        "staging/next",
        "release/v*",
    ]

    interop = jobs["policy-engine-interop"]
    assert _step(interop, "Set up Python")["with"]["python-version"] == "3.12"
    install = _step(interop, "Install pinned policy engines")["run"]
    assert "github.com/open-policy-agent/opa@v1.17.0" in install
    assert "cuelang.org/go/cmd/cue@v0.16.1" in install
    assert (
        "make acceptance-policy-interop"
        in _step(
            interop,
            "Run policy-engine interoperability matrix",
        )["run"]
    )

    fast = jobs["verify-fast"]
    _assert_core_wheel_install(fast)
    assert _step(fast, "Set up uv")["with"]["version"] == "0.10.10"
    assert _step(fast, "Run repository checks")["run"] == "make verify-checks"
    assert _step(fast, "Build, install, and validate distributions")["run"] == (
        "make addins-install-smoke inspect-judge-sdk-test"
    )
    assert _step(fast, "Lint workflows")["run"].endswith("make workflow-lint\n")

    minimum = jobs["minimum-python"]
    _assert_core_wheel_install(minimum)
    assert _step(minimum, "Set up uv")["with"]["version"] == "0.10.10"
    python = _step(minimum, "Set up Python")
    assert python["with"]["python-version"] == "3.12"
    assert _step(minimum, "Run minimum-Python tests")["run"] == (
        "make test-fast addins-test PYTEST_WORKERS=auto"
    )
    assert _step(minimum, "Check command surface")["run"] == "make cli-smoke-core"
    assert _step(minimum, "Build, install, and validate distributions")["run"] == (
        "make addins-install-smoke inspect-judge-sdk-test"
    )
    assert minimum["timeout-minutes"] >= 35

    coverage = jobs["coverage"]
    _assert_core_wheel_install(coverage)
    capacity = _step(fast, "Check full-capacity signed verification")
    assert capacity["run"] == (
        "python -m pytest -q 'tests/evaluation_comparison/test_capacity.py::"
        "test_full_capacity_signed_independent_recipient[50000]'"
    )
    assert set(coverage["needs"]) == {"coverage-tests", "verify-fast"}
    assert "always()" in coverage["if"]
    assert _step(coverage, "Enforce coverage")["run"] == "make coverage-report"
    shards = jobs["coverage-tests"]
    _assert_core_wheel_install(shards)
    assert shards["strategy"]["fail-fast"] is False
    assert set(shards["strategy"]["matrix"]["shard"]) == {
        "core",
        "examples",
        "support",
        "addins",
    }
    assert (
        _step(shards, "Retain coverage and test timings")["with"][
            "include-hidden-files"
        ]
        is True
    )
    assert (
        _step(coverage, "Download coverage measurements")["with"]["pattern"]
        == "coverage-shard-*"
    )

    supply_chain = jobs["supply-chain"]
    audit = _step(supply_chain, "Audit maintained dependency locks")
    assert "scripts/security/cve_audit.py" in audit["run"]
    upload = _step(supply_chain, "Upload dependency audit report")
    assert upload["if"] == "${{ always() }}"
    assert "cve-audit.json" in upload["with"]["path"]
    assert supply_chain["steps"].index(upload) == supply_chain["steps"].index(audit) + 1


def test_manual_full_ci_uses_standard_repository_and_distribution_gates() -> None:
    workflow = _load(".github/workflows/ci.yml")
    full = workflow["jobs"]["verify-full"]

    _assert_core_wheel_install(full)
    assert (
        "python -m pip install --require-hashes "
        "-r requirements/workflows/docs-ci-py313.txt"
    ) in _step(full, "Install dependencies")["run"]
    assert "workflow_dispatch" in full["if"]
    assert _step(full, "Set up uv")["with"]["version"] == "0.10.10"
    assert _step(full, "Install documentation linters")["run"] == "npm ci"
    assert _step(full, "Run complete repository gates")["run"] == "make verify"
    assert _step(full, "Build, install, and validate distributions")["run"] == (
        "make addins-install-smoke inspect-judge-sdk-test"
    )


def test_ci_has_no_retired_product_workflows_or_jobs() -> None:
    workflows = Path(".github/workflows")
    assert not (workflows / "guard-effect-benchmark.yml").exists()
    assert not (workflows / "statistical-calibration.yml").exists()

    text = (workflows / "ci.yml").read_text(encoding="utf-8").lower()
    retired = ("guard", "calibration", "training", "edit", "quantization", "catalog")
    assert [marker for marker in retired if marker in text] == []


def test_docs_ci_reports_for_every_pull_request_and_scopes_pushes() -> None:
    workflow = _load(".github/workflows/docs-ci.yml")
    docs = workflow["jobs"]["docs"]

    assert _step(docs, "Exercise documented commands")["run"] == ("make docs-live-fast")
    assert workflow["on"]["pull_request"] == {
        "branches": ["main", "staging/next", "release/v*"]
    }

    paths = workflow["on"]["push"]["paths"]
    expected_paths = {
        "*.md",
        "*.MD",
        "**/*.md",
        "**/*.MD",
        "docs/**",
        "tests/docs/**",
        "mkdocs.yml",
        "Makefile",
        "package.json",
        "package-lock.json",
        "requirements/workflows/docs-ci-py313.txt",
        ".github/workflows/docs-ci.yml",
    }
    assert len(paths) == len(expected_paths)
    assert set(paths) == expected_paths


@pytest.mark.parametrize(
    "failures, expected_calls, expected_status", [(0, 2, 0), (1, 3, 0), (3, 3, 7)]
)
def test_policy_install_retries_are_bounded_and_fail_closed(
    failures, expected_calls, expected_status
):
    install = _step(
        _load(".github/workflows/ci.yml")["jobs"]["policy-engine-interop"],
        "Install pinned policy engines",
    )["run"]
    fake_tools = """
calls=0
go() {
  calls=$((calls + 1))
  printf '%s\\n' "$*"
  if [ "$calls" -le "$FAILURES" ]; then return 7; fi
  return 0
}
sleep() { :; }
"""
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            f"FAILURES={failures}\n" + fake_tools + install,
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected_status
    calls = result.stdout.splitlines()
    assert len(calls) == expected_calls
    assert all(
        line
        in {
            "install github.com/open-policy-agent/opa@v1.17.0",
            "install cuelang.org/go/cmd/cue@v0.16.1",
        }
        for line in calls
    )
    if expected_status == 0:
        assert calls[-1] == "install cuelang.org/go/cmd/cue@v0.16.1"
    else:
        assert all("opa@v1.17.0" in line for line in calls)
    assert "GOSUMDB" not in install and "GOINSECURE" not in install


@pytest.mark.parametrize(
    "shards,checks,accepted",
    [
        ("success", "success", True),
        ("failure", "success", False),
        ("success", "failure", False),
        ("skipped", "success", False),
        ("cancelled", "success", False),
        ("success", "skipped", False),
        ("", "success", False),
    ],
)
def test_coverage_gate_rejects_incomplete_execution(shards, checks, accepted):
    import os

    workflow = _load(".github/workflows/ci.yml")
    command = _step(
        workflow["jobs"]["coverage"], "Require successful test and capacity checks"
    )["run"]
    result = subprocess.run(
        ["bash", "-c", command],
        env={**os.environ, "SHARD_RESULT": shards, "CHECK_RESULT": checks},
        check=False,
    )
    assert (result.returncode == 0) is accepted


def test_main_ci_runs_hardened_accelerate_apis_in_installed_linux_environment() -> None:
    workflow = _load(".github/workflows/ci.yml")
    for name in ("verify-fast", "verify-full"):
        job = workflow["jobs"][name]
        assert job["runs-on"] == "ubuntu-latest"
        steps = job["steps"]
        smoke = next(
            step
            for step in steps
            if step.get("name") == "Check installed hardened Accelerate APIs"
        )
        assert smoke["run"] == "python -I tests/scripts/hardened_accelerate_checks.py"
        assert "if" not in smoke
        install = next(
            step for step in steps if step.get("name") == "Install dependencies"
        )
        assert steps.index(install) < steps.index(smoke)
        assert "--no-deps --force-reinstall dist/*.whl" in install["run"]
