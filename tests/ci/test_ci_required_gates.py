from __future__ import annotations

from pathlib import Path
from typing import Any

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
        "minimum-python-tests",
        "minimum-python-packages",
        "coverage-tests",
        "coverage",
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
    assert _step(fast, "Check full-capacity signed verification")["run"] == (
        "python -m pytest -q 'tests/evaluation_comparison/test_capacity.py::test_full_capacity_signed_independent_recipient[50000]'"
    )
    assert _step(fast, "Build, install, and validate distributions")["run"] == (
        "make install-smoke inspect-judge-sdk-test langfuse-sdk-test"
    )
    assert _step(
        fast,
        "Check all dedicated evaluator profiles with an installed recipient",
    )["run"] == (
        "INVARLOCK_REPLAY_RETAINED_CAMPAIGNS=0 "
        "PYTHON=python bash scripts/evaluator_parity_gate.sh"
    )
    assert _step(fast, "Lint workflows")["run"].endswith("make workflow-lint\n")

    minimum = jobs["minimum-python-packages"]
    _assert_core_wheel_install(minimum)
    assert _step(minimum, "Set up uv")["with"]["version"] == "0.10.10"
    python = _step(minimum, "Set up Python")
    assert python["with"]["python-version"] == "3.12"
    assert _step(minimum, "Check command surface")["run"] == "make cli-smoke-core"
    assert _step(minimum, "Build, install, and validate distributions")["run"] == (
        "make install-smoke inspect-judge-sdk-test langfuse-sdk-test"
    )
    assert _step(
        minimum,
        "Check all dedicated evaluator profiles with an installed recipient",
    )["run"] == (
        "INVARLOCK_REPLAY_RETAINED_CAMPAIGNS=0 "
        "PYTHON=python bash scripts/evaluator_parity_gate.sh"
    )
    assert minimum["timeout-minutes"] >= 35

    gate = jobs["minimum-python"]
    assert set(gate["needs"]) == {"minimum-python-tests", "minimum-python-packages"}
    assert "always()" in gate["if"]
    assert _step(gate, "Require every minimum-Python check")["run"] == (
        'test "$TEST_RESULT" = success && test "$PACKAGE_RESULT" = success'
    )
    tests = jobs["minimum-python-tests"]
    assert tests["strategy"]["matrix"] == jobs["coverage-tests"]["strategy"]["matrix"]
    assert _step(tests, "Run disjoint minimum-Python tests")["run"] == (
        "python scripts/ci/coverage_runner.py test ${{ matrix.shard }} --workers 2"
    )
    coverage_tests = jobs["coverage-tests"]
    _assert_core_wheel_install(coverage_tests)
    assert coverage_tests["strategy"]["matrix"]["shard"] == [
        "core",
        "examples",
        "examples-comparisons",
        "examples-judge",
        "support",
        "runtime",
    ]
    assert _step(coverage_tests, "Collect disjoint test coverage")["run"] == (
        "make coverage-collect-${{ matrix.shard }} PYTEST_WORKERS=2"
    )

    coverage = jobs["coverage"]
    _assert_core_wheel_install(coverage)
    assert _step(coverage, "Enforce coverage")["run"] == "make coverage-report"

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
    assert "workflow_dispatch" in full["if"]
    assert _step(full, "Set up uv")["with"]["version"] == "0.10.10"
    assert _step(full, "Install documentation linters")["run"] == "npm ci"
    assert _step(full, "Run complete repository gates")["run"] == "make verify"
    assert _step(full, "Build, install, and validate distributions")["run"] == (
        "make install-smoke inspect-judge-sdk-test langfuse-sdk-test"
    )


def test_ci_has_no_retired_product_workflows_or_jobs() -> None:
    workflows = Path(".github/workflows")
    assert not (workflows / "guard-effect-benchmark.yml").exists()
    assert not (workflows / "statistical-calibration.yml").exists()

    text = (workflows / "ci.yml").read_text(encoding="utf-8").lower()
    retired = ("guard", "calibration", "training", "edit", "quantization", "catalog")
    assert [marker for marker in retired if marker in text] == []


def test_evaluator_sdk_gate_tracks_shared_capture_and_scorer_dependencies() -> None:
    workflow = _load(".github/workflows/evaluator-sdk.yml")
    paths = set(workflow["on"]["pull_request"]["paths"])
    required = {
        "src/invarlock/**",
        "contracts/**",
        "requirements/workflows/core-py312.txt",
        "requirements/workflows/release-install-py312.txt",
        "pyproject.toml",
        "uv.lock",
        "Makefile",
    }
    assert required <= paths
    assert not any(
        path.startswith("src/invarlock/") and path != "src/invarlock/**"
        for path in paths
    )


def test_docs_ci_reports_for_every_pull_request_and_scopes_pushes() -> None:
    workflow = _load(".github/workflows/docs-ci.yml")
    docs = workflow["jobs"]["docs"]

    assert _step(docs, "Exercise documented commands")["run"] == "make docs-live-fast"
    assert _step(docs, "Confirm a clean source tree")["run"].endswith(
        "git diff --exit-code\n"
    )
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
