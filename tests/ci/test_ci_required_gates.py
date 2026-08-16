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
    assert _step(fast, "Run fast repository gates")["run"] == "make verify-fast"
    assert _step(fast, "Build, install, and validate distributions")["run"] == (
        "make addins-install-smoke"
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
        "make addins-install-smoke"
    )
    assert minimum["timeout-minutes"] >= 35

    coverage = jobs["coverage"]
    _assert_core_wheel_install(coverage)
    assert _step(coverage, "Enforce coverage")["run"] == "make coverage-enforce"

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
        "make addins-install-smoke"
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

    assert _step(docs, "Check documentation")["run"] == "make docs-check"
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
