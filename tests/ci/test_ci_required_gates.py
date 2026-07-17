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


def test_ci_runs_the_repository_gates() -> None:
    workflow = _load(".github/workflows/ci.yml")
    jobs = workflow["jobs"]

    assert set(jobs) == {
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

    fast = jobs["verify-fast"]
    assert _step(fast, "Run fast repository gates")["run"] == "make verify-fast"
    assert _step(fast, "Build, install, and validate distributions")["run"] == (
        "make addins-install-smoke"
    )
    assert _step(fast, "Lint workflows")["run"].endswith("make workflow-lint\n")

    minimum = jobs["minimum-python"]
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
    assert _step(coverage, "Enforce coverage")["run"] == "make coverage-enforce"


def test_manual_full_ci_uses_standard_repository_and_distribution_gates() -> None:
    workflow = _load(".github/workflows/ci.yml")
    full = workflow["jobs"]["verify-full"]

    assert "workflow_dispatch" in full["if"]
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


def test_docs_ci_uses_the_current_documentation_targets() -> None:
    workflow = _load(".github/workflows/docs-ci.yml")
    docs = workflow["jobs"]["docs"]

    assert _step(docs, "Check documentation")["run"] == "make docs-check"
    assert _step(docs, "Exercise documented commands")["run"] == ("make docs-live-fast")

    paths = workflow["on"]["pull_request"]["paths"]
    assert "docs/**" in paths
    assert "*.md" in paths
    for maintained_surface in (
        ".github/**/*.md",
        "addins/**/*.md",
        "examples/**/*.md",
        "public_evidence/**/*.md",
        "requirements/**/*.md",
        "scripts/**/*.md",
        "tests/README.md",
        "tests/docs/**",
    ):
        assert maintained_surface in paths
    assert not any(path.startswith("notebooks/") for path in paths)
    assert not any(path.startswith("scripts/docs/") for path in paths)
