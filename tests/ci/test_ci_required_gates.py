from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_workflow(path: Path) -> dict[str, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _find_step_by_name(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_ci_adds_actionlint_and_packaging_smoke_gates() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))

    tests_docs = workflow["jobs"]["tests-docs"]
    docs_steps = tests_docs["steps"]
    actionlint_step = _find_step_by_name(docs_steps, "Run actionlint")
    assert (
        "go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.7"
        in actionlint_step["run"]
    )
    assert "make actionlint" in actionlint_step["run"]
    fast_step = _find_step_by_name(docs_steps, "Run complete fast lane")
    assert fast_step["run"] == "make test-fast"
    local_hf_step = _find_step_by_name(docs_steps, "Run local HF pipeline smoke")
    assert local_hf_step["run"] == "make local-hf-pipeline-smoke"
    mutation_step = _find_step_by_name(docs_steps, "Run mutation smoke")
    assert mutation_step["run"] == "make mutation-smoke"
    docs_live_fast_step = _find_step_by_name(docs_steps, "Run curated live examples")
    assert docs_live_fast_step["run"] == "make docs-live-fast"

    min_py312 = workflow["jobs"]["tests-min-py312"]
    min_steps = min_py312["steps"]
    packaging_step = _find_step_by_name(min_steps, "Packaging smoke (minimal install)")
    assert packaging_step["run"] == "make packaging-smoke-minimal"


def test_ci_verify_full_runs_explicit_closure_gates() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    verify_full = workflow["jobs"]["verify-full"]
    node_step = _find_step_by_name(verify_full["steps"], "Set up Node.js")
    npm_step = _find_step_by_name(verify_full["steps"], "Install docs lint toolchain")
    verify_step = _find_step_by_name(verify_full["steps"], "Full verify")

    assert node_step["with"]["node-version"] == "22"
    assert npm_step["run"] == "npm ci"
    assert "make verify" in verify_step["run"]
    assert "make actionlint" in verify_step["run"]
    assert "make packaging-smoke-minimal" in verify_step["run"]
    assert "mkdocs build --strict" in verify_step["run"]
