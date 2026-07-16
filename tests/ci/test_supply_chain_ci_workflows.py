from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(".github/workflows")
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow.get("jobs", {}).values()
        if isinstance(job, dict)
        for step in job.get("steps", [])
        if isinstance(step, dict)
    ]


def _step(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_workflows_pin_actions_and_declare_permissions() -> None:
    unpinned: list[str] = []
    missing_permissions: list[str] = []

    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = _load(path)
        if "permissions" not in workflow:
            missing_permissions.append(path.name)
        for step in _steps(workflow):
            uses = str(step.get("uses", ""))
            if uses and not uses.startswith("./") and not PINNED_ACTION.fullmatch(uses):
                unpinned.append(f"{path.name}: {uses}")

    assert unpinned == []
    assert missing_permissions == []


def test_workflow_pip_installs_use_hashed_lock_files() -> None:
    offenders: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for step in _steps(_load(path)):
            command = step.get("run")
            if not isinstance(command, str):
                continue
            for line in command.splitlines():
                line = line.strip()
                if "pip install" not in line:
                    continue
                is_built_artifact = "dist/*.whl" in line or "wheelhouse/*.whl" in line
                if not is_built_artifact and "--require-hashes" not in line:
                    offenders.append(f"{path.name}: {line}")

    assert offenders == []


def test_pr_supply_chain_scans_only_shipped_dependency_surfaces() -> None:
    workflow = _load(WORKFLOWS / "supply-chain-pr.yml")
    scan = workflow["jobs"]["scan"]
    steps = scan["steps"]
    names = [step.get("name") for step in steps]

    for required in (
        "Run gitleaks PR git delta scan",
        "Build release wheel",
        "Run pip-audit",
        "Generate install-surface SBOM",
        "Run HF surface pip-audit",
    ):
        assert required in names
    assert not any("advanced" in str(name).lower() for name in names)

    secret_scan = _step(steps, "Run gitleaks PR git delta scan")
    assert "--config .gitleaks.toml" in secret_scan["run"]
    assert "--log-opts" in secret_scan["run"]
    assert "--redact" in secret_scan["run"]


def test_release_builds_from_the_resolved_tag_and_uses_trusted_publishing() -> None:
    workflow = _load(WORKFLOWS / "release.yml")
    jobs = workflow["jobs"]

    resolve = jobs["resolve_release_ref"]
    resolve_step = _step(resolve["steps"], "Resolve release ref")
    assert "git ls-remote --tags" in resolve_step["run"]
    assert "release_tag must start with v" in resolve_step["run"]

    build = jobs["build_check"]
    checkout = build["steps"][0]
    assert checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    distribution_build = _step(build["steps"], "Build first-party distributions")["run"]
    assert "python -m build --no-isolation" in distribution_build
    for addin in ("diagnostics", "gguf", "multimodal", "tensorrt_llm"):
        assert f"addins/{addin}" in distribution_build

    twine = _step(build["steps"], "Twine check")["run"]
    assert "dist/*.whl dist/*.tar.gz" in twine
    assert "dist/addins/*" in twine

    publish = jobs["publish"]
    assert publish["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    publish_step = next(
        step
        for step in publish["steps"]
        if str(step.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )
    assert "user" not in publish_step.get("with", {})
    assert "password" not in publish_step.get("with", {})

    testpypi = jobs["testpypi_smoke"]
    testpypi_checkout = testpypi["steps"][0]
    assert testpypi_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    smoke = _step(testpypi["steps"], "Install published wheel and smoke test")
    assert "requirements/workflows/pip-bootstrap-py313.txt" in smoke["run"]


def test_codeql_and_scorecards_keep_least_privilege() -> None:
    codeql = _load(WORKFLOWS / "codeql.yml")
    analyze = codeql["jobs"]["analyze"]
    assert analyze["permissions"] == {
        "contents": "read",
        "actions": "read",
        "security-events": "write",
    }

    scorecards = _load(WORKFLOWS / "scorecards.yml")
    assert scorecards["permissions"] == {"contents": "read"}
    assert scorecards["jobs"]["analysis"]["permissions"] == {
        "id-token": "write",
        "security-events": "write",
    }


def test_repo_hygiene_covers_integration_branch_and_renames() -> None:
    workflow = _load(WORKFLOWS / "repo-hygiene.yml")
    assert workflow["on"]["pull_request"]["branches"] == ["main", "staging/next"]

    generated = _step(
        workflow["jobs"]["no-generated-artifacts"]["steps"],
        "Detect forbidden files in PR diff",
    )
    large = _step(
        workflow["jobs"]["large-files"]["steps"],
        "Prevent >10MB files in PR diff",
    )
    assert "--diff-filter=ACMR" in generated["run"]
    assert "--diff-filter=ACMR" in large["run"]
