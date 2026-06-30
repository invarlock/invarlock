from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ACTIONS_CACHE_PIN = "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"


def _load_workflow(path: Path) -> dict[str, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _find_step_by_name(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_secret_history_workflow_runs_scheduled_full_history_gitleaks() -> None:
    workflow = _load_workflow(Path(".github/workflows/secret-history.yml"))
    triggers = workflow["on"]

    assert triggers["schedule"][0]["cron"] == "17 9 * * 1"
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["gitleaks-history"]
    assert job["name"] == "gitleaks-history"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 45

    steps = job["steps"]
    checkout_step = _find_step_by_name(steps, "Checkout repository")
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert checkout_step["with"]["fetch-depth"] == 0

    cache_step = _find_step_by_name(steps, "Cache gitleaks binary")
    assert cache_step["uses"] == ACTIONS_CACHE_PIN
    assert cache_step["with"]["path"] == "~/go/bin/gitleaks"
    assert "gitleaks-v8.30.0" in cache_step["with"]["key"]

    install_step = _find_step_by_name(steps, "Install gitleaks")
    assert (
        "go install github.com/zricethezav/gitleaks/v8@v8.30.0" in install_step["run"]
    )
    assert 'if [ ! -x "${gitleaks_bin}" ]; then' in install_step["run"]

    scan_step = _find_step_by_name(steps, "Run gitleaks full history scan")
    assert "gitleaks git ." in scan_step["run"]
    assert "--config .gitleaks.toml" in scan_step["run"]
    assert "--report-format json" in scan_step["run"]
    assert "--report-format sarif" not in scan_step["run"]
    assert "artifacts/supply-chain/gitleaks-history.json" in scan_step["run"]
    assert "--log-opts" not in scan_step["run"]

    upload_step = _find_step_by_name(steps, "Upload gitleaks history artifact")
    assert upload_step["uses"].startswith("actions/upload-artifact@")
    assert upload_step["with"]["name"] == "gitleaks-history"
    assert upload_step["with"]["path"] == "artifacts/supply-chain/gitleaks-history.json"

    fail_step = _find_step_by_name(steps, "Fail on secret findings")
    assert "gitleaks scan did not publish an exit code" in fail_step["run"]
    assert "gitleaks detected secrets" in fail_step["run"]


def test_supply_chain_docs_match_workflow_truth() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    workflows_doc = (repo_root / ".github" / "WORKFLOWS.md").read_text(encoding="utf-8")
    allowlist_doc = (
        repo_root / "docs" / "security" / "pip-audit-allowlist.md"
    ).read_text(encoding="utf-8")
    release_doc = (
        repo_root / "docs" / "security" / "release-verification.md"
    ).read_text(encoding="utf-8")
    architecture_doc = (repo_root / "docs" / "security" / "architecture.md").read_text(
        encoding="utf-8"
    )

    assert "install-surface SBOM" in workflows_doc
    assert "base, `hf`, and" in workflows_doc
    assert "`advanced` shipped dependency surfaces" in workflows_doc
    assert "gitleaks" in workflows_doc
    assert "git-delta JSON artifacts" in workflows_doc
    assert "secret-history.yml" in workflows_doc
    assert "full-history" in workflows_doc
    assert "scripts/security/run_pip_audit.py" in allowlist_doc
    assert "scripts/security/pip_audit_allowlist.json" in allowlist_doc
    assert "installed release surface" in release_doc
    assert "resolved commit SHA" in release_doc
    assert "PyPI" in release_doc
    assert "delta" in release_doc
    assert "scheduled workflow" in release_doc
    assert "gitleaks" in architecture_doc
    assert "git-delta scanning" in architecture_doc
    assert "scheduled full-history scanning" in architecture_doc
    assert "installed-artifact environment" in architecture_doc


def test_ci_verify_full_pins_make_to_setup_python() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    verify_full = workflow["jobs"]["verify-full"]

    env = verify_full.get("env", {})
    assert env["PYTHON"] == "python"

    steps = verify_full.get("steps", [])
    setup_node_step = _find_step_by_name(steps, "Set up Node.js")
    npm_step = _find_step_by_name(steps, "Install docs lint toolchain")
    verify_step = _find_step_by_name(steps, "Full verify")
    assert setup_node_step["with"]["node-version"] == "22"
    assert npm_step["run"] == "npm ci"
    assert "make verify" in verify_step["run"]
    assert "mkdocs build --strict" in verify_step["run"]
