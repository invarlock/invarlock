from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load(path: str) -> dict[str, Any]:
    workflow = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _step(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_secret_history_workflow_runs_a_scheduled_full_history_scan() -> None:
    workflow = _load(".github/workflows/secret-history.yml")
    assert workflow["on"]["schedule"] == [{"cron": "17 9 * * 1"}]
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["gitleaks-history"]
    checkout = _step(job["steps"], "Checkout repository")
    assert checkout["with"]["fetch-depth"] == 0

    scan = _step(job["steps"], "Run gitleaks full history scan")
    assert "gitleaks git ." in scan["run"]
    assert "--config .gitleaks.toml" in scan["run"]
    assert "--log-opts" not in scan["run"]


def test_full_ci_pins_make_to_setup_python() -> None:
    workflow = _load(".github/workflows/ci.yml")
    full = workflow["jobs"]["verify-full"]
    assert full["env"]["PYTHON"] == "python"
    assert _step(full["steps"], "Run complete repository gates")["run"] == (
        "make verify"
    )
