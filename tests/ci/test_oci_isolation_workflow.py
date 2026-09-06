"""Require actual OCI isolation execution and retained outcomes in CI."""

from pathlib import Path

import yaml


def test_front_door_runs_isolation_cases_and_retains_failed_outcomes():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-front-door-smoke.yml").read_text()
    )
    steps = workflow["jobs"]["smoke"]["steps"]
    journey = next(
        step for step in steps if "signed canary journey" in step.get("name", "")
    )
    assert "tests/integration/test_oci_isolation.py" in journey["run"]
    assert "INVARLOCK_OCI_ISOLATION_RESULTS=" in journey["run"]
    upload = next(
        step for step in steps if step.get("name") == "Retain OCI isolation outcomes"
    )
    assert "always()" in upload["if"]
    assert upload["with"]["path"] == "artifacts/oci-isolation"
