from pathlib import Path

import yaml


def test_supply_chain_job_configured():
    """Test supply-chain job includes core security checks.

    Note: gitleaks secret scanning was removed to reduce CI cost.
    CodeQL workflow handles security scanning instead.
    """
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow definition not found"

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    assert "supply-chain" in jobs, "Supply-chain job missing from CI workflow"

    supply_job = jobs["supply-chain"]
    steps = supply_job.get("steps", [])
    step_names = [step.get("name") for step in steps if isinstance(step, dict)]

    assert "Generate SBOM" in step_names
    assert "Run pip-audit" in step_names

    audit_commands = [step.get("run", "") for step in steps if isinstance(step, dict)]
    assert any("pip-audit" in cmd for cmd in audit_commands)
    assert any("--ignore-vuln GHSA-4xh5-x5gv-qwph" in cmd for cmd in audit_commands)


def test_generate_sbom_script_exists():
    script_path = Path("scripts/generate_sbom.sh")
    assert script_path.exists(), "SBOM generator script missing"

    contents = script_path.read_text(encoding="utf-8")
    assert "cyclonedx-bom" in contents
    assert "SBOM written to" in contents
