import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS_DIR = Path(".github/workflows")
PINNED_ACTION_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _load_workflow(path: Path) -> dict[str, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _iter_job_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = workflow.get("jobs", {})
    steps: list[dict[str, Any]] = []
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if isinstance(step, dict):
                steps.append(step)
    return steps


def _requires_job_permissions(step: dict[str, Any]) -> bool:
    uses = str(step.get("uses", ""))
    return any(
        marker in uses
        for marker in (
            "actions/github-script",
            "pypa/gh-action-pypi-publish",
            "actions/attest-build-provenance",
            "github/codeql-action/upload-sarif",
            "nwtgck/actions-netlify",
            "ossf/scorecard-action",
            "sigstore/gh-action-sigstore-python",
        )
    )


def _find_step_by_name(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def _find_step_by_uses_prefix(
    steps: list[dict[str, Any]], uses_prefix: str
) -> dict[str, Any]:
    return next(
        step for step in steps if str(step.get("uses", "")).startswith(uses_prefix)
    )


def test_supply_chain_job_configured():
    """Test supply-chain job includes core security checks.

    Note: gitleaks secret scanning was removed to reduce CI cost.
    CodeQL workflow handles security scanning instead.
    """
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow definition not found"

    workflow = _load_workflow(workflow_path)
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


def test_workflows_pin_github_actions_to_full_shas():
    offenders: list[str] = []
    for workflow_path in _workflow_paths():
        workflow = _load_workflow(workflow_path)
        for step in _iter_job_steps(workflow):
            uses = step.get("uses")
            if not uses or str(uses).startswith("./"):
                continue
            if not PINNED_ACTION_RE.match(str(uses)):
                offenders.append(f"{workflow_path.name}: {uses}")

    assert not offenders, "Unpinned workflow actions:\n" + "\n".join(offenders)


def test_workflows_declare_explicit_permissions():
    missing_top_level: list[str] = []
    missing_job_level: list[str] = []

    for workflow_path in _workflow_paths():
        workflow = _load_workflow(workflow_path)
        if "permissions" not in workflow:
            missing_top_level.append(workflow_path.name)

        jobs = workflow.get("jobs", {})
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            needs_job_permissions = any(
                _requires_job_permissions(step) for step in job.get("steps", [])
            )
            if needs_job_permissions and "permissions" not in job:
                missing_job_level.append(f"{workflow_path.name}:{job_name}")

    assert not missing_top_level, "Missing workflow permissions:\n" + "\n".join(
        missing_top_level
    )
    assert not missing_job_level, "Missing job permissions:\n" + "\n".join(
        missing_job_level
    )


def test_release_workflow_uses_trusted_publishing():
    workflow = _load_workflow(Path(".github/workflows/release.yml"))
    publish = workflow["jobs"]["publish"]
    permissions = publish.get("permissions", {})

    assert permissions == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert publish.get("environment") == (
        "${{ github.event_name == 'push' && 'pypi' || inputs.target }}"
    )

    steps = publish.get("steps", [])
    assert "startsWith(github.ref, 'refs/tags/v')" in publish["if"]
    assert "github.event_name == 'push'" in publish["if"]
    assert "inputs.publish == true" in publish["if"]

    attest_step = _find_step_by_uses_prefix(steps, "actions/attest-build-provenance@")
    assert attest_step["with"]["subject-path"] == "dist/*"
    assert attest_step["id"] == "attest_release"

    provenance_step = _find_step_by_name(steps, "Upload provenance bundle")
    assert provenance_step["uses"].startswith("actions/upload-artifact@")
    assert provenance_step["with"]["name"] == "release-provenance"
    assert (
        provenance_step["with"]["path"]
        == "${{ steps.attest_release.outputs.bundle-path }}"
    )

    publish_step = _find_step_by_uses_prefix(steps, "pypa/gh-action-pypi-publish@")
    step_with = publish_step.get("with", {})
    assert "user" not in step_with
    assert "password" not in step_with
    assert step_with["packages-dir"] == "dist"
    assert "steps.vars.outputs.publish_repository_url" in step_with["repository-url"]


def test_release_workflow_builds_and_bundles_release_assets():
    workflow = _load_workflow(Path(".github/workflows/release.yml"))
    build_check = workflow["jobs"]["build_check"]
    build_steps = build_check.get("steps", [])

    install_step = _find_step_by_name(build_steps, "Install build tooling")
    assert ".[release-ci,security-ci]" in install_step["run"]

    assert _find_step_by_name(build_steps, "Generate release SBOM")
    sbom_upload = _find_step_by_name(build_steps, "Upload SBOM artifact")
    assert sbom_upload["uses"].startswith("actions/upload-artifact@")
    assert sbom_upload["with"]["name"] == "release-sbom"

    bundle = workflow["jobs"]["bundle_release"]
    assert bundle["permissions"] == {
        "contents": "write",
        "id-token": "write",
    }
    assert "startsWith(github.ref, 'refs/tags/v')" in bundle["if"]
    assert "inputs.target == 'pypi'" in bundle["if"]

    bundle_steps = bundle.get("steps", [])
    sigstore_step = _find_step_by_uses_prefix(
        bundle_steps, "sigstore/gh-action-sigstore-python@"
    )
    assert sigstore_step["with"]["inputs"] == "dist/*"
    assert sigstore_step["with"]["upload-signing-artifacts"] is True

    release_step = _find_step_by_name(bundle_steps, "Create or update GitHub release")
    assert "gh release create" in release_step["run"]
    assert "gh release upload" in release_step["run"]
    assert "release-assets/*" in release_step["run"]


def test_scorecard_workflow_is_configured():
    workflow_path = Path(".github/workflows/scorecards.yml")
    assert workflow_path.exists(), "Scorecard workflow definition not found"

    workflow = _load_workflow(workflow_path)
    triggers = workflow["on"]
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["schedule"]
    assert "workflow_dispatch" in triggers

    analysis = workflow["jobs"]["analysis"]
    assert analysis["permissions"] == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
        "security-events": "write",
    }

    steps = analysis.get("steps", [])
    scorecard_step = _find_step_by_uses_prefix(steps, "ossf/scorecard-action@")
    assert scorecard_step["with"] == {
        "publish_results": True,
        "results_file": "results.sarif",
        "results_format": "sarif",
    }

    upload_sarif_step = _find_step_by_uses_prefix(
        steps, "github/codeql-action/upload-sarif@"
    )
    assert upload_sarif_step["with"]["sarif_file"] == "results.sarif"


def test_docs_workflow_enforces_docs_lint_on_main_and_staging() -> None:
    workflow = _load_workflow(Path(".github/workflows/docs-ci.yml"))
    triggers = workflow["on"]

    assert triggers["push"]["branches"] == ["main", "staging/next"]
    assert triggers["pull_request"]["branches"] == ["main", "develop", "staging/next"]

    expected_paths = [
        "docs/**",
        "README.md",
        "CONTRIBUTING.md",
        "mkdocs.yml",
        ".github/workflows/docs-ci.yml",
    ]
    assert triggers["push"]["paths"] == expected_paths
    assert triggers["pull_request"]["paths"] == expected_paths

    steps = workflow["jobs"]["docs-validate"]["steps"]
    markdown_step = _find_step_by_name(steps, "Lint markdown")
    spell_step = _find_step_by_name(steps, "Spell check")

    assert markdown_step["run"] == "python scripts/docs_lint.py --markdown"
    assert "continue-on-error" not in markdown_step
    assert spell_step["run"] == "python scripts/docs_lint.py --spell"
    assert "continue-on-error" not in spell_step


def test_readme_exposes_scorecard_badge():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "OpenSSF Scorecard" in readme
    assert (
        "https://api.scorecard.dev/projects/github.com/invarlock/invarlock/badge"
        in readme
    )
    assert "https://scorecard.dev/viewer/?uri=github.com/invarlock/invarlock" in readme


def test_codeql_workflow_uses_repo_config():
    workflow = _load_workflow(Path(".github/workflows/codeql.yml"))
    init_step = _find_step_by_uses_prefix(
        workflow["jobs"]["analyze"]["steps"], "github/codeql-action/init@"
    )
    assert init_step["with"]["config-file"] == ".github/codeql/codeql-config.yml"


def test_codeql_config_scopes_analysis_to_shipped_python():
    config_path = Path(".github/codeql/codeql-config.yml")
    assert config_path.exists(), "CodeQL config file missing"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["paths"] == ["src/invarlock"]

    excluded_ids = set(config["query-filters"][0]["exclude"]["id"])
    assert "py/empty-except" in excluded_ids
    assert "py/unused-local-variable" in excluded_ids
