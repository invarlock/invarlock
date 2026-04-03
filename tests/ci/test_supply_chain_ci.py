import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS_DIR = Path(".github/workflows")
PINNED_ACTION_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
TRANSFORMERS_LOCKFILES = (
    Path("requirements/workflows/ci-hf-py312.txt"),
    Path("requirements/workflows/ci-hf-py313.txt"),
    Path("requirements/workflows/hf-py313.txt"),
    Path("requirements/workflows/runtime-image-py312.txt"),
    Path("requirements/workflows/runtime-image-py312-aarch64.txt"),
)
TRANSFORMERS_550_HASHES = {
    "821a9ff0961abbb29eb1eb686d78df1c85929fdf213a3fe49dc6bd94f9efa944",
    "c8db656cf51c600cd8c75f06b20ef85c72e8b8ff9abc880c5d3e8bc70e0ddcbd",
}
TRANSFORMERS_550_RE = re.compile(
    r"transformers==5\.5\.0 \\\n"
    r"(?P<hash1>\s+--hash=sha256:(?P<digest1>[0-9a-f]{64}) \\\n)"
    r"(?P<hash2>\s+--hash=sha256:(?P<digest2>[0-9a-f]{64}))",
    re.MULTILINE,
)


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


def _iter_pip_install_commands(workflow: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for step in _iter_job_steps(workflow):
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for line in run.splitlines():
            stripped = line.strip()
            if "pip install" in stripped:
                commands.append(stripped)
    return commands


def _extract_transformers_550_hashes(path: Path) -> set[str]:
    match = TRANSFORMERS_550_RE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"transformers==5.5.0 stanza missing in {path}"
    return {match.group("digest1"), match.group("digest2")}


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


def test_offline_bundle_script_exists():
    script_path = Path("scripts/release/make_offline_bundle.sh")
    assert script_path.exists(), "offline bundle generator script missing"

    contents = script_path.read_text(encoding="utf-8")
    assert "release-offline-bundle-v1" in contents
    assert "Offline release bundle written to" in contents


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


def test_workflows_pin_pip_installs_by_hash() -> None:
    offenders: list[str] = []
    for workflow_path in _workflow_paths():
        workflow = _load_workflow(workflow_path)
        for command in _iter_pip_install_commands(workflow):
            if "--require-hashes" not in command:
                offenders.append(f"{workflow_path.name}: {command}")

    assert not offenders, "Unhashed workflow pip installs:\n" + "\n".join(offenders)


def test_transformers_550_hashes_match_pypi_across_requirement_locks() -> None:
    mismatches = {
        str(path): sorted(_extract_transformers_550_hashes(path))
        for path in TRANSFORMERS_LOCKFILES
        if _extract_transformers_550_hashes(path) != TRANSFORMERS_550_HASHES
    }

    assert not mismatches, (
        "transformers==5.5.0 hashes drifted from the current PyPI wheel/sdist:\n"
        + "\n".join(f"{path}: {hashes}" for path, hashes in mismatches.items())
    )


def test_scorecards_workflow_uses_least_privilege_top_level_permissions() -> None:
    workflow = _load_workflow(Path(".github/workflows/scorecards.yml"))
    assert workflow["permissions"] == {"contents": "read"}

    analysis = workflow["jobs"]["analysis"]
    assert analysis["permissions"] == {
        "id-token": "write",
        "security-events": "write",
    }


def test_release_workflow_uses_trusted_publishing():
    workflow = _load_workflow(Path(".github/workflows/release.yml"))
    triggers = workflow["on"]
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]

    assert dispatch_inputs["release_tag"]["type"] == "string"
    assert dispatch_inputs["release_tag"]["default"] == ""

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
    assert "inputs.release_tag != ''" in publish["if"]

    checkout_step = steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert (
        checkout_step["with"]["ref"]
        == "${{ github.event_name == 'push' && github.ref || inputs.release_tag }}"
    )

    dist_download_step = _find_step_by_name(steps, "Download dist artifacts")
    assert dist_download_step["with"]["path"] == "_release_dist"

    stage_step = _find_step_by_name(steps, "Stage publish distributions")
    assert "rm -rf publish-dist" in stage_step["run"]
    assert "cp _release_dist/*.whl publish-dist/" in stage_step["run"]
    assert "cp _release_dist/*.tar.gz publish-dist/" in stage_step["run"]

    attest_step = _find_step_by_uses_prefix(steps, "actions/attest-build-provenance@")
    assert attest_step["with"]["subject-path"] == "publish-dist/*"
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
    assert step_with["packages-dir"] == "publish-dist"
    assert "steps.vars.outputs.publish_repository_url" in step_with["repository-url"]
    assert step_with["skip-existing"] is True


def test_release_workflow_builds_and_bundles_release_assets():
    workflow = _load_workflow(Path(".github/workflows/release.yml"))
    build_check = workflow["jobs"]["build_check"]
    build_steps = build_check.get("steps", [])

    install_step = _find_step_by_name(build_steps, "Install build tooling")
    assert (
        install_step["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/release-security-py313.txt"
    )

    smoke_step = _find_step_by_name(build_steps, "Install smoke from wheel")
    assert ".smoke/smoke-requirements.txt" in smoke_step["run"]
    assert "--require-hashes" in smoke_step["run"]

    checkout_step = build_steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert (
        checkout_step["with"]["ref"]
        == "${{ github.event_name == 'push' && github.ref || inputs.release_tag }}"
    )

    assert _find_step_by_name(build_steps, "Generate release SBOM")
    sbom_upload = _find_step_by_name(build_steps, "Upload SBOM artifact")
    assert sbom_upload["uses"].startswith("actions/upload-artifact@")
    assert sbom_upload["with"]["name"] == "release-sbom"

    dist_upload = _find_step_by_name(build_steps, "Upload dist artifacts")
    assert dist_upload["with"]["path"] == "dist/*.whl\ndist/*.tar.gz\n"

    bundle = workflow["jobs"]["bundle_release"]
    assert bundle["permissions"] == {
        "contents": "write",
        "id-token": "write",
    }
    assert "startsWith(github.ref, 'refs/tags/v')" in bundle["if"]
    assert "inputs.target == 'pypi'" in bundle["if"]
    assert "inputs.release_tag != ''" in bundle["if"]

    bundle_steps = bundle.get("steps", [])
    checkout_step = bundle_steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert "with" not in checkout_step

    sigstore_steps = [
        step
        for step in bundle_steps
        if str(step.get("uses", "")).startswith("sigstore/gh-action-sigstore-python@")
    ]
    assert len(sigstore_steps) == 2
    assert sigstore_steps[0]["name"] == "Sign release artifacts"
    assert sigstore_steps[0]["with"]["inputs"] == "dist/*"
    assert sigstore_steps[0]["with"]["upload-signing-artifacts"] is True

    bundle_build_step = _find_step_by_name(
        bundle_steps, "Create offline verification bundle"
    )
    assert "INVARLOCK_RELEASE_TAG" in bundle_build_step["env"]
    assert "scripts/release/make_offline_bundle.sh" in bundle_build_step["run"]
    assert "--dist-dir dist" in bundle_build_step["run"]
    assert "--sbom artifacts/supply-chain/sbom.json" in bundle_build_step["run"]
    assert "--provenance-dir provenance" in bundle_build_step["run"]
    assert "--output-dir release-assets" in bundle_build_step["run"]

    assert sigstore_steps[1]["name"] == "Sign offline verification bundle"
    assert (
        sigstore_steps[1]["with"]["inputs"] == "release-assets/*-offline-bundle.tar.gz"
    )
    assert sigstore_steps[1]["with"]["upload-signing-artifacts"] is False

    release_step = _find_step_by_name(bundle_steps, "Create or update GitHub release")
    assert bundle_steps.index(sigstore_steps[1]) < bundle_steps.index(release_step)
    assert "*.sigstore.json" in release_step["run"]
    assert "cp dist/* release-assets/" in release_step["run"]
    assert 'gh release upload "$tag" release-assets/* --clobber' in release_step["run"]
    assert "release-assets/*-offline-bundle.tar.gz" not in release_step["run"]
    assert "gh release create" in release_step["run"]
    assert "gh release upload" in release_step["run"]
    assert "release-assets/*" in release_step["run"]

    testpypi_smoke = workflow["jobs"]["testpypi_smoke"]
    assert "inputs.release_tag != ''" in testpypi_smoke["if"]
    smoke_steps = testpypi_smoke.get("steps", [])
    download_step = _find_step_by_name(smoke_steps, "Download published TestPyPI wheel")
    assert "https://test.pypi.org/pypi/invarlock/" in download_step["run"]
    assert "wheelhouse/requirements.txt" in download_step["run"]
    assert (
        download_step["env"]["INVARLOCK_RELEASE_VERSION"] == "${{ inputs.release_tag }}"
    )

    install_published_step = _find_step_by_name(
        smoke_steps, "Install published wheel and smoke test"
    )
    assert "--require-hashes" in install_published_step["run"]
    assert "wheelhouse/requirements.txt" in install_published_step["run"]


def test_ci_verify_full_pins_make_to_setup_python() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    verify_full = workflow["jobs"]["verify-full"]

    env = verify_full.get("env", {})
    assert env["PYTHON"] == "python"

    steps = verify_full.get("steps", [])
    verify_step = _find_step_by_name(steps, "Full verify")
    assert "make verify" in verify_step["run"]
    assert "mkdocs build --strict" in verify_step["run"]


def test_scorecard_workflow_is_configured():
    workflow_path = Path(".github/workflows/scorecards.yml")
    assert workflow_path.exists(), "Scorecard workflow definition not found"

    workflow = _load_workflow(workflow_path)
    triggers = workflow["on"]
    assert triggers["push"]["branches"] == ["main"]
    assert "branch_protection_rule" in triggers
    assert triggers["schedule"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    analysis = workflow["jobs"]["analysis"]
    assert analysis["permissions"] == {
        "id-token": "write",
        "security-events": "write",
    }

    steps = analysis.get("steps", [])
    scorecard_step = _find_step_by_uses_prefix(steps, "ossf/scorecard-action@")
    assert scorecard_step["with"] == {
        "repo_token": "${{ secrets.SCORECARD_TOKEN || github.token }}",
        "publish_results": True,
        "results_file": "results.sarif",
        "results_format": "sarif",
    }

    upload_sarif_step = _find_step_by_uses_prefix(
        steps, "github/codeql-action/upload-sarif@"
    )
    assert (
        upload_sarif_step["uses"]
        == "github/codeql-action/upload-sarif@b20883b0cd1f46c72ae0ba6d1090936928f9fa30"
    )
    assert upload_sarif_step["with"]["sarif_file"] == "results.sarif"

    upload_artifact_step = _find_step_by_uses_prefix(steps, "actions/upload-artifact@")
    assert (
        upload_artifact_step["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )


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
    assert workflow["permissions"] == {
        "contents": "read",
        "actions": "read",
    }

    analyze = workflow["jobs"]["analyze"]
    assert analyze["permissions"] == {
        "contents": "read",
        "actions": "read",
        "security-events": "write",
    }

    init_step = _find_step_by_uses_prefix(
        analyze["steps"], "github/codeql-action/init@"
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


def test_model_evidence_workflow_is_configured() -> None:
    workflow = _load_workflow(Path(".github/workflows/model-evidence-sweep.yml"))

    triggers = workflow["on"]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "0 5 * * *"}]
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["model-evidence-sweep"]
    assert job["runs-on"] == ["self-hosted", "linux", "gpu"]
    assert job["timeout-minutes"] == 1440

    env = job["env"]
    assert env["PYTHONPATH"] == "${{ github.workspace }}/src"
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"

    steps = job["steps"]
    checkout = _find_step_by_name(steps, "Checkout repository")
    assert checkout["uses"].startswith("actions/checkout@")

    install = _find_step_by_name(steps, "Install (core + hf)")
    assert (
        install["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/ci-hf-py313.txt"
    )

    sweep = _find_step_by_name(steps, "Run shipped-model evidence sweep")
    assert "scripts/model_evidence_sweep.py" in sweep["run"]
    assert "--profile ci" in sweep["run"]
    assert "reports/model_evidence/${{ github.run_id }}" in sweep["run"]

    upload = _find_step_by_uses_prefix(steps, "actions/upload-artifact@")
    assert upload["with"]["name"] == "model-evidence-${{ github.run_id }}"
    assert upload["with"]["path"] == "reports/model_evidence/${{ github.run_id }}/"


def test_gpt2_smoke_workflow_is_configured() -> None:
    workflow = _load_workflow(Path(".github/workflows/gpt2-smoke.yml"))
    triggers = workflow["on"]

    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "0 4 * * 1"}]
    assert "push" not in triggers
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["smoke"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 60

    env = job["env"]
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert env["INVARLOCK_SMOKE_MODE"] == "attested"
    assert env["INVARLOCK_SMOKE_PROFILE"] == "dev"
    assert env["INVARLOCK_RUNTIME_IMAGE"] == "invarlock-runtime:local"

    steps = job["steps"]
    install = _find_step_by_name(steps, "Install dependencies")
    assert "pip install --require-hashes" in install["run"]

    runtime_image = _find_step_by_name(steps, "Build runtime image")
    assert "make runtime-image" in runtime_image["run"]

    smoke = _find_step_by_name(steps, "Run GPT-2 smoke campaign")
    assert "scripts/run_gpt2_smoke_campaign.sh" in smoke["run"]


def test_tiny_attested_smoke_workflow_is_configured() -> None:
    workflow = _load_workflow(Path(".github/workflows/tiny-attested-smoke.yml"))
    triggers = workflow["on"]

    assert triggers["push"]["branches"] == ["staging/next"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["smoke"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 45

    env = job["env"]
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert env["INVARLOCK_SMOKE_MODE"] == "attested"
    assert env["INVARLOCK_SMOKE_PROFILE"] == "dev"
    assert env["INVARLOCK_RUNTIME_IMAGE"] == "invarlock-runtime:local"

    steps = job["steps"]
    install = _find_step_by_name(steps, "Install dependencies")
    assert "pip install --require-hashes" in install["run"]

    runtime_image = _find_step_by_name(steps, "Build runtime image")
    assert "make runtime-image" in runtime_image["run"]

    smoke = _find_step_by_name(steps, "Run tiny attested smoke campaign")
    assert "scripts/run_tiny_attested_smoke.sh" in smoke["run"]
