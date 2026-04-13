import json
import re
from datetime import date
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

    The scheduled/tag CI lane keeps the tool-environment SBOM + pip-audit
    backstop; the PR-time workflow carries the install-surface SBOM and the
    gitleaks artifact scan.
    """
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow definition not found"

    workflow = _load_workflow(workflow_path)
    jobs = workflow.get("jobs", {})
    assert "supply-chain" in jobs, "Supply-chain job missing from CI workflow"

    supply_job = jobs["supply-chain"]
    steps = supply_job.get("steps", [])
    step_names = [step.get("name") for step in steps if isinstance(step, dict)]

    assert "Generate tool-environment SBOM" in step_names
    assert "Run pip-audit" in step_names

    audit_commands = [step.get("run", "") for step in steps if isinstance(step, dict)]
    assert any("scripts/security/run_pip_audit.py" in cmd for cmd in audit_commands)
    assert any("--scope tool-environment" in cmd for cmd in audit_commands)


def test_repo_hygiene_checks_uv_lock_sync() -> None:
    workflow = _load_workflow(Path(".github/workflows/repo-hygiene.yml"))
    jobs = workflow["jobs"]

    assert "lockfile-sync" in jobs
    job = jobs["lockfile-sync"]
    steps = job["steps"]
    step_names = [step.get("name") for step in steps if isinstance(step, dict)]

    assert step_names == ["Checkout repository", "Set up uv", "Check uv.lock sync"]

    checkout_step = steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert checkout_step["with"]["fetch-depth"] == 0

    uv_step = _find_step_by_name(steps, "Set up uv")
    assert (
        uv_step["uses"] == "astral-sh/setup-uv@681c641aba71e4a1c380be3ab5e12ad51f415867"
    )
    assert uv_step["with"]["version"] == "0.10.10"

    check_step = _find_step_by_name(steps, "Check uv.lock sync")
    assert check_step["run"] == "make lock-sync"


def test_pr_supply_chain_workflow_is_configured() -> None:
    workflow = _load_workflow(Path(".github/workflows/supply-chain-pr.yml"))
    triggers = workflow["on"]

    assert triggers["pull_request"]["branches"] == ["main", "staging/next"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["scan"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15

    steps = job["steps"]
    step_names = [step.get("name") for step in steps if isinstance(step, dict)]

    assert step_names == [
        "Checkout repository",
        "Set up Python",
        "Install supply-chain tools",
        "Install gitleaks",
        "Build release wheel",
        "Create install-surface venv",
        "Run pip-audit",
        "Generate install-surface SBOM",
        "Create HF surface venv",
        "Run HF surface pip-audit",
        "Create advanced surface venv",
        "Run advanced surface pip-audit",
        "Run gitleaks history scan",
        "Upload supply-chain artifacts",
        "Fail on secret findings",
    ]

    install_step = _find_step_by_name(steps, "Install supply-chain tools")
    assert (
        install_step["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/release-security-py313.txt"
    )

    checkout_step = steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert checkout_step["with"]["fetch-depth"] == 0

    gitleaks_install = _find_step_by_name(steps, "Install gitleaks")
    assert (
        "go install github.com/zricethezav/gitleaks/v8@v8.30.0"
        in gitleaks_install["run"]
    )

    build_step = _find_step_by_name(steps, "Build release wheel")
    assert "rm -rf build dist" in build_step["run"]
    assert "python -m build" in build_step["run"]

    venv_step = _find_step_by_name(steps, "Create install-surface venv")
    assert venv_step["id"] == "install_surface"
    assert "python -m venv .artifact-venv" in venv_step["run"]
    assert "python -m pip install dist/*.whl" in venv_step["run"]
    assert "site_packages=" in venv_step["run"]

    sbom_step = _find_step_by_name(steps, "Generate install-surface SBOM")
    assert (
        "scripts/generate_sbom.sh --scope install-surface --python" in sbom_step["run"]
    )
    assert "artifacts/supply-chain/sbom.json" in sbom_step["run"]

    audit_step = _find_step_by_name(steps, "Run pip-audit")
    assert "python scripts/security/run_pip_audit.py --path" in audit_step["run"]
    assert "${{ steps.install_surface.outputs.site_packages }}" in audit_step["run"]

    hf_venv_step = _find_step_by_name(steps, "Create HF surface venv")
    assert hf_venv_step["id"] == "hf_surface"
    assert "python -m pip install dist/*.whl --no-deps" in hf_venv_step["run"]
    assert (
        "python -m pip install --require-hashes -r requirements/workflows/hf-py313.txt"
        in hf_venv_step["run"]
    )

    hf_audit_step = _find_step_by_name(steps, "Run HF surface pip-audit")
    assert "python scripts/security/run_pip_audit.py --path" in hf_audit_step["run"]
    assert "${{ steps.hf_surface.outputs.site_packages }}" in hf_audit_step["run"]

    advanced_venv_step = _find_step_by_name(steps, "Create advanced surface venv")
    assert advanced_venv_step["id"] == "advanced_surface"
    assert "python -m pip install dist/*.whl --no-deps" in advanced_venv_step["run"]
    assert (
        "python -m pip install --require-hashes -r requirements/workflows/advanced-py313.txt"
        in advanced_venv_step["run"]
    )

    advanced_audit_step = _find_step_by_name(steps, "Run advanced surface pip-audit")
    assert (
        "python scripts/security/run_pip_audit.py --path" in advanced_audit_step["run"]
    )
    assert (
        "${{ steps.advanced_surface.outputs.site_packages }}"
        in advanced_audit_step["run"]
    )

    secret_scan_step = _find_step_by_name(steps, "Run gitleaks history scan")
    assert "gitleaks git ." in secret_scan_step["run"]
    assert "--report-format json" in secret_scan_step["run"]
    assert "--report-format sarif" in secret_scan_step["run"]
    assert "artifacts/supply-chain/gitleaks.json" in secret_scan_step["run"]
    assert "artifacts/supply-chain/gitleaks.sarif" in secret_scan_step["run"]

    upload_step = _find_step_by_name(steps, "Upload supply-chain artifacts")
    assert upload_step["uses"].startswith("actions/upload-artifact@")
    assert upload_step["with"]["name"] == "supply-chain-pr-artifacts"
    assert "artifacts/supply-chain/sbom.json" in upload_step["with"]["path"]
    assert "artifacts/supply-chain/gitleaks.json" in upload_step["with"]["path"]
    assert "artifacts/supply-chain/gitleaks.sarif" in upload_step["with"]["path"]

    fail_step = _find_step_by_name(steps, "Fail on secret findings")
    assert "gitleaks detected secrets" in fail_step["run"]


def test_generate_sbom_script_exists():
    script_path = Path("scripts/generate_sbom.sh")
    assert script_path.exists(), "SBOM generator script missing"

    contents = script_path.read_text(encoding="utf-8")
    assert "cyclonedx-bom" in contents
    assert "--scope install-surface" in contents
    assert "SBOM written to" in contents


def test_pip_audit_allowlist_is_owned_and_time_boxed() -> None:
    allowlist_path = Path("scripts/security/pip_audit_allowlist.json")
    payload = json.loads(allowlist_path.read_text(encoding="utf-8"))

    assert payload["owner"] == "security-maintainers"
    assert payload["entries"]

    entry = payload["entries"][0]
    assert entry["advisory"] == "GHSA-4xh5-x5gv-qwph"
    assert entry["owner"] == "security-maintainers"
    expires = date.fromisoformat(entry["expires"])
    assert 0 <= (expires - date.today()).days <= 30
    assert entry["tracking_issue"] == "https://github.com/pypa/pip/issues/13607"
    assert "reason" in entry


def test_codeowners_protect_security_control_surfaces() -> None:
    codeowners = Path(".github/CODEOWNERS").read_text(encoding="utf-8")

    for required_entry in (
        ".github/workflows/codeql.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/dependabot-main-guard.yml",
        ".github/workflows/release.yml",
        ".github/workflows/supply-chain-pr.yml",
        ".github/codeql/",
        ".github/dependabot.yml",
        "docs/security/",
        "requirements/workflows/*security*.txt",
        "scripts/security/",
    ):
        assert required_entry in codeowners


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
            if workflow_path.name in {"release.yml", "supply-chain-pr.yml"} and (
                "python -m pip install dist/*.whl" in command
                or "python -m pip install wheelhouse/*.whl" in command
            ):
                continue
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

    resolve = workflow["jobs"]["resolve_release_ref"]
    assert resolve["permissions"] == {"contents": "read"}

    resolve_step = _find_step_by_name(resolve["steps"], "Resolve release ref")
    assert "release_tag is required for manual release dispatch" in resolve_step["run"]
    assert "release_tag must start with v" in resolve_step["run"]
    assert "git ls-remote --tags" in resolve_step["run"]

    resolve_outputs = resolve["outputs"]
    assert (
        resolve_outputs["release_tag"]
        == "${{ steps.resolve_release_ref.outputs.release_tag }}"
    )
    assert (
        resolve_outputs["release_sha"]
        == "${{ steps.resolve_release_ref.outputs.release_sha }}"
    )

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
    assert publish["needs"] == ["build_check", "resolve_release_ref"]

    steps = publish.get("steps", [])
    assert "startsWith(github.ref, 'refs/tags/v')" in publish["if"]
    assert "github.event_name == 'push'" in publish["if"]
    assert "inputs.publish == true" in publish["if"]

    checkout_step = steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert (
        checkout_step["with"]["ref"]
        == "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    assert checkout_step["with"]["fetch-depth"] == 0

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
    resolve = workflow["jobs"]["resolve_release_ref"]
    assert resolve["steps"][0]["name"] == "Resolve release ref"

    resolve_step = resolve["steps"][0]
    assert "git ls-remote --tags" in resolve_step["run"]
    assert "refs/tags/${tag}^{}" in resolve_step["run"]
    assert "release_tag must start with v" in resolve_step["run"]

    build_check = workflow["jobs"]["build_check"]
    assert build_check["needs"] == "resolve_release_ref"
    build_steps = build_check.get("steps", [])

    install_step = _find_step_by_name(build_steps, "Install build tooling")
    assert (
        install_step["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/release-security-py313.txt"
    )

    gitleaks_install = _find_step_by_name(build_steps, "Install gitleaks")
    assert (
        "go install github.com/zricethezav/gitleaks/v8@v8.30.0"
        in gitleaks_install["run"]
    )

    gitleaks_scan = _find_step_by_name(build_steps, "Run gitleaks history scan")
    assert "gitleaks git ." in gitleaks_scan["run"]
    assert "artifacts/supply-chain/gitleaks.json" in gitleaks_scan["run"]
    assert "artifacts/supply-chain/gitleaks.sarif" in gitleaks_scan["run"]
    assert "--report-format json" in gitleaks_scan["run"]
    assert "--report-format sarif" in gitleaks_scan["run"]

    smoke_step = _find_step_by_name(build_steps, "Install smoke from wheel")
    assert "python -m pip install dist/*.whl" in smoke_step["run"]
    assert "invarlock --help" in smoke_step["run"]
    assert 'python -c "import invarlock.cli.app"' in smoke_step["run"]

    checkout_step = build_steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert (
        checkout_step["with"]["ref"]
        == "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    assert checkout_step["with"]["fetch-depth"] == 0

    install_surface_step = _find_step_by_name(
        build_steps, "Create release install-surface venv"
    )
    assert install_surface_step["id"] == "install_surface"
    assert "python -m venv .artifact-venv" in install_surface_step["run"]
    assert "python -m pip install dist/*.whl" in install_surface_step["run"]

    audit_step = _find_step_by_name(build_steps, "Run release pip-audit")
    assert "python scripts/security/run_pip_audit.py --path" in audit_step["run"]
    assert "${{ steps.install_surface.outputs.site_packages }}" in audit_step["run"]

    sbom_step = _find_step_by_name(build_steps, "Generate release install-surface SBOM")
    assert "--scope install-surface --python" in sbom_step["run"]
    assert "artifacts/supply-chain/sbom.json" in sbom_step["run"]

    sbom_upload = _find_step_by_name(build_steps, "Upload SBOM artifact")
    assert sbom_upload["uses"].startswith("actions/upload-artifact@")
    assert sbom_upload["with"]["name"] == "release-sbom"

    gitleaks_upload = _find_step_by_name(build_steps, "Upload gitleaks artifacts")
    assert gitleaks_upload["uses"].startswith("actions/upload-artifact@")
    assert gitleaks_upload["with"]["name"] == "release-gitleaks"
    assert "artifacts/supply-chain/gitleaks.json" in gitleaks_upload["with"]["path"]
    assert "artifacts/supply-chain/gitleaks.sarif" in gitleaks_upload["with"]["path"]

    fail_step = _find_step_by_name(build_steps, "Fail on secret findings")
    assert "gitleaks detected secrets" in fail_step["run"]

    dist_upload = _find_step_by_name(build_steps, "Upload dist artifacts")
    assert dist_upload["with"]["path"] == "dist/*.whl\ndist/*.tar.gz\n"

    bundle = workflow["jobs"]["bundle_release"]
    assert bundle["needs"] == ["publish", "resolve_release_ref"]
    assert bundle["permissions"] == {
        "contents": "write",
        "id-token": "write",
    }
    assert "startsWith(github.ref, 'refs/tags/v')" in bundle["if"]
    assert "inputs.target == 'pypi'" in bundle["if"]

    bundle_steps = bundle.get("steps", [])
    checkout_step = bundle_steps[0]
    assert checkout_step["uses"].startswith("actions/checkout@")
    assert (
        checkout_step["with"]["ref"]
        == "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    assert checkout_step["with"]["fetch-depth"] == 0

    sigstore_steps = [
        step
        for step in bundle_steps
        if str(step.get("uses", "")).startswith("sigstore/gh-action-sigstore-python@")
    ]
    assert len(sigstore_steps) == 3
    assert sigstore_steps[0]["name"] == "Sign release artifacts"
    assert sigstore_steps[0]["with"]["inputs"] == "dist/*"
    assert sigstore_steps[0]["with"]["upload-signing-artifacts"] is True

    bundle_build_step = _find_step_by_name(
        bundle_steps, "Create offline verification bundle"
    )
    assert (
        bundle_build_step["env"]["INVARLOCK_RELEASE_TAG"]
        == "${{ needs.resolve_release_ref.outputs.release_tag }}"
    )
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

    public_bundle_step = _find_step_by_name(
        bundle_steps, "Create public contract bundle"
    )
    assert public_bundle_step["env"]["INVARLOCK_RELEASE_TAG"] == (
        "${{ needs.resolve_release_ref.outputs.release_tag }}"
    )
    assert public_bundle_step["env"]["INVARLOCK_RELEASE_SHA"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    assert "scripts/release/make_public_contract_bundle.py" in public_bundle_step["run"]
    assert "--contracts-dir contracts" in public_bundle_step["run"]
    assert "--runtime-dir src/invarlock/_data/runtime" in public_bundle_step["run"]
    assert "--output-dir release-assets" in public_bundle_step["run"]

    assert sigstore_steps[2]["name"] == "Sign public contract bundle"
    assert (
        sigstore_steps[2]["with"]["inputs"]
        == "release-assets/*-public-contract-bundle.tar.gz"
    )
    assert sigstore_steps[2]["with"]["upload-signing-artifacts"] is False

    release_step = _find_step_by_name(bundle_steps, "Create or update GitHub release")
    assert bundle_steps.index(sigstore_steps[2]) < bundle_steps.index(release_step)
    assert "*.sigstore.json" in release_step["run"]
    assert "cp dist/* release-assets/" in release_step["run"]
    assert 'gh release upload "$tag" release-assets/* --clobber' in release_step["run"]
    assert "release-assets/*-offline-bundle.tar.gz" not in release_step["run"]
    assert "release-assets/*-public-contract-bundle.tar.gz" not in release_step["run"]
    assert "gh release create" in release_step["run"]
    assert "gh release upload" in release_step["run"]
    assert "release-assets/*" in release_step["run"]

    testpypi_smoke = workflow["jobs"]["testpypi_smoke"]
    assert testpypi_smoke["needs"] == ["publish", "resolve_release_ref"]
    assert "inputs.target == 'testpypi'" in testpypi_smoke["if"]
    smoke_steps = testpypi_smoke.get("steps", [])
    download_step = _find_step_by_name(smoke_steps, "Download published TestPyPI wheel")
    assert "https://test.pypi.org/pypi/invarlock/" in download_step["run"]
    assert "wheelhouse/requirements.txt" in download_step["run"]
    assert (
        download_step["env"]["INVARLOCK_RELEASE_VERSION"]
        == "${{ needs.resolve_release_ref.outputs.release_tag }}"
    )

    install_published_step = _find_step_by_name(
        smoke_steps, "Install published wheel and smoke test"
    )
    assert "python -m pip install wheelhouse/*.whl" in install_published_step["run"]
    assert "invarlock --help" in install_published_step["run"]
    assert 'python -c "import invarlock.cli.app"' in install_published_step["run"]


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
    assert "base, `hf`, and `advanced` shipped dependency surfaces" in workflows_doc
    assert "gitleaks" in workflows_doc
    assert "scripts/security/run_pip_audit.py" in allowlist_doc
    assert "scripts/security/pip_audit_allowlist.json" in allowlist_doc
    assert "installed release surface" in release_doc
    assert "resolved commit SHA" in release_doc
    assert "public-contract-bundle" in release_doc
    assert "gitleaks" in architecture_doc
    assert "installed-artifact environment" in architecture_doc


def test_ci_verify_full_pins_make_to_setup_python() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    verify_full = workflow["jobs"]["verify-full"]

    env = verify_full.get("env", {})
    assert env["PYTHON"] == "python"

    steps = verify_full.get("steps", [])
    verify_step = _find_step_by_name(steps, "Full verify")
    assert "make verify" in verify_step["run"]
    assert "mkdocs build --strict" in verify_step["run"]


def test_ci_pr_assurance_gates_are_required_jobs() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    triggers = workflow["on"]

    assert "paths-ignore" not in triggers["push"]
    assert triggers["pull_request"] is None

    coverage_job = workflow["jobs"]["coverage-enforce"]
    assert coverage_job["runs-on"] == "ubuntu-latest"
    assert coverage_job["env"]["PYTHON"] == "python"
    coverage_steps = coverage_job["steps"]

    coverage_install = _find_step_by_name(
        coverage_steps, "Install (hf + assurance tools)"
    )
    assert (
        coverage_install["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/assurance-ci-py313.txt"
    )

    coverage_step = _find_step_by_name(coverage_steps, "Enforce coverage thresholds")
    assert coverage_step["run"] == "make coverage-enforce"

    typed_job = workflow["jobs"]["typed-surface"]
    assert typed_job["runs-on"] == "ubuntu-latest"
    assert typed_job["env"]["PYTHON"] == "python"
    typed_steps = typed_job["steps"]

    typed_install = _find_step_by_name(typed_steps, "Install (hf + assurance tools)")
    assert (
        typed_install["run"]
        == "python -m pip install --require-hashes -r requirements/workflows/assurance-ci-py313.txt"
    )

    typed_step = _find_step_by_name(typed_steps, "Run typed-surface mypy")
    assert typed_step["run"] == "make mypy-typed-surface"


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
    assert [step.get("name") for step in steps] == [
        "Checkout repository",
        "Run OpenSSF Scorecard analysis",
    ]
    assert all("uses" in step for step in steps)

    scorecard_step = _find_step_by_uses_prefix(steps, "ossf/scorecard-action@")
    assert scorecard_step["with"] == {
        "repo_token": "${{ secrets.SCORECARD_TOKEN || github.token }}",
        "publish_results": True,
        "results_file": "results.sarif",
        "results_format": "sarif",
    }


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

    link_step = _find_step_by_name(
        workflow["jobs"]["check-external-links"]["steps"],
        "Check external links",
    )
    assert link_step["run"] == "linkchecker --check-extern docs/"
    assert "continue-on-error" not in link_step


def test_repo_hygiene_covers_staging_next_and_renames() -> None:
    workflow = _load_workflow(Path(".github/workflows/repo-hygiene.yml"))
    triggers = workflow["on"]

    assert triggers["pull_request"]["branches"] == ["main", "staging/next"]

    artifact_step = _find_step_by_name(
        workflow["jobs"]["no-generated-artifacts"]["steps"],
        "Detect forbidden files in PR diff",
    )
    large_file_step = _find_step_by_name(
        workflow["jobs"]["large-files"]["steps"],
        "Prevent >10MB files in PR diff",
    )
    assert "--diff-filter=ACMR" in artifact_step["run"]
    assert "--diff-filter=ACMR" in large_file_step["run"]


def test_readme_exposes_scorecard_badge():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "OpenSSF Scorecard" in readme
    assert (
        "https://api.scorecard.dev/projects/github.com/invarlock/invarlock/badge"
        in readme
    )
    assert "https://scorecard.dev/viewer/?uri=github.com/invarlock/invarlock" in readme


def test_readme_mentions_probes_extra() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "invarlock[probes]" in readme


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
    autobuild_step = _find_step_by_uses_prefix(
        analyze["steps"], "github/codeql-action/autobuild@"
    )
    analyze_step = _find_step_by_uses_prefix(
        analyze["steps"], "github/codeql-action/analyze@"
    )
    expected_pin = "c10b8064de6f491fea524254123dbe5e09572f13"

    assert init_step["uses"] == f"github/codeql-action/init@{expected_pin}"
    assert autobuild_step["uses"] == f"github/codeql-action/autobuild@{expected_pin}"
    assert analyze_step["uses"] == f"github/codeql-action/analyze@{expected_pin}"
    assert init_step["with"]["config-file"] == ".github/codeql/codeql-config.yml"
    assert "continue-on-error" not in analyze_step


def test_dependabot_tracks_codeql_action_updates() -> None:
    config = yaml.safe_load(Path(".github/dependabot.yml").read_text(encoding="utf-8"))

    actions_update = next(
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "github-actions"
    )
    ignored = {
        entry["dependency-name"]
        for entry in actions_update.get("ignore", [])
        if isinstance(entry, dict) and "dependency-name" in entry
    }

    assert "github/codeql-action" not in ignored


def test_codeql_config_scopes_analysis_to_shipped_python():
    config_path = Path(".github/codeql/codeql-config.yml")
    assert config_path.exists(), "CodeQL config file missing"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["paths"] == ["src/invarlock", "scripts"]

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


def test_ci_hf_lockfiles_include_hypothesis_for_property_tests() -> None:
    for path in (
        Path("requirements/workflows/ci-hf-py312.txt"),
        Path("requirements/workflows/ci-hf-py313.txt"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "hypothesis==" in text, f"hypothesis missing from {path}"


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
