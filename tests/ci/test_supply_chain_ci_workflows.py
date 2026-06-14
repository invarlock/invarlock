import json
import re
import subprocess
import tomllib
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
TRANSFORMERS_512_HASHES = {
    "500be9eb644ede81c3103eee7687fc36d05dd75d1c76686c3820b26396fe7c7c",
    "f0cf42ae1464c2eb41e7e0e66d7fd4b66145f48af17093b4cc0b2e9781faa7f4",
}
TRANSFORMERS_512_RE = re.compile(
    r"transformers==5\.12\.0 \\\n"
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


def _extract_transformers_512_hashes(path: Path) -> set[str]:
    match = TRANSFORMERS_512_RE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"transformers==5.12.0 stanza missing in {path}"
    return {match.group("digest1"), match.group("digest2")}


def test_precommit_workflow_uses_named_check_context() -> None:
    workflow = _load_workflow(Path(".github/workflows/pre-commit.yml"))

    assert workflow["jobs"]["run"]["name"] == "pre-commit"


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
        uv_step["uses"] == "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
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
        "Remove install-surface venv",
        "Create HF surface venv",
        "Run HF surface pip-audit",
        "Remove HF surface venv",
        "Create advanced surface venv",
        "Run advanced surface pip-audit",
        "Run gitleaks PR file scan",
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
    assert (
        "python -m pip install --upgrade --require-hashes -r requirements/workflows/pip-bootstrap-py313.txt"
        in venv_step["run"]
    )
    assert "python -m pip install dist/*.whl" in venv_step["run"]
    assert "site_packages=" in venv_step["run"]

    sbom_step = _find_step_by_name(steps, "Generate install-surface SBOM")
    assert (
        "scripts/security/generate_sbom.sh --scope install-surface --python"
        in sbom_step["run"]
    )
    assert "artifacts/supply-chain/sbom.json" in sbom_step["run"]

    install_cleanup_step = _find_step_by_name(steps, "Remove install-surface venv")
    assert "rm -rf .artifact-venv" in install_cleanup_step["run"]

    audit_step = _find_step_by_name(steps, "Run pip-audit")
    assert "python scripts/security/run_pip_audit.py --path" in audit_step["run"]
    assert "${{ steps.install_surface.outputs.site_packages }}" in audit_step["run"]

    hf_venv_step = _find_step_by_name(steps, "Create HF surface venv")
    assert hf_venv_step["id"] == "hf_surface"
    assert (
        "python -m pip install --upgrade --require-hashes -r requirements/workflows/pip-bootstrap-py313.txt"
        in hf_venv_step["run"]
    )
    assert "python -m pip install dist/*.whl --no-deps" in hf_venv_step["run"]
    assert (
        "python -m pip install --require-hashes -r requirements/workflows/hf-py313.txt"
        in hf_venv_step["run"]
    )

    hf_audit_step = _find_step_by_name(steps, "Run HF surface pip-audit")
    assert "python scripts/security/run_pip_audit.py --path" in hf_audit_step["run"]
    assert "${{ steps.hf_surface.outputs.site_packages }}" in hf_audit_step["run"]

    hf_cleanup_step = _find_step_by_name(steps, "Remove HF surface venv")
    assert "rm -rf .hf-venv" in hf_cleanup_step["run"]

    advanced_venv_step = _find_step_by_name(steps, "Create advanced surface venv")
    assert advanced_venv_step["id"] == "advanced_surface"
    assert (
        "python -m pip install --upgrade --require-hashes -r requirements/workflows/pip-bootstrap-py313.txt"
        in advanced_venv_step["run"]
    )
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

    secret_scan_step = _find_step_by_name(steps, "Run gitleaks PR file scan")
    assert (
        secret_scan_step["env"]["PR_BASE_SHA"]
        == "${{ github.event.pull_request.base.sha }}"
    )
    assert (
        secret_scan_step["env"]["PR_HEAD_SHA"]
        == "${{ github.event.pull_request.head.sha }}"
    )
    assert (
        'git diff --name-only --diff-filter=ACMRT "${PR_BASE_SHA}" "${PR_HEAD_SHA}"'
        in secret_scan_step["run"]
    )
    assert 'scan_root="artifacts/supply-chain/pr-files"' in secret_scan_step["run"]
    assert 'gitleaks dir "${scan_root}"' in secret_scan_step["run"]
    assert 'scan_range="${PR_BASE_SHA}..${PR_HEAD_SHA}"' in secret_scan_step["run"]
    assert 'scan_range="-1 HEAD"' in secret_scan_step["run"]
    assert "scanned_file_count=" in secret_scan_step["run"]
    assert "--report-format json" in secret_scan_step["run"]
    assert "--report-format sarif" in secret_scan_step["run"]
    assert "artifacts/supply-chain/gitleaks.changed-files" in secret_scan_step["run"]
    assert "artifacts/supply-chain/gitleaks.json" in secret_scan_step["run"]
    assert "artifacts/supply-chain/gitleaks.sarif" in secret_scan_step["run"]

    upload_step = _find_step_by_name(steps, "Upload supply-chain artifacts")
    assert upload_step["uses"].startswith("actions/upload-artifact@")
    assert upload_step["with"]["name"] == "supply-chain-pr-artifacts"
    assert "artifacts/supply-chain/sbom.json" in upload_step["with"]["path"]
    assert (
        "artifacts/supply-chain/gitleaks.changed-files" in upload_step["with"]["path"]
    )
    assert "artifacts/supply-chain/gitleaks.json" in upload_step["with"]["path"]
    assert "artifacts/supply-chain/gitleaks.sarif" in upload_step["with"]["path"]

    fail_step = _find_step_by_name(steps, "Fail on secret findings")
    assert "gitleaks scan did not publish an exit code" in fail_step["run"]
    assert "gitleaks detected secrets" in fail_step["run"]


def test_generate_sbom_script_exists():
    script_path = Path("scripts/security/generate_sbom.sh")
    assert script_path.exists(), "SBOM generator script missing"

    contents = script_path.read_text(encoding="utf-8")
    assert "cyclonedx-bom" in contents
    assert "--scope install-surface" in contents
    assert "SBOM written to" in contents


def test_generate_sbom_rejects_unknown_scope_before_tool_lookup() -> None:
    script_path = Path("scripts/security/generate_sbom.sh")

    result = subprocess.run(
        ["bash", str(script_path), "--scope", "unknown"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--scope must be environment" in result.stderr


def test_pip_audit_allowlist_is_owned_and_time_boxed() -> None:
    allowlist_path = Path("scripts/security/pip_audit_allowlist.json")
    payload = json.loads(allowlist_path.read_text(encoding="utf-8"))
    allowlist_doc = Path("docs/security/pip-audit-allowlist.md").read_text(
        encoding="utf-8"
    )

    assert payload["owner"] == "security-maintainers"
    assert isinstance(payload["entries"], list)
    if not payload["entries"]:
        assert "There are no active exceptions." in allowlist_doc

    for entry in payload["entries"]:
        advisory = entry["advisory"]
        assert entry["owner"] == "security-maintainers"
        expires = date.fromisoformat(entry["expires"])
        assert 0 <= (expires - date.today()).days <= 30
        assert re.fullmatch(
            r"https://github\.com/[^/]+/[^/]+/issues/[1-9]\d*",
            entry["tracking_issue"],
        )
        assert "reason" in entry
        assert f"`{advisory}`" in allowlist_doc
        assert entry["expires"] in allowlist_doc
        assert entry["tracking_issue"] in allowlist_doc


def test_ruff_toolchain_pins_are_aligned() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    ci_deps = pyproject["project"]["optional-dependencies"]["ci"]
    ruff_pin = next(dep for dep in ci_deps if dep.startswith("ruff=="))
    ruff_version = ruff_pin.removeprefix("ruff==")

    precommit_config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert f"rev: v{ruff_version}" in precommit_config

    for lockfile in (
        Path("requirements/workflows/ci-hf-py312.txt"),
        Path("requirements/workflows/ci-hf-py313.txt"),
        Path("requirements/workflows/docs-ci-py313.txt"),
        Path("requirements/workflows/assurance-ci-py313.txt"),
    ):
        text = lockfile.read_text(encoding="utf-8")
        assert f"ruff=={ruff_version} \\" in text


def test_security_workflow_lxml_pin_is_remediated() -> None:
    for lockfile in (
        Path("requirements/workflows/security-ci-py313.txt"),
        Path("requirements/workflows/release-security-py313.txt"),
    ):
        text = lockfile.read_text(encoding="utf-8")
        match = re.search(r"^lxml==(?P<version>\d+\.\d+\.\d+) \\", text, re.MULTILINE)
        assert match is not None
        assert match.group("version") == "6.1.0"
        assert "lxml==6.0.2" not in text

    uv_lock = Path("uv.lock").read_text(encoding="utf-8")
    assert 'name = "lxml"\nversion = "6.1.0"' in uv_lock
    assert 'name = "lxml"\nversion = "6.0.2"' not in uv_lock


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
        "requirements/workflows/pip-bootstrap-py313.txt",
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


def test_transformers_512_hashes_match_pypi_across_requirement_locks() -> None:
    mismatches = {
        str(path): sorted(_extract_transformers_512_hashes(path))
        for path in TRANSFORMERS_LOCKFILES
        if _extract_transformers_512_hashes(path) != TRANSFORMERS_512_HASHES
    }

    assert not mismatches, (
        "transformers==5.12.0 hashes drifted from the current PyPI wheel/sdist:\n"
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


def test_release_workflow_builds_and_publishes_tag_only_artifacts():
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
    assert (
        "python -m pip install --upgrade --require-hashes -r requirements/workflows/pip-bootstrap-py313.txt"
        in smoke_step["run"]
    )
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
    assert (
        "python -m pip install --upgrade --require-hashes -r requirements/workflows/pip-bootstrap-py313.txt"
        in install_surface_step["run"]
    )
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
    retired_bundle_job = "bundle" + "_release"
    release_cli_prefix = "gh" + " release"
    assert retired_bundle_job not in workflow["jobs"]
    assert not any(
        release_cli_prefix in str(step.get("run", ""))
        for step in _iter_job_steps(workflow)
    )

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
    assert "PyPI" in release_doc
    assert "gitleaks" in architecture_doc
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
