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


def test_ci_pr_assurance_gates_are_required_jobs() -> None:
    workflow = _load_workflow(Path(".github/workflows/ci.yml"))
    triggers = workflow["on"]

    assert triggers["push"]["branches"] == ["main", "release/v*"]
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

    assert triggers["push"]["branches"] == ["main", "staging/next", "release/v*"]
    assert triggers["pull_request"]["branches"] == [
        "main",
        "develop",
        "staging/next",
        "release/v*",
    ]

    expected_paths = [
        "docs/**",
        "README.md",
        "CONTRIBUTING.md",
        "mkdocs.yml",
        "Makefile",
        "notebooks/**",
        "package.json",
        "package-lock.json",
        "requirements/workflows/docs-ci-py313.txt",
        "scripts/check_claim_surface_consistency.py",
        "scripts/check_cli_completeness.py",
        "scripts/check_config_schema_sync.py",
        "scripts/check_docs_links.py",
        "scripts/check_guard_completeness.py",
        "scripts/check_internal_links.py",
        "scripts/check_version_consistency.py",
        "scripts/docs_check.py",
        "scripts/docs_lint.py",
        "scripts/lint_assurance_xrefs.py",
        "scripts/test_cli_examples.py",
        "scripts/validate_doc_references.py",
        "scripts/validate_docs_api_refs.py",
        "scripts/validate_python_examples.py",
        "scripts/validate_yaml_examples.py",
        "scripts/verify_live_examples.py",
        "scripts/verify_markdown_bash_blocks.py",
        "scripts/verify_notebooks_smoke.py",
        ".github/workflows/docs-ci.yml",
    ]
    assert triggers["push"]["paths"] == expected_paths
    assert triggers["pull_request"]["paths"] == expected_paths

    steps = workflow["jobs"]["docs-validate"]["steps"]
    node_step = _find_step_by_name(steps, "Setup Node.js")
    install_node_step = _find_step_by_name(steps, "Install docs lint toolchain")
    markdown_step = _find_step_by_name(steps, "Lint markdown")
    spell_step = _find_step_by_name(steps, "Spell check")
    upload_step = _find_step_by_name(steps, "Upload build artifacts")
    step_names = [step.get("name") for step in steps]

    assert node_step["with"]["node-version"] == "22"
    assert install_node_step["run"] == "npm ci"
    assert markdown_step["run"] == "python scripts/docs_lint.py --markdown"
    assert "continue-on-error" not in markdown_step
    assert spell_step["run"] == "python scripts/docs_lint.py --spell"
    assert "continue-on-error" not in spell_step
    assert upload_step["with"]["path"] == "site/"
    assert step_names.index("Upload build artifacts") < step_names.index(
        "Ensure clean working tree"
    )

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


def test_readme_does_not_expose_public_scorecard_badge() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "OpenSSF Scorecard" not in readme
    assert (
        "https://api.scorecard.dev/projects/github.com/invarlock/invarlock/badge"
        not in readme
    )
    assert (
        "https://scorecard.dev/viewer/?uri=github.com/invarlock/invarlock" not in readme
    )


def test_readme_mentions_probes_extra() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "invarlock[probes]" in readme


def test_codeql_workflow_uses_repo_config():
    workflow = _load_workflow(Path(".github/workflows/codeql.yml"))
    triggers = workflow["on"]
    assert triggers["push"]["branches"] == ["main", "staging/next", "release/v*"]
    assert triggers["pull_request"]["branches"] == [
        "main",
        "staging/next",
        "release/v*",
    ]
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
    expected_pin = "7211b7c8077ea37d8641b6271f6a365a22a5fbfa"

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


def test_dependabot_does_not_enable_routine_uv_version_updates() -> None:
    config = yaml.safe_load(Path(".github/dependabot.yml").read_text(encoding="utf-8"))

    ecosystems = {update["package-ecosystem"] for update in config["updates"]}

    assert "uv" not in ecosystems


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
    assert env["INVARLOCK_SMOKE_MODE"] == "container"
    assert env["INVARLOCK_SMOKE_PROFILE"] == "dev"
    assert (
        env["INVARLOCK_SMOKE_JOURNEYS"]
        == "strict-bundle,noop,quantized,edited,negative"
    )
    assert env["INVARLOCK_RUNTIME_IMAGE"] == "invarlock-runtime:local"

    steps = job["steps"]
    install = _find_step_by_name(steps, "Install dependencies")
    assert "pip install --require-hashes" in install["run"]

    runtime_image = _find_step_by_name(steps, "Build runtime image")
    assert "make runtime-image" in runtime_image["run"]

    smoke = _find_step_by_name(steps, "Run GPT-2 user journey smoke")
    assert "scripts/run_gpt2_user_journey_smoke.sh" in smoke["run"]


def test_ci_hf_lockfiles_include_hypothesis_for_property_tests() -> None:
    for path in (
        Path("requirements/workflows/ci-hf-py312.txt"),
        Path("requirements/workflows/ci-hf-py313.txt"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "hypothesis==" in text, f"hypothesis missing from {path}"


def test_tiny_container_smoke_workflow_is_configured() -> None:
    workflow = _load_workflow(Path(".github/workflows/tiny-container-smoke.yml"))
    triggers = workflow["on"]

    assert triggers["push"]["branches"] == ["staging/next", "release/v*"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["smoke"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 45

    env = job["env"]
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert env["INVARLOCK_SMOKE_MODE"] == "container"
    assert env["INVARLOCK_SMOKE_PROFILE"] == "dev"
    assert env["INVARLOCK_RUNTIME_IMAGE"] == "invarlock-runtime:local"

    steps = job["steps"]
    install = _find_step_by_name(steps, "Install dependencies")
    assert "pip install --require-hashes" in install["run"]

    runtime_image = _find_step_by_name(steps, "Build runtime image")
    assert "make runtime-image" in runtime_image["run"]

    smoke = _find_step_by_name(steps, "Run tiny container smoke campaign")
    assert "scripts/run_tiny_container_smoke.sh" in smoke["run"]
