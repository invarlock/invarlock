from __future__ import annotations

import inspect
import math
import re
from pathlib import Path
from typing import Any

import yaml

from scripts.release import verify_hosted_distributions as hosted_verifier

WORKFLOWS = Path(".github/workflows")
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
RELEASE_INSTALL_LOCKS = (
    Path("requirements/workflows/release-install-py312.txt"),
    Path("requirements/workflows/release-install-py313.txt"),
)


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
    malformed_steps: list[str] = []

    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = _load(path)
        if "permissions" not in workflow:
            missing_permissions.append(path.name)
        for index, step in enumerate(_steps(workflow)):
            execution_keys = [key for key in ("run", "uses") if key in step]
            if len(execution_keys) != 1:
                malformed_steps.append(
                    f"{path.name}: step {index} {step.get('name', '<unnamed>')}"
                )
            uses = str(step.get("uses", ""))
            if uses and not uses.startswith("./") and not PINNED_ACTION.fullmatch(uses):
                unpinned.append(f"{path.name}: {uses}")

    assert unpinned == []
    assert missing_permissions == []
    assert malformed_steps == []


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


def test_container_front_door_authenticates_its_runtime_source_bundle() -> None:
    workflow = _load(WORKFLOWS / "container-front-door-smoke.yml")
    steps = workflow["jobs"]["smoke"]["steps"]
    authentication = _step(steps, "Authenticate runtime source")["run"]

    assert "scripts/qualification_source.py create" in authentication
    assert "scripts/qualification_source.py verify" in authentication
    assert "RUNTIME_SOURCE_COMMIT=" in authentication
    assert "RUNTIME_SOURCE_BUNDLE=" in authentication
    assert "RUNTIME_SOURCE_BUNDLE_SHA256=" in authentication
    candidate = _step(steps, "Install candidate wheel")["run"]
    assert "python -m build --no-isolation --wheel" in candidate
    assert "pip install --no-deps --force-reinstall" in candidate
    journey = _step(steps, "Exercise installed-wheel OCI and signed canary journey")[
        "run"
    ]
    assert "make runtime-image" in journey
    assert "INVARLOCK_CONTAINER_SMOKE_INSTALLED_WHEEL=1" in journey
    assert "test_container_front_door_journey.py" in journey


def test_release_install_dependency_closure_is_hash_pinned_and_refreshable() -> None:
    for path in RELEASE_INSTALL_LOCKS:
        lock = path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert "numpy==" in lock
        assert "pillow==" in lock
        assert "typer==" in lock

    refresh = Path("scripts/security/refresh_pinned_requirements.sh").read_text(
        encoding="utf-8"
    )
    assert "compile_release_install" in refresh
    assert '"${WORKFLOW_DIR}/release-install-py312.txt"' in refresh
    assert '"${WORKFLOW_DIR}/release-install-py313.txt"' in refresh


def test_pr_supply_chain_scans_only_shipped_dependency_surfaces() -> None:
    workflow = _load(WORKFLOWS / "supply-chain-pr.yml")
    scan = workflow["jobs"]["scan"]
    steps = scan["steps"]
    names = [step.get("name") for step in steps]

    for required in (
        "Audit maintained dependency locks",
        "Test gitleaks allowlist boundary",
        "Run gitleaks PR git delta scan",
        "Build release wheel",
        "Run pip-audit",
        "Generate install-surface SBOM",
        "Run HF surface pip-audit",
    ):
        assert required in names
    assert not any("advanced" in str(name).lower() for name in names)

    allowlist_probe = _step(steps, "Test gitleaks allowlist boundary")["run"]
    assert '"api_key":"%s"' in allowlist_probe
    assert (
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        in allowlist_probe
    )
    assert "gitleaks git ." in allowlist_probe
    assert "exit_code" in allowlist_probe

    secret_scan = _step(steps, "Run gitleaks PR git delta scan")
    assert "--config .gitleaks.toml" in secret_scan["run"]
    assert "--log-opts" in secret_scan["run"]
    assert "--redact" in secret_scan["run"]
    lock_audit = _step(steps, "Audit maintained dependency locks")["run"]
    assert "scripts/security/cve_audit.py" in lock_audit
    assert "artifacts/supply-chain/cve-audit.json" in lock_audit
    lock_upload = _step(steps, "Upload dependency audit report")
    assert lock_upload["if"] == "${{ always() }}"
    assert "cve-audit.json" in lock_upload["with"]["path"]
    assert (
        steps.index(lock_upload)
        == steps.index(_step(steps, "Audit maintained dependency locks")) + 1
    )


def test_release_builds_from_the_resolved_tag_and_uses_trusted_publishing() -> None:
    workflow = _load(WORKFLOWS / "release.yml")
    jobs = workflow["jobs"]

    resolve = jobs["resolve_release_ref"]
    resolve_step = _step(resolve["steps"], "Resolve release ref")
    assert "git ls-remote --tags" in resolve_step["run"]
    assert "release_tag must be a version tag" in resolve_step["run"]
    assert (
        "release tag no longer identifies the workflow event commit"
        in resolve_step["run"]
    )
    assert '"${sha}" != "${INVARLOCK_EVENT_SHA}"' in resolve_step["run"]
    assert "${{ inputs.release_tag }}" not in resolve_step["run"]
    assert "${{ github.sha }}" not in resolve_step["run"]
    assert resolve_step["env"]["INVARLOCK_EVENT_SHA"] == "${{ github.sha }}"
    assert resolve_step["env"]["INVARLOCK_MANUAL_RELEASE_TAG"] == (
        "${{ inputs.release_tag }}"
    )
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["promotion_run_id"]["default"] == ""

    build = jobs["build_check"]
    assert build["timeout-minutes"] >= 120
    checkout = build["steps"][0]
    assert checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    distribution_build = _step(build["steps"], "Build first-party distributions")["run"]
    assert "python -m build --no-isolation" in distribution_build
    for addin in ("diagnostics", "gguf", "multimodal", "tensorrt_llm"):
        assert f"addins/{addin}" in distribution_build

    assert _step(build["steps"], "Run complete repository gates")["run"] == (
        "make verify"
    )
    build_tooling = _step(build["steps"], "Install build tooling")["run"]
    assert "requirements/workflows/ci-hf-py313.txt" in build_tooling
    assert "requirements/workflows/docs-ci-py313.txt" in build_tooling
    assert "requirements/workflows/release-security-py313.txt" in build_tooling
    release_lock_audit = _step(build["steps"], "Audit maintained dependency locks")[
        "run"
    ]
    assert "scripts/security/cve_audit.py" in release_lock_audit
    release_lock_upload = _step(build["steps"], "Upload dependency audit report")
    assert release_lock_upload["if"] == "${{ always() }}"
    assert "cve-audit.json" in release_lock_upload["with"]["path"]
    assert (
        build["steps"].index(release_lock_upload)
        == build["steps"].index(
            _step(build["steps"], "Audit maintained dependency locks")
        )
        + 1
    )
    assert _step(build["steps"], "Enforce release coverage")["run"] == (
        "make coverage-enforce"
    )
    assert (
        "make workflow-lint" in _step(build["steps"], "Lint release workflows")["run"]
    )

    digest_record = _step(build["steps"], "Record distribution digests")["run"]
    assert 'dist / "SHA256SUMS"' in digest_record
    assert "len(wheels) != 5" in digest_record
    assert "len(source_archives) != 5" in digest_record
    assert "ledger_sha256=" in digest_record
    assert build["outputs"]["dist_ledger_sha256"] == (
        "${{ steps.dist_digests.outputs.ledger_sha256 }}"
    )

    twine = _step(build["steps"], "Twine check")["run"]
    assert "dist/*.whl dist/*.tar.gz" in twine
    assert "dist/addins/*" in twine

    parity = _step(build["steps"], "Validate first-party distribution source parity")[
        "run"
    ]
    assert "first_party_distribution_validation.py" in parity
    assert "--core-dist-dir dist" in parity
    assert "--addin-dist-dir dist/addins" in parity

    preflight = _step(build["steps"], "Run clean-checkout release preflight")["run"]
    assert "git worktree add --detach" in preflight
    assert '"${release_checkout}/scripts/release/release_preflight.py"' in preflight
    assert '--repo-root "${release_checkout}"' in preflight
    assert '--release-sha "${INVARLOCK_RELEASE_SHA}"' in preflight
    assert '--hash-manifest "${hash_manifest}"' in preflight

    install_smoke = _step(build["steps"], "Install smoke from wheel")["run"]
    assert install_smoke.index("release-install-py313.txt") < install_smoke.index(
        "--no-deps --force-reinstall dist/*.whl dist/addins/*.whl"
    )
    for isolation_command in (
        "export PYTHONNOUSERSITE=1",
        "export PYTHONSAFEPATH=1",
        "unset PYTHONPATH",
    ):
        assert install_smoke.index(isolation_command) < install_smoke.index(
            "--no-deps --force-reinstall dist/*.whl dist/addins/*.whl"
        )
    assert "python -m pip check" in install_smoke
    assert "CoreRegistry" in install_smoke
    assert "get_runtime_provider" in install_smoke
    assert "get_plugin_info" in install_smoke
    assert "is_relative_to(site)" in install_smoke
    assert "first-party version mismatch" in install_smoke
    assert "release tag/version mismatch" in install_smoke
    assert (
        _step(build["steps"], "Install smoke from wheel")["env"][
            "INVARLOCK_RELEASE_TAG"
        ]
        == "${{ needs.resolve_release_ref.outputs.release_tag }}"
    )
    for project in (
        "invarlock-diagnostics",
        "invarlock-runtime-gguf",
        "invarlock-runtime-hf-vision-text",
        "invarlock-runtime-tensorrt-llm",
    ):
        assert project in install_smoke

    install_surface = _step(build["steps"], "Create release install-surface venv")[
        "run"
    ]
    assert install_surface.index("release-install-py313.txt") < install_surface.index(
        "--no-deps --force-reinstall dist/*.whl dist/addins/*.whl"
    )
    for isolation_command in (
        "export PYTHONNOUSERSITE=1",
        "export PYTHONSAFEPATH=1",
        "unset PYTHONPATH",
    ):
        assert install_surface.index(isolation_command) < install_surface.index(
            "--no-deps --force-reinstall dist/*.whl dist/addins/*.whl"
        )
    assert "python -m pip check" in install_surface

    upload_verify = _step(
        build["steps"], "Verify distribution digests before artifact upload"
    )["run"]
    assert "EXPECTED_DIST_LEDGER_SHA256" in upload_verify
    assert "SHA256SUMS | sha256sum --check -" in upload_verify
    assert "sha256sum --check SHA256SUMS" in upload_verify
    upload_step = _step(build["steps"], "Upload dist artifacts")
    assert upload_step.get("if", "${{ success() }}") == "${{ success() }}"
    assert "dist/SHA256SUMS" in upload_step["with"]["path"]
    upload_verify_index = next(
        index
        for index, step in enumerate(build["steps"])
        if step.get("name") == "Verify distribution digests before artifact upload"
    )
    upload_index = next(
        index
        for index, step in enumerate(build["steps"])
        if step.get("name") == "Upload dist artifacts"
    )
    assert upload_verify_index + 1 == upload_index

    gitleaks_range = _step(build["steps"], "Resolve gitleaks release range")
    assert "previous release tag is not a valid version tag" in gitleaks_range["run"]
    assert "previous_sha" in gitleaks_range["run"]
    gitleaks_scan = _step(build["steps"], "Run gitleaks release delta scan")
    assert "${{ steps.gitleaks_range.outputs.log_opts }}" not in gitleaks_scan["run"]
    assert gitleaks_scan["env"]["INVARLOCK_GITLEAKS_LOG_OPTS"] == (
        "${{ steps.gitleaks_range.outputs.log_opts }}"
    )
    secret_upload_index = next(
        index
        for index, step in enumerate(build["steps"])
        if step.get("name") == "Upload gitleaks artifacts"
    )
    secret_failure_index = next(
        index
        for index, step in enumerate(build["steps"])
        if step.get("name") == "Fail on secret findings"
    )
    distribution_build_index = next(
        index
        for index, step in enumerate(build["steps"])
        if step.get("name") == "Build first-party distributions"
    )
    assert build["steps"].index(gitleaks_scan) < secret_upload_index
    assert secret_upload_index + 1 == secret_failure_index
    assert secret_failure_index < distribution_build_index

    authorization = jobs["authorize_candidate"]
    assert authorization["permissions"] == {"actions": "read", "contents": "read"}
    authorization_checkout = _step(authorization["steps"], "Checkout release source")
    assert authorization_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    validate_publication = _step(authorization["steps"], "Validate publication inputs")
    assert validate_publication["env"]["INVARLOCK_PROMOTION_RUN_ID"] == (
        "${{ inputs.promotion_run_id }}"
    )
    assert (
        "promotion_run_id must identify a successful TestPyPI run"
        in (validate_publication["run"])
    )
    promotion_download = _step(
        authorization["steps"], "Download promoted candidate authorization"
    )
    assert promotion_download["if"] == "${{ inputs.target == 'pypi' }}"
    assert promotion_download["with"]["name"] == "testpypi-promotion"
    assert promotion_download["with"]["run-id"] == (
        "${{ steps.validate_publication.outputs.candidate_run_id }}"
    )
    assert promotion_download["with"]["github-token"] == ("${{ secrets.GITHUB_TOKEN }}")
    authorize = _step(authorization["steps"], "Authorize exact publication candidate")[
        "run"
    ]
    assert authorization["steps"].index(authorization_checkout) < next(
        index
        for index, step in enumerate(authorization["steps"])
        if step.get("name") == "Authorize exact publication candidate"
    )
    for required in (
        "scripts/release/testpypi_promotion.py current",
        "scripts/release/testpypi_promotion.py authorize",
        "--manifest _promotion/promotion.json",
        '--release-sha "${INVARLOCK_RELEASE_SHA}"',
        '--release-tag "${INVARLOCK_RELEASE_TAG}"',
        '--candidate-run-id "${INVARLOCK_CANDIDATE_RUN_ID}"',
        '--repository "${GITHUB_REPOSITORY}"',
        '--api-url "${GITHUB_API_URL}"',
        '--github-output "${GITHUB_OUTPUT}"',
    ):
        assert required in authorize

    publish = jobs["publish"]
    assert publish["permissions"] == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert "github.event_name == 'push'" not in publish["if"]
    assert "github.event_name == 'workflow_dispatch'" in publish["if"]
    publish_download = _step(publish["steps"], "Download dist artifacts")
    assert publish_download["with"]["run-id"] == (
        "${{ needs.authorize_candidate.outputs.artifact_run_id }}"
    )
    assert publish_download["with"]["github-token"] == ("${{ secrets.GITHUB_TOKEN }}")
    publish_step = next(
        step
        for step in publish["steps"]
        if str(step.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )
    assert "user" not in publish_step.get("with", {})
    assert "password" not in publish_step.get("with", {})
    assert publish_step["with"]["skip-existing"] is True
    tag_recheck = _step(publish["steps"], "Reconfirm immutable release tag")
    assert tag_recheck["env"] == {
        "INVARLOCK_RELEASE_SHA": (
            "${{ needs.resolve_release_ref.outputs.release_sha }}"
        ),
        "INVARLOCK_RELEASE_TAG": (
            "${{ needs.resolve_release_ref.outputs.release_tag }}"
        ),
    }
    assert "git ls-remote --tags" in tag_recheck["run"]
    assert "release tag changed after candidate authorization" in tag_recheck["run"]
    assert publish["steps"].index(tag_recheck) + 1 == publish["steps"].index(
        publish_step
    )
    publish_verify = _step(
        publish["steps"], "Verify distribution digests before publish"
    )
    assert publish_verify["env"]["EXPECTED_DIST_LEDGER_SHA256"] == (
        "${{ needs.authorize_candidate.outputs.dist_ledger_sha256 }}"
    )
    publish_verify = publish_verify["run"]
    assert 'Path("_release_dist/SHA256SUMS")' in publish_verify
    assert "distribution digest ledger changed after build" in publish_verify
    assert 'Path("publish-dist").iterdir()' in publish_verify
    assert "set(staged) != set(expected)" in publish_verify
    assert "distribution digest mismatch" in publish_verify
    verify_index = next(
        index
        for index, step in enumerate(publish["steps"])
        if step.get("name") == "Verify distribution digests before publish"
    )
    attest_index = next(
        index
        for index, step in enumerate(publish["steps"])
        if step.get("name") == "Attest release artifacts"
    )
    assert verify_index + 1 == attest_index

    hosted_verify = _step(
        publish["steps"], "Verify hosted release artifacts match build"
    )
    assert hosted_verify["env"] == {
        "EXPECTED_DIST_LEDGER_SHA256": (
            "${{ needs.authorize_candidate.outputs.dist_ledger_sha256 }}"
        ),
        "INVARLOCK_PUBLISH_TARGET": "${{ steps.vars.outputs.publish_target }}",
        "INVARLOCK_RELEASE_VERSION": (
            "${{ needs.resolve_release_ref.outputs.release_tag }}"
        ),
    }
    assert "scripts/release/verify_hosted_distributions.py" in hosted_verify["run"]
    assert "--ledger _release_dist/SHA256SUMS" in hosted_verify["run"]
    publish_index = publish["steps"].index(publish_step)
    assert publish_index + 1 == publish["steps"].index(hosted_verify)
    assert "if" not in hosted_verify

    testpypi = jobs["testpypi_smoke"]
    assert set(testpypi["needs"]) == {
        "authorize_candidate",
        "publish",
        "resolve_release_ref",
    }
    assert testpypi["permissions"] == {"actions": "read", "contents": "read"}
    testpypi_checkout = testpypi["steps"][0]
    assert testpypi_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    hosted_download = _step(
        testpypi["steps"], "Download ledger-selected TestPyPI wheels"
    )
    assert hosted_download["env"]["EXPECTED_DIST_LEDGER_SHA256"] == (
        "${{ needs.authorize_candidate.outputs.dist_ledger_sha256 }}"
    )
    assert "scripts/release/verify_hosted_distributions.py" in hosted_download["run"]
    assert "--ledger _release_dist/SHA256SUMS" in hosted_download["run"]
    assert "--wheelhouse wheelhouse" in hosted_download["run"]
    assert "urllib.request" not in hosted_download["run"]
    smoke = _step(testpypi["steps"], "Install published wheel and smoke test")
    assert "requirements/workflows/pip-bootstrap.txt" in smoke["run"]
    assert smoke["run"].index("release-install-py313.txt") < smoke["run"].index(
        "--no-deps --force-reinstall wheelhouse/*.whl"
    )
    for isolation_command in (
        "export PYTHONNOUSERSITE=1",
        "export PYTHONSAFEPATH=1",
        "unset PYTHONPATH",
    ):
        assert smoke["run"].index(isolation_command) < smoke["run"].index(
            "--no-deps --force-reinstall wheelhouse/*.whl"
        )
    assert "python -m pip check" in smoke["run"]
    assert "CoreRegistry" in smoke["run"]
    assert "get_runtime_provider" in smoke["run"]
    assert "get_plugin_info" in smoke["run"]
    assert "is_relative_to(site)" in smoke["run"]
    assert "first-party version mismatch" in smoke["run"]

    promotion = jobs["record_testpypi_promotion"]
    assert set(promotion["needs"]) == {
        "authorize_candidate",
        "resolve_release_ref",
        "testpypi_smoke",
    }
    assert "needs.testpypi_smoke.result == 'success'" in promotion["if"]
    promotion_checkout = _step(promotion["steps"], "Checkout release source")
    assert promotion_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    promotion_record = _step(
        promotion["steps"], "Record exact TestPyPI promotion candidate"
    )["run"]
    assert promotion["steps"].index(promotion_checkout) < next(
        index
        for index, step in enumerate(promotion["steps"])
        if step.get("name") == "Record exact TestPyPI promotion candidate"
    )
    assert "scripts/release/testpypi_promotion.py record" in promotion_record
    assert '--release-sha "${INVARLOCK_RELEASE_SHA}"' in promotion_record
    assert '--release-tag "${INVARLOCK_RELEASE_TAG}"' in promotion_record
    assert '--source-run-id "${GITHUB_RUN_ID}"' in promotion_record
    promotion_upload = _step(
        promotion["steps"], "Upload TestPyPI promotion authorization"
    )
    assert promotion_upload["with"]["name"] == "testpypi-promotion"
    assert promotion_upload["with"]["if-no-files-found"] == "error"


def test_release_jobs_outlive_hosted_verification_worst_case() -> None:
    workflow = _load(WORKFLOWS / "release.yml")
    jobs = workflow["jobs"]
    signature = inspect.signature(hosted_verifier.verify_hosted_distributions)
    attempts = signature.parameters["attempts"].default
    retry_delay = signature.parameters["retry_delay"].default
    request_timeout = signature.parameters["timeout"].default

    assert isinstance(attempts, int)
    assert isinstance(retry_delay, float)
    assert isinstance(request_timeout, float)
    requests_per_attempt = len(hosted_verifier.PROJECTS) * 3
    verifier_worst_case_seconds = (
        attempts * requests_per_attempt * request_timeout + (attempts - 1) * retry_delay
    )
    required_job_minutes = math.ceil(verifier_worst_case_seconds / 60) + 15

    assert jobs["publish"]["timeout-minutes"] >= required_job_minutes
    assert jobs["testpypi_smoke"]["timeout-minutes"] >= required_job_minutes


def test_docs_publish_validates_dispatch_input_before_using_it_as_a_path() -> None:
    workflow = _load(WORKFLOWS / "docs-publish.yml")
    steps = workflow["jobs"]["publish"]["steps"]
    resolve = _step(steps, "Resolve docs version")

    assert "${{ github.event.inputs.docs_version }}" not in resolve["run"]
    assert resolve["env"]["INVARLOCK_DOCS_VERSION_OVERRIDE"] == (
        "${{ github.event.inputs.docs_version }}"
    )
    assert "one safe path component" in resolve["run"]
    assert "[A-Za-z0-9._-]" in resolve["run"]


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
