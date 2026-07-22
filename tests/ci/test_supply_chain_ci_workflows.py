from __future__ import annotations

import inspect
import json
import math
import os
import re
import subprocess
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
    assert inputs["candidate_run_id"]["default"] == ""
    assert inputs["publication_phase"]["default"] == "complete"
    assert inputs["publication_phase"]["options"] == [
        "complete",
        "bootstrap",
        "finish",
    ]

    build = jobs["build_check"]
    assert "github.event_name == 'push'" in build["if"]
    assert "inputs.publish != true" in build["if"]
    assert build["timeout-minutes"] >= 120
    assert build["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
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
    candidate_attestation = _step(build["steps"], "Attest tagged release candidate")
    assert candidate_attestation["if"] == "${{ github.event_name == 'push' }}"
    assert candidate_attestation["with"]["subject-checksums"] == ("dist/SHA256SUMS")
    assert upload_verify_index + 1 == build["steps"].index(candidate_attestation)
    assert build["steps"].index(candidate_attestation) + 1 == upload_index

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
    assert authorization["needs"] == "resolve_release_ref"
    assert authorization["permissions"] == {"actions": "read", "contents": "read"}
    authorization_checkout = _step(authorization["steps"], "Checkout release source")
    assert authorization_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    validate_publication = _step(authorization["steps"], "Validate publication inputs")
    assert validate_publication["env"]["INVARLOCK_CANDIDATE_RUN_ID"] == (
        "${{ inputs.candidate_run_id }}"
    )
    assert (
        "candidate_run_id must identify a successful tagged release run"
        in (validate_publication["run"])
    )
    authenticate = _step(authorization["steps"], "Authenticate tagged candidate run")
    authenticate_run = authenticate["run"]
    for required in (
        "scripts/release/tagged_release_candidate.py authenticate",
        '--release-sha "${INVARLOCK_RELEASE_SHA}"',
        '--release-tag "${INVARLOCK_RELEASE_TAG}"',
        '--candidate-run-id "${INVARLOCK_CANDIDATE_RUN_ID}"',
        '--repository "${GITHUB_REPOSITORY}"',
        '--api-url "${GITHUB_API_URL}"',
        '--github-output "${GITHUB_OUTPUT}"',
    ):
        assert required in authenticate_run
    candidate_download = _step(
        authorization["steps"], "Download tagged candidate distributions"
    )
    assert candidate_download["with"]["name"] == "dist"
    assert candidate_download["with"]["run-id"] == (
        "${{ steps.authenticate.outputs.artifact_run_id }}"
    )
    assert candidate_download["with"]["github-token"] == ("${{ secrets.GITHUB_TOKEN }}")
    verify_candidate = _step(authorization["steps"], "Verify tagged candidate ledger")
    assert (
        "scripts/release/tagged_release_candidate.py verify-ledger"
        in (verify_candidate["run"])
    )
    assert "--dist-dir _release_dist" in verify_candidate["run"]
    assert authorization["outputs"] == {
        "artifact_run_id": "${{ steps.authenticate.outputs.artifact_run_id }}",
        "dist_ledger_sha256": (
            "${{ steps.verify_candidate.outputs.dist_ledger_sha256 }}"
        ),
    }
    assert (
        authorization["steps"].index(authenticate)
        < authorization["steps"].index(candidate_download)
        < authorization["steps"].index(verify_candidate)
    )
    verify_bootstrap = _step(
        authorization["steps"], "Verify completed bootstrap publication"
    )
    assert "inputs.target == 'pypi'" in verify_bootstrap["if"]
    assert "inputs.publication_phase == 'finish'" in verify_bootstrap["if"]
    assert verify_bootstrap["env"] == {
        "EXPECTED_DIST_LEDGER_SHA256": (
            "${{ steps.verify_candidate.outputs.dist_ledger_sha256 }}"
        ),
        "INVARLOCK_RELEASE_VERSION": (
            "${{ needs.resolve_release_ref.outputs.release_tag }}"
        ),
    }
    for project in (
        "invarlock",
        "invarlock-diagnostics",
        "invarlock-runtime-gguf",
        "invarlock-runtime-hf-vision-text",
    ):
        assert f"--project {project}" in verify_bootstrap["run"]
    assert "invarlock-runtime-tensorrt-llm" not in verify_bootstrap["run"]

    publication_plan = jobs["publication_plan"]
    assert set(publication_plan["needs"]) == {
        "authorize_candidate",
        "resolve_release_ref",
    }
    plan_step = _step(publication_plan["steps"], "Resolve publication phase")
    assert plan_step["env"] == {
        "INVARLOCK_PUBLICATION_PHASE": "${{ inputs.publication_phase }}",
        "INVARLOCK_PUBLISH_TARGET": "${{ inputs.target }}",
    }
    for phase in ("complete", "bootstrap", "finish"):
        assert f"{phase})" in plan_step["run"]
    assert "bootstrap publication is only valid for production PyPI" in plan_step["run"]
    assert "finish publication is only valid for production PyPI" in plan_step["run"]

    publish = jobs["publish"]
    assert set(publish["needs"]) == {
        "authorize_candidate",
        "publication_plan",
        "resolve_release_ref",
    }
    assert publish["permissions"] == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
    }
    assert "github.event_name == 'push'" not in publish["if"]
    assert "github.event_name == 'workflow_dispatch'" in publish["if"]
    assert publish["strategy"]["fail-fast"] is False
    assert publish["strategy"]["matrix"] == (
        "${{ fromJSON(needs.publication_plan.outputs.matrix) }}"
    )
    assert publish["environment"] == (
        "${{ format('{0}{1}', inputs.target, matrix.environment_suffix) }}"
    )
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
    assert publish_step["with"]["skip-existing"] == (
        "${{ inputs.target == 'testpypi' }}"
    )
    stage_publish = _step(publish["steps"], "Stage publish distributions")
    assert stage_publish["env"]["INVARLOCK_PACKAGE"] == "${{ matrix.package }}"
    for package in (
        "core",
        "diagnostics",
        "runtime-gguf",
        "runtime-hf-vision-text",
        "runtime-tensorrt-llm",
    ):
        assert f"{package})" in stage_publish["run"]
    assert 'if [ "${#distributions[@]}" -ne 2 ]' in stage_publish["run"]
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
    assert "len(staged) != 2" in publish_verify
    assert "unknown = set(staged) - set(expected)" in publish_verify
    assert "distribution digest mismatch" in publish_verify
    verify_index = next(
        index
        for index, step in enumerate(publish["steps"])
        if step.get("name") == "Verify distribution digests before publish"
    )
    tag_index = publish["steps"].index(tag_recheck)
    assert verify_index < tag_index

    hosted_job = jobs["verify_hosted_release"]
    assert set(hosted_job["needs"]) == {
        "authorize_candidate",
        "publish",
        "resolve_release_ref",
    }
    assert hosted_job["permissions"] == {"actions": "read", "contents": "read"}
    assert "inputs.publication_phase != 'bootstrap'" in hosted_job["if"]
    hosted_checkout = hosted_job["steps"][0]
    assert hosted_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    hosted_ledger = _step(hosted_job["steps"], "Download candidate distribution ledger")
    assert hosted_ledger["with"]["run-id"] == (
        "${{ needs.authorize_candidate.outputs.artifact_run_id }}"
    )
    hosted_verify = _step(
        hosted_job["steps"], "Verify hosted release artifacts match build"
    )
    assert hosted_verify["env"] == {
        "EXPECTED_DIST_LEDGER_SHA256": (
            "${{ needs.authorize_candidate.outputs.dist_ledger_sha256 }}"
        ),
        "INVARLOCK_PUBLISH_TARGET": "${{ inputs.target }}",
        "INVARLOCK_RELEASE_VERSION": (
            "${{ needs.resolve_release_ref.outputs.release_tag }}"
        ),
    }
    assert "scripts/release/verify_hosted_distributions.py" in hosted_verify["run"]
    assert "--ledger _release_dist/SHA256SUMS" in hosted_verify["run"]
    assert "if" not in hosted_verify

    published_smoke = jobs["published_install_smoke"]
    assert set(published_smoke["needs"]) == {
        "authorize_candidate",
        "resolve_release_ref",
        "verify_hosted_release",
    }
    assert published_smoke["permissions"] == {"actions": "read", "contents": "read"}
    assert "inputs.publication_phase != 'bootstrap'" in published_smoke["if"]
    published_checkout = published_smoke["steps"][0]
    assert published_checkout["with"]["ref"] == (
        "${{ needs.resolve_release_ref.outputs.release_sha }}"
    )
    hosted_download = _step(
        published_smoke["steps"], "Download ledger-selected hosted wheels"
    )
    assert hosted_download["env"]["EXPECTED_DIST_LEDGER_SHA256"] == (
        "${{ needs.authorize_candidate.outputs.dist_ledger_sha256 }}"
    )
    assert "scripts/release/verify_hosted_distributions.py" in hosted_download["run"]
    assert "--ledger _release_dist/SHA256SUMS" in hosted_download["run"]
    assert '--target "${INVARLOCK_PUBLISH_TARGET}"' in hosted_download["run"]
    assert "--wheelhouse wheelhouse" in hosted_download["run"]
    assert "urllib.request" not in hosted_download["run"]
    smoke = _step(published_smoke["steps"], "Install published wheels and smoke test")
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

    assert "record_testpypi_promotion" not in jobs


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

    assert jobs["verify_hosted_release"]["timeout-minutes"] >= required_job_minutes
    assert jobs["published_install_smoke"]["timeout-minutes"] >= required_job_minutes


def test_release_publication_plan_is_closed_and_phase_specific(tmp_path: Path) -> None:
    workflow = _load(WORKFLOWS / "release.yml")
    step = _step(
        workflow["jobs"]["publication_plan"]["steps"],
        "Resolve publication phase",
    )
    expected = {
        "complete": [
            "core",
            "diagnostics",
            "runtime-gguf",
            "runtime-hf-vision-text",
            "runtime-tensorrt-llm",
        ],
        "bootstrap": [
            "core",
            "diagnostics",
            "runtime-gguf",
            "runtime-hf-vision-text",
        ],
        "finish": ["runtime-tensorrt-llm"],
    }
    output = tmp_path / "github-output"
    for phase, expected_packages in expected.items():
        output.write_text("", encoding="utf-8")
        subprocess.run(
            ["bash", "-c", step["run"]],
            env={
                **os.environ,
                "GITHUB_OUTPUT": str(output),
                "INVARLOCK_PUBLICATION_PHASE": phase,
                "INVARLOCK_PUBLISH_TARGET": "pypi",
            },
            check=True,
        )
        line = output.read_text(encoding="utf-8").strip()
        assert line.startswith("matrix=")
        matrix = json.loads(line.removeprefix("matrix="))
        assert [entry["package"] for entry in matrix["include"]] == expected_packages

    rejected = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "INVARLOCK_PUBLICATION_PHASE": "bootstrap",
            "INVARLOCK_PUBLISH_TARGET": "testpypi",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "only valid for production PyPI" in rejected.stderr


def test_release_stages_each_distribution_in_its_own_publish_job(
    tmp_path: Path,
) -> None:
    workflow = _load(WORKFLOWS / "release.yml")
    publish = workflow["jobs"]["publish"]
    stage = _step(publish["steps"], "Stage publish distributions")["run"]
    release_dist = tmp_path / "_release_dist"
    addin_dist = release_dist / "addins"
    addin_dist.mkdir(parents=True)

    expected_prefixes = {
        "core": (release_dist, "invarlock"),
        "diagnostics": (addin_dist, "invarlock_diagnostics"),
        "runtime-gguf": (addin_dist, "invarlock_runtime_gguf"),
        "runtime-hf-vision-text": (
            addin_dist,
            "invarlock_runtime_hf_vision_text",
        ),
        "runtime-tensorrt-llm": (
            addin_dist,
            "invarlock_runtime_tensorrt_llm",
        ),
    }
    for root, prefix in expected_prefixes.values():
        (root / f"{prefix}-0.13.0-py3-none-any.whl").write_bytes(b"wheel")
        (root / f"{prefix}-0.13.0.tar.gz").write_bytes(b"source")

    for package, (_, prefix) in expected_prefixes.items():
        subprocess.run(
            ["bash", "-c", stage],
            cwd=tmp_path,
            env={**os.environ, "INVARLOCK_PACKAGE": package},
            check=True,
        )
        staged = sorted(path.name for path in (tmp_path / "publish-dist").iterdir())
        assert staged == [
            f"{prefix}-0.13.0-py3-none-any.whl",
            f"{prefix}-0.13.0.tar.gz",
        ]


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
