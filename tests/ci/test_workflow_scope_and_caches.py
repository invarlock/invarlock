from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def _load(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in document:
        document["on"] = document.pop(True)
    return document


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_pip_cache_tracks_each_jobs_installed_locks(path: Path) -> None:
    for name, job in _load(path)["jobs"].items():
        steps = job.get("steps", [])
        for setup in steps:
            if not setup.get("uses", "").startswith("actions/setup-python@"):
                continue
            options = setup.get("with", {})
            if options.get("cache") != "pip":
                continue
            commands = "\n".join(step.get("run", "") for step in steps)
            installed = set(
                re.findall(
                    r"pip install[^\n]*?-r (requirements/workflows/[\w.-]+\.txt)",
                    commands,
                )
            )
            tag = options["python-version"].replace(".", "")
            # These gates install additional locked environments through Make
            # and a shell helper; their downloads share the job's pip cache.
            if "install-smoke" in commands:
                installed.update(
                    (
                        f"requirements/workflows/release-install-py{tag}.txt",
                        f"requirements/workflows/release-options-py{tag}.txt",
                        "requirements/workflows/pip-bootstrap.txt",
                    )
                )
            if (
                "inspect-judge-sdk-test" in commands
                or "scripts/inspect_judge_sdk_gate.sh" in commands
            ):
                installed.update(
                    (
                        f"requirements/workflows/inspect-judge-tests-py{tag}.txt",
                        "requirements/workflows/pip-bootstrap.txt",
                    )
                )
            if "langfuse-sdk-test" in commands.split():
                installed.update(
                    (
                        f"requirements/workflows/langfuse-sdk-tests-py{tag}.txt",
                        f"requirements/workflows/release-install-py{tag}.txt",
                        "requirements/workflows/pip-bootstrap.txt",
                    )
                )
            cached = options.get("cache-dependency-path", "").splitlines()
            assert installed, (path.name, name)
            assert set(cached) == installed, (path.name, name)
            assert len(cached) == len(set(cached))
            assert all((ROOT / lock).is_file() for lock in cached)


def test_docs_workflow_builds_and_lints_once_and_checks_commands() -> None:
    job = _load(WORKFLOWS / "docs-ci.yml")["jobs"]["docs"]
    commands = [
        step["run"] for step in job["steps"] if step.get("run", "").startswith("make ")
    ]
    assert commands
    planned = "\n".join(
        subprocess.run(
            ["make", "--dry-run", *shlex.split(command)[1:]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for command in commands
    )
    for gate in (
        "-m mkdocs build --strict",
        "markdownlint-cli2 --",
        "cspell --no-progress",
        "scripts/checks/check_public_text.py",
        "render_docs_matrix.py --check",
        "-m invarlock evaluate --help",
        "-m invarlock verify --help",
        "-m invarlock report --help",
    ):
        assert planned.count(gate) == 1, (gate, planned)


def test_codeql_includes_complete_distribution_sources_without_tests() -> None:
    config = _load(ROOT / ".github/codeql/codeql-config.yml")
    paths = [Path(path) for path in config["paths"]]
    assert {Path("src/invarlock"), Path("scripts")} <= set(paths)
    assert all((ROOT / path).is_dir() for path in paths)
    shipped = (
        ROOT / "src/invarlock/diagnostics",
        ROOT / "src/invarlock/judge_measurements",
        ROOT / "src/invarlock/runtime_providers",
    )
    assert all(path.is_dir() for path in shipped)
    assert all(
        path.relative_to(ROOT).is_relative_to(source_root)
        for path in shipped
        for source_root in (Path("src/invarlock"),)
    )
    assert not any(Path("tests").is_relative_to(path) for path in paths)


def test_precommit_reports_for_every_pull_request() -> None:
    workflow = _load(WORKFLOWS / "pre-commit.yml")
    assert workflow["on"]["pull_request"] is None
    job = workflow["jobs"]["run"]
    assert job["name"] == "pre-commit"
    assert any(
        step.get("run") == "pre-commit run --all-files --show-diff-on-failure"
        for step in job["steps"]
    )


def test_hygiene_cancels_obsolete_runs_and_retains_delta_history() -> None:
    workflow = _load(WORKFLOWS / "repo-hygiene.yml")
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.ref" in workflow["concurrency"]["group"]
    jobs = workflow["jobs"]
    assert set(jobs) == {"lockfile-sync", "no-generated-artifacts", "large-files"}
    for name, job in jobs.items():
        assert 1 <= job["timeout-minutes"] <= 15
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        depth = checkout.get("with", {}).get("fetch-depth", 1)
        assert depth == (1 if name == "lockfile-sync" else 0)


def test_runtime_coverage_requires_pinned_sdk_tests_in_its_measured_interpreter():
    jobs = _load(WORKFLOWS / "ci.yml")["jobs"]
    job = jobs["coverage-tests"]
    steps = job["steps"]
    sdk_steps = [
        step
        for step in steps
        if "pip install" in step.get("run", "")
        and "inspect-judge-tests-py313.txt" in step["run"]
    ]
    assert len(sdk_steps) == 2
    install = next(
        step for step in sdk_steps if step["if"] == "${{ matrix.shard == 'runtime' }}"
    )
    examples = next(
        step
        for step in sdk_steps
        if step["if"] == "${{ startsWith(matrix.shard, 'examples') }}"
    )
    assert (
        "--require-hashes -r requirements/workflows/langfuse-sdk-tests-py313.txt"
        in examples["run"]
    )
    assert examples["run"].strip().endswith("python -m pip check")
    assert install["if"] == "${{ matrix.shard == 'runtime' }}"
    assert "python -m pip install --require-hashes -r " in install["run"]
    assert install["run"].strip().endswith("python -m pip check")
    collect = next(
        step for step in steps if "make coverage-collect-" in step.get("run", "")
    )
    assert steps.index(install) < steps.index(collect)
    assert steps.index(examples) < steps.index(collect)
    assert collect["env"]["INVARLOCK_REQUIRE_INSPECT_SDK"] == (
        "${{ (matrix.shard == 'runtime' || startsWith(matrix.shard, 'examples')) && '1' || '0' }}"
    )
    assert "INVARLOCK_REQUIRE_INSPECT_SDK" not in jobs["coverage"].get("env", {})


def test_full_verification_requires_sdk_tests_before_collecting_coverage():
    steps = _load(WORKFLOWS / "ci.yml")["jobs"]["verify-full"]["steps"]
    install = next(
        step
        for step in steps
        if "pip install" in step.get("run", "")
        and "inspect-judge-tests-py313.txt" in step["run"]
    )
    verify = next(step for step in steps if step.get("run") == "make verify")
    assert "python -m pip install --require-hashes -r " in install["run"]
    assert install["run"].strip().endswith("python -m pip check")
    assert "if" not in install
    assert steps.index(install) < steps.index(verify)
    assert verify["env"]["INVARLOCK_REQUIRE_INSPECT_SDK"] == "1"


def test_release_requires_sdk_tests_in_its_measured_interpreter():
    steps = _load(WORKFLOWS / "release.yml")["jobs"]["build_check"]["steps"]
    install = next(
        step
        for step in steps
        if step.get("name") == "Install measured SDK dependencies"
    )
    for lock in (
        "requirements/workflows/inspect-judge-tests-py313.txt",
        "requirements/workflows/langfuse-sdk-tests-py313.txt",
    ):
        assert f"python -m pip install --require-hashes -r {lock}" in install["run"]
    commands = install["run"].strip().splitlines()
    assert commands[-2:] == [
        "python -m pip check",
        "python -m pytest tests/cli/test_cli_surface.py -q",
    ]

    verify = next(
        item
        for item in steps
        if item.get("name") == "Run repository and supplemental behavior gates"
    )
    assert steps.index(verify) < steps.index(install)
    assert "env" not in verify
    coverage_job = _load(WORKFLOWS / "release.yml")["jobs"]["coverage_check"]
    coverage_steps = coverage_job["steps"]
    coverage_install = next(
        item
        for item in coverage_steps
        if item.get("name") == "Install measured coverage dependencies"
    )
    for lock in (
        "requirements/workflows/inspect-judge-tests-py313.txt",
        "requirements/workflows/langfuse-sdk-tests-py313.txt",
    ):
        assert (
            f"python -m pip install --require-hashes -r {lock}"
            in coverage_install["run"]
        )
    coverage = next(
        item
        for item in coverage_steps
        if item.get("name") == "Enforce release coverage"
    )
    assert coverage_steps.index(coverage_install) < coverage_steps.index(coverage)
    assert coverage_job["env"]["INVARLOCK_REQUIRE_INSPECT_SDK"] == "1"
    assert coverage_job["env"]["INVARLOCK_REQUIRE_LANGFUSE_SDK"] == "1"


def test_release_replays_installed_evaluator_campaigns_from_frozen_wheel():
    steps = _load(WORKFLOWS / "release.yml")["jobs"]["build_check"]["steps"]
    build = next(
        item for item in steps if item.get("name") == "Build first-party distributions"
    )
    ledger = next(
        item for item in steps if item.get("name") == "Record distribution digests"
    )
    replay = next(
        item
        for item in steps
        if item.get("name")
        == "Qualify installed evaluator integrations and retained campaigns"
    )
    digest_check = next(
        item
        for item in steps
        if item.get("name") == "Verify distribution digests before artifact upload"
    )
    assert replay["run"] == "PYTHON=python bash scripts/evaluator_parity_gate.sh"
    assert steps.index(build) < steps.index(ledger) < steps.index(replay)
    assert steps.index(replay) < steps.index(digest_check)


@pytest.mark.parametrize(
    "name", ["ci.yml", "docs-ci.yml", "codeql.yml", "container-front-door-smoke.yml"]
)
def test_promotion_uses_pr_checks_without_duplicate_branch_push(name):
    events = _load(WORKFLOWS / name)["on"]
    assert "release/v*" not in events["push"]["branches"]
    assert "staging/next" in events["push"]["branches"]
    assert "pull_request" in events
    destinations = (events["pull_request"] or {}).get("branches")
    assert destinations is None or "main" in destinations
    if name == "codeql.yml":
        assert "schedule" in events and "workflow_dispatch" in events
    if name == "ci.yml":
        assert events["push"]["tags"] == ["v*"]
