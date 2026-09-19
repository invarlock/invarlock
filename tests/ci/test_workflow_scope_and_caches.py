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
