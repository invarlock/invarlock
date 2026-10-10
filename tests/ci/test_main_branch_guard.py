"""Exercise the actual main-branch workflow gate without GitHub mutations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(".github/workflows/main-branch-guard.yml")


def _workflow() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text())
    if True in document:
        document["on"] = document.pop(True)
    return document


def _run_guard(tmp_path: Path, **changes: str) -> subprocess.CompletedProcess[str]:
    job = next(iter(_workflow()["jobs"].values()))
    command = job["steps"][-1]["run"]
    environment = {
        "PATH": os.defpath,
        "GITHUB_REPOSITORY": "invarlock/invarlock",
        "BASE_REPO": "invarlock/invarlock",
        "HEAD_REPO": "invarlock/invarlock",
        "BASE_REF": "main",
        "HEAD_REF": "release/v0.17.0",
        "PR_AUTHOR": "maintainer",
        "PR_URL": "https://github.com/invarlock/invarlock/pull/1",
        **changes,
    }
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-c", command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "head", ["work/promote-v0.17.0", "staging/next", "main", "dependabot/uv/update"]
)
def test_rejects_non_release_heads_for_every_author(tmp_path: Path, head: str) -> None:
    result = _run_guard(tmp_path, HEAD_REF=head)
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize("head", ["release/v0.17.0", "release/v1.2.3-rc.1"])
def test_accepts_versioned_release_heads(tmp_path: Path, head: str) -> None:
    result = _run_guard(tmp_path, HEAD_REF=head)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"HEAD_REPO": "other/invarlock"},
        {"HEAD_REPO": ""},
        {"BASE_REPO": "other/invarlock"},
        {"BASE_REPO": ""},
        {"BASE_REF": "staging/next"},
        {"HEAD_REF": "release/v0.17"},
        {"HEAD_REF": "release/v0.17.0/extra"},
        {"HEAD_REF": "release/vnext"},
        {"HEAD_REF": ""},
    ],
)
def test_rejects_wrong_repository_target_or_malformed_version(
    tmp_path: Path, changes: dict[str, str]
) -> None:
    assert _run_guard(tmp_path, **changes).returncode != 0


def test_branch_metadata_is_never_executed(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, HEAD_REF="release/v0.17.0$(touch injected)")
    assert result.returncode != 0
    assert not (tmp_path / "injected").exists()


def test_required_job_cannot_skip_disallowed_heads() -> None:
    workflow = _workflow()
    assert workflow["on"] == {
        "pull_request": {
            "branches": ["main"],
            "types": [
                "opened",
                "synchronize",
                "reopened",
                "edited",
                "ready_for_review",
            ],
        }
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"release-branch"}
    job = workflow["jobs"]["release-branch"]
    assert "if" not in job and not job.get("continue-on-error", False)
    assert 1 <= job["timeout-minutes"] <= 5
    for step in job["steps"]:
        assert "if" not in step and not step.get("continue-on-error", False)
        assert "uses" not in step
        assert "${{" not in step["run"]
    assert job["steps"][-1]["env"] == {
        "HEAD_REF": "${{ github.event.pull_request.head.ref }}",
        "HEAD_REPO": "${{ github.event.pull_request.head.repo.full_name }}",
        "BASE_REF": "${{ github.event.pull_request.base.ref }}",
        "BASE_REPO": "${{ github.event.pull_request.base.repo.full_name }}",
    }
