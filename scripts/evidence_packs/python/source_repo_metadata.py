from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class SourceRepoMetadataError(RuntimeError):
    """Raised when evidence-pack source provenance cannot be collected safely."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_text(
    *args: str,
    repo_dir: Path | None = None,
    required: bool = True,
) -> str:
    resolved_repo_dir = repo_dir or repo_root()
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=resolved_repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        if required:
            raise SourceRepoMetadataError(
                "git is required to collect evidence-pack source provenance."
            ) from exc
        return ""
    if proc.returncode != 0:
        if not required:
            return ""
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise SourceRepoMetadataError(
            "git "
            + " ".join(args)
            + " failed while collecting evidence-pack source provenance: "
            + detail
        )
    return proc.stdout.strip()


def build_source_repo_payload(repo_dir: Path | None = None) -> dict[str, Any]:
    resolved_repo_dir = repo_dir or repo_root()
    remote_url = git_text(
        "config",
        "--get",
        "remote.origin.url",
        repo_dir=resolved_repo_dir,
        required=False,
    )
    commit = git_text("rev-parse", "HEAD", repo_dir=resolved_repo_dir)
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD", repo_dir=resolved_repo_dir)
    describe = git_text(
        "describe",
        "--tags",
        "--always",
        "--dirty",
        repo_dir=resolved_repo_dir,
    )
    dirty = bool(git_text("status", "--porcelain", repo_dir=resolved_repo_dir))

    return {
        "uri": f"git+{remote_url}" if remote_url else resolved_repo_dir.as_uri(),
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }
