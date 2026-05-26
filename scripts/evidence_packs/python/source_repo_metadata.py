from __future__ import annotations

import os
import re
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


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


def _snapshot_marker_payload(repo_dir: Path) -> dict[str, Any] | None:
    marker_candidates: list[Path] = []
    explicit_marker = os.environ.get("INVARLOCK_SOURCE_REPO_MARKER", "").strip()
    if explicit_marker:
        marker_candidates.append(Path(explicit_marker))
    marker_candidates.append(repo_dir / "GPU_RUN_SOURCE.txt")

    env_values = {
        "source_uri": os.environ.get("INVARLOCK_SOURCE_REPO_URI", "").strip(),
        "source_commit": os.environ.get("INVARLOCK_SOURCE_COMMIT", "").strip(),
        "source_branch": os.environ.get("INVARLOCK_SOURCE_BRANCH", "").strip(),
        "source_describe": os.environ.get("INVARLOCK_SOURCE_DESCRIBE", "").strip(),
        "source_dirty": os.environ.get("INVARLOCK_SOURCE_DIRTY", "").strip(),
    }

    for marker_path in marker_candidates:
        marker_values = _read_key_value_file(marker_path)
        values = {**marker_values, **{k: v for k, v in env_values.items() if v}}
        commit = values.get("source_commit") or values.get("commit") or ""
        commit = commit.strip()
        if not _COMMIT_RE.match(commit):
            continue

        uri = values.get("source_uri") or values.get("uri") or repo_dir.as_uri()
        branch = (
            values.get("source_branch") or values.get("branch") or "detached-snapshot"
        )
        describe = (
            values.get("source_describe") or values.get("describe") or commit[:12]
        )
        dirty = _parse_bool(
            values.get("source_dirty") or values.get("dirty"), default=False
        )

        return {
            "uri": uri,
            "commit": commit,
            "branch": branch,
            "describe": describe,
            "dirty": dirty,
            "metadata_source": str(marker_path),
        }

    return None


def build_source_repo_payload(repo_dir: Path | None = None) -> dict[str, Any]:
    resolved_repo_dir = repo_dir or repo_root()
    remote_url = git_text(
        "config",
        "--get",
        "remote.origin.url",
        repo_dir=resolved_repo_dir,
        required=False,
    )
    try:
        commit = git_text("rev-parse", "HEAD", repo_dir=resolved_repo_dir)
        branch = git_text(
            "rev-parse", "--abbrev-ref", "HEAD", repo_dir=resolved_repo_dir
        )
        describe = git_text(
            "describe",
            "--tags",
            "--always",
            "--dirty",
            repo_dir=resolved_repo_dir,
        )
        dirty = bool(git_text("status", "--porcelain", repo_dir=resolved_repo_dir))
    except SourceRepoMetadataError:
        marker_payload = _snapshot_marker_payload(resolved_repo_dir)
        if marker_payload is not None:
            return marker_payload
        raise

    return {
        "uri": f"git+{remote_url}" if remote_url else resolved_repo_dir.as_uri(),
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }
