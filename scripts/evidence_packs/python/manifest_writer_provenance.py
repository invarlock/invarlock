from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017
_JSON_READ_ERRORS = (OSError, TypeError, ValueError)
_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = _REPO_ROOT / "src"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class SourceRepoMetadataError(RuntimeError):
    """Raised when evidence-pack source provenance cannot be collected safely."""


def _ensure_repo_src_path() -> None:
    src = str(_REPO_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _repo_root() -> Path:
    return _REPO_ROOT


def _git_text(
    *args: str,
    repo_dir: Path | None = None,
    required: bool = True,
) -> str:
    resolved_repo_dir = repo_dir or _repo_root()
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
    resolved_repo_dir = repo_dir or _repo_root()
    remote_url = _git_text(
        "config",
        "--get",
        "remote.origin.url",
        repo_dir=resolved_repo_dir,
        required=False,
    )
    try:
        commit = _git_text("rev-parse", "HEAD", repo_dir=resolved_repo_dir)
        branch = _git_text(
            "rev-parse", "--abbrev-ref", "HEAD", repo_dir=resolved_repo_dir
        )
        describe = _git_text(
            "describe",
            "--tags",
            "--always",
            "--dirty",
            repo_dir=resolved_repo_dir,
        )
        dirty = bool(_git_text("status", "--porcelain", repo_dir=resolved_repo_dir))
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


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except _JSON_READ_ERRORS:
        return None


def _utc_now() -> str:
    return dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_number(value: str | None) -> int | float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except _COERCE_ERRORS:
        return None
    if number.is_integer():
        return int(number)
    return number


def _load_run_state_environment(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "state" / "environment.json"
    if not state_path.is_file():
        return {}
    payload = _load_json(state_path)
    return payload if isinstance(payload, dict) else {}


def build_environment_payload(run_dir: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if run_dir is not None:
        payload.update(_load_run_state_environment(run_dir))

    payload.setdefault("recorded_at", _utc_now())
    payload.setdefault("platform", platform.platform())
    payload.setdefault("python_version", platform.python_version())
    payload.setdefault("gpu_name", os.environ.get("PACK_GPU_NAME", ""))
    payload.setdefault("gpu_count", _maybe_number(os.environ.get("PACK_GPU_COUNT")))
    payload.setdefault(
        "gpu_memory_gb", _maybe_number(os.environ.get("PACK_GPU_MEM_GB"))
    )
    payload.setdefault(
        "fp8_native_support",
        _truthy(os.environ.get("FP8_NATIVE_SUPPORT")),
    )
    return payload


def write_source_repo_metadata(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_source_repo_payload()
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_environment_metadata(*, out_path: Path, run_dir: Path | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(build_environment_payload(run_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "SourceRepoMetadataError",
    "_ensure_repo_src_path",
    "_git_text",
    "_load_run_state_environment",
    "_maybe_number",
    "_repo_root",
    "_snapshot_marker_payload",
    "_truthy",
    "_utc_now",
    "build_environment_payload",
    "build_source_repo_payload",
    "write_environment_metadata",
    "write_source_repo_metadata",
]
