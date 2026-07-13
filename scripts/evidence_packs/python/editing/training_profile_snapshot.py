# ruff: noqa: E402  # Direct script execution must establish package import roots first.
"""Stage a sealed immutable training-profile snapshot for an evidence pack.

The snapshot records the exact profile used by a training artifact-replay proof
plus an explicit reviewed scope. Scope is required as input: this tool never infers
semantic scope from adapter-module names.  Its byte digest is intended for the
``training_profile.snapshot_sha256`` field of a typed scenario record.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)

if __package__ not in {None, ""}:
    from .training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        TRAINING_PROFILES_SCHEMA,
        training_profile_errors,
    )
else:  # pragma: no cover - direct script-path loading
    from scripts.evidence_packs.python.editing.training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        TRAINING_PROFILES_SCHEMA,
        training_profile_errors,
    )


TRAINING_PROFILE_SNAPSHOT_SCHEMA = (
    "invarlock/evidence-pack-training-profile-snapshot-v1"
)
_SCOPES = frozenset({"all", "attn", "ffn"})


class TrainingProfileSnapshotError(RuntimeError):
    """Raised when an immutable profile cannot become pack evidence."""


def _canonical_snapshot_bytes(payload: dict[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - closed inputs
        raise TrainingProfileSnapshotError(
            "training profile snapshot cannot be canonicalized"
        ) from exc


def _load_verified_profile(
    *,
    profile_id: str,
    profiles_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    try:
        _, document = read_json_object_snapshot(
            profiles_path,
            label="immutable training profiles",
        )
    except (OSError, StrictJsonError) as exc:
        raise TrainingProfileSnapshotError(
            "immutable training profiles are unavailable or not strict JSON"
        ) from exc
    if set(document) != {"schema", "profiles"} or document.get("schema") != (
        TRAINING_PROFILES_SCHEMA
    ):
        raise TrainingProfileSnapshotError(
            "immutable training profiles have unknown schema"
        )
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise TrainingProfileSnapshotError("immutable training profiles are invalid")
    profile = profiles.get(profile_id)
    if not isinstance(profile, dict):
        raise TrainingProfileSnapshotError("requested training profile is unavailable")
    errors = training_profile_errors(
        profile_id,
        profile,
        repo_root=repo_root,
        verify_data_file=True,
    )
    if errors:
        raise TrainingProfileSnapshotError(
            "immutable training profile is invalid: " + "; ".join(errors)
        )
    return dict(profile)


def _publish_no_replace(path: Path, payload: bytes) -> None:
    try:
        if path.exists() or path.is_symlink():
            existing = read_regular_file_bytes(path, label="existing profile snapshot")
            if existing == payload:
                return
            raise TrainingProfileSnapshotError(
                "refusing to overwrite a different training profile snapshot"
            )
    except StrictJsonError as exc:
        raise TrainingProfileSnapshotError(
            "existing training profile snapshot is unavailable"
        ) from exc

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_stat = path.parent.lstat()
    except OSError as exc:
        raise TrainingProfileSnapshotError(
            "training profile snapshot parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise TrainingProfileSnapshotError(
            "training profile snapshot parent must be a regular directory"
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".training-profile-snapshot.", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            existing = read_regular_file_bytes(path, label="existing profile snapshot")
            if existing != payload:
                raise TrainingProfileSnapshotError(
                    "refusing to overwrite a different training profile snapshot"
                ) from None
    except OSError as exc:
        raise TrainingProfileSnapshotError(
            "could not atomically publish training profile snapshot"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def produce_training_profile_snapshot(
    *,
    profile_id: str,
    scope: str,
    output_path: Path,
    profiles_path: Path = DEFAULT_TRAINING_PROFILES_PATH,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, object]:
    """Validate a reviewed profile and publish its deterministic pack snapshot."""

    if not isinstance(profile_id, str) or not profile_id:
        raise TrainingProfileSnapshotError("training profile id is invalid")
    if scope not in _SCOPES:
        raise TrainingProfileSnapshotError(
            "training profile scope must be all, attn, or ffn"
        )
    profile = _load_verified_profile(
        profile_id=profile_id,
        profiles_path=profiles_path,
        repo_root=repo_root,
    )
    profile_sha256 = profile.get("profile_sha256")
    if not isinstance(profile_sha256, str):  # covered above; preserves typing
        raise TrainingProfileSnapshotError("training profile digest is invalid")
    snapshot: dict[str, object] = {
        "schema": TRAINING_PROFILE_SNAPSHOT_SCHEMA,
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "scope": scope,
        "profile": profile,
    }
    payload = _canonical_snapshot_bytes(snapshot)
    _publish_no_replace(output_path, payload)
    return {
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "snapshot_path": str(output_path),
        "snapshot_sha256": sha256_prefixed(payload),
        "scope": scope,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--scope", required=True, choices=sorted(_SCOPES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--profiles-path",
        type=Path,
        default=DEFAULT_TRAINING_PROFILES_PATH,
        help="immutable training-profile JSON document",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="repository root used to authenticate the immutable training data",
    )
    args = parser.parse_args(argv)
    try:
        result = produce_training_profile_snapshot(
            profile_id=args.profile_id,
            scope=args.scope,
            output_path=args.out,
            profiles_path=args.profiles_path,
            repo_root=args.repo_root,
        )
    except TrainingProfileSnapshotError as exc:
        parser.error(str(exc))
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "TRAINING_PROFILE_SNAPSHOT_SCHEMA",
    "TrainingProfileSnapshotError",
    "produce_training_profile_snapshot",
]
