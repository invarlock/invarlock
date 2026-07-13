# ruff: noqa: E402, I001
"""Validate, stage, and resolve clean-pruning selection evidence.

This public bridge is deliberately consumer-only. It snapshots an already
produced selection bundle, validates its retained sidecars, and resolves the
selected parameters. Campaign execution, artifact copying, and promotion
belong to an external orchestration system.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - direct shell path
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct shell path
    sys.path.insert(0, str(REPO_ROOT))

from invarlock.clean_pruning_selection_common import (
    CleanPruningSelectionBundleSnapshot,
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_pruning_selection_contracts.snapshot import (
    selected_clean_pruning_artifact_identity_for,
    selected_clean_pruning_entry_for,
    snapshot_clean_pruning_selection_bundle_file,
    verify_clean_pruning_selection_snapshot_tree,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_regular_file_bytes,
    sha256_prefixed,
)

STAGED_SELECTION_BUNDLE = "bundle.json"


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        return cast(bytes, read_regular_file_bytes(path, label=label))
    except StrictJsonError as exc:
        raise CleanPruningSelectionEvidenceError(str(exc)) from exc


def _regular_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CleanPruningSelectionEvidenceError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CleanPruningSelectionEvidenceError(f"{label} must be a regular directory")


def _atomic_write_identical_or_fail(destination: Path, payload: bytes) -> None:
    """Write one immutable sidecar, never replacing different evidence."""

    if destination.exists() or destination.is_symlink():
        if (
            _regular_file_bytes(destination, label="existing selection destination")
            != payload
        ):
            raise CleanPruningSelectionEvidenceError(
                "refusing to overwrite different clean-pruning evidence"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(destination.parent, label="selection destination parent")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise CleanPruningSelectionEvidenceError(
            f"could not atomically stage clean-pruning evidence: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_destination(root: Path, relative: str) -> Path:
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise CleanPruningSelectionEvidenceError(
            "clean-pruning evidence reference is not a safe relative path"
        )
    return root.joinpath(*relative.split("/"))


def stage_clean_pruning_selection_bundle(
    *,
    bundle_path: Path,
    destination: Path,
    evidence_root: Path | None = None,
) -> Path:
    """Stage exact verified sidecar bytes at the bounded pack-facing layout."""

    source = snapshot_clean_pruning_selection_bundle_file(
        bundle_path, evidence_root=evidence_root
    )
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        existing = verify_clean_pruning_selection_snapshot_tree(destination)
        if existing.bundle_bytes != source.bundle_bytes or dict(
            existing.sidecar_bytes
        ) != dict(source.sidecar_bytes):
            raise CleanPruningSelectionEvidenceError(
                "existing clean-pruning staging tree differs from verified source"
            )
        return destination / STAGED_SELECTION_BUNDLE
    destination.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(destination.parent, label="clean-pruning staging parent")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _atomic_write_identical_or_fail(
            temporary / STAGED_SELECTION_BUNDLE, source.bundle_bytes
        )
        for relative, raw in source.sidecar_bytes.items():
            _atomic_write_identical_or_fail(_safe_destination(temporary, relative), raw)
        staged = verify_clean_pruning_selection_snapshot_tree(temporary)
        if staged.bundle_bytes != source.bundle_bytes or dict(
            staged.sidecar_bytes
        ) != dict(source.sidecar_bytes):
            raise CleanPruningSelectionEvidenceError(
                "clean-pruning staging snapshot changed before publication"
            )
        os.replace(temporary, destination)
    except OSError as exc:
        raise CleanPruningSelectionEvidenceError(
            f"could not publish clean-pruning staging tree: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / STAGED_SELECTION_BUNDLE


def _selected_entry(
    *, bundle_path: Path, evidence_root: Path, model_key: str, requested_scope: str
) -> tuple[CleanPruningSelectionBundleSnapshot, dict[str, object]]:
    snapshot = snapshot_clean_pruning_selection_bundle_file(
        bundle_path, evidence_root=evidence_root
    )
    entry = selected_clean_pruning_entry_for(
        snapshot.bundle,
        model_key=model_key,
        requested_scope=requested_scope,
    )
    return snapshot, entry


def resolve_clean_pruning_selection(
    *, bundle_path: Path, evidence_root: Path, model_key: str, requested_scope: str = ""
) -> dict[str, object]:
    """Return only verifier-selected magnitude-pruning parameters."""

    snapshot, entry = _selected_entry(
        bundle_path=bundle_path,
        evidence_root=evidence_root,
        model_key=model_key,
        requested_scope=requested_scope,
    )
    selected = entry["selected_entry"]
    if not isinstance(selected, Mapping):  # pragma: no cover - verifier normalized
        raise CleanPruningSelectionEvidenceError("selected pruning entry is invalid")
    receipt = selected.get("selection_receipt")
    if not isinstance(receipt, Mapping):  # pragma: no cover - verifier normalized
        raise CleanPruningSelectionEvidenceError("selected pruning receipt is invalid")
    pruning = receipt.get("selected_pruning")
    if not isinstance(pruning, Mapping):  # pragma: no cover - verifier normalized
        raise CleanPruningSelectionEvidenceError(
            "selected pruning specification is invalid"
        )
    sparsity = pruning.get("target_sparsity")
    scope = pruning.get("scope")
    if (
        isinstance(sparsity, bool)
        or not isinstance(sparsity, int | float)
        or not 0.0 < float(sparsity) < 1.0
        or not isinstance(scope, str)
    ):
        raise CleanPruningSelectionEvidenceError(
            "selected pruning parameters are invalid"
        )
    artifact_identity = selected_clean_pruning_artifact_identity_for(
        cast(Mapping[str, object], snapshot.bundle), model_key=model_key
    )
    baseline_identity = receipt.get("baseline_identity")
    if not isinstance(baseline_identity, Mapping):
        raise CleanPruningSelectionEvidenceError(
            "selected pruning baseline identity is invalid"
        )
    return {
        "status": "selected",
        "reason": "receipt_bound_clean_pruning_selection_v1",
        "edit_type": "magnitude_prune",
        "requested_edit_type": "magnitude_prune",
        "param1": repr(float(sparsity)),
        "param2": "",
        "scope": scope,
        "version": "clean",
        "edit_dir_name": "clean_magnitude_prune",
        "selected_candidate_id": receipt.get("selected_candidate_id"),
        "candidate_set_sha256": receipt.get("candidate_set_sha256"),
        "selection_bundle_sha256": sha256_prefixed(snapshot.bundle_bytes),
        "artifact_identity": dict(artifact_identity),
        "baseline_identity": dict(baseline_identity),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--bundle", required=True, type=Path)
    stage.add_argument("--dest", required=True, type=Path)
    stage.add_argument("--evidence-root", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--evidence-root", required=True, type=Path)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--bundle", required=True, type=Path)
    resolve.add_argument("--model-key", required=True)
    resolve.add_argument("--requested-scope", default="")
    resolve.add_argument("--evidence-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "stage":
        print(
            stage_clean_pruning_selection_bundle(
                bundle_path=args.bundle,
                destination=args.dest,
                evidence_root=args.evidence_root,
            )
        )
        return 0
    if args.command == "verify":
        snapshot = verify_clean_pruning_selection_snapshot_tree(args.evidence_root)
        expected = args.evidence_root / STAGED_SELECTION_BUNDLE
        if expected != args.bundle or snapshot.bundle_bytes != _regular_file_bytes(
            args.bundle, label="staged clean-pruning bundle"
        ):
            raise CleanPruningSelectionEvidenceError(
                "staged clean-pruning bundle path or bytes are inconsistent"
            )
        return 0
    if args.command == "resolve":
        root = args.evidence_root or args.bundle.parent
        print(
            json.dumps(
                resolve_clean_pruning_selection(
                    bundle_path=args.bundle,
                    evidence_root=root,
                    model_key=args.model_key,
                    requested_scope=args.requested_scope,
                ),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover - direct CLI entry point
    try:
        raise SystemExit(_main())
    except CleanPruningSelectionEvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
