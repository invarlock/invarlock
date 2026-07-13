"""Immutable clean-pruning selection bundle and evidence-tree snapshots."""

from __future__ import annotations

import math
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from invarlock.clean_pruning_selection_artifacts import (
    _assert_pruning_replay,
    _assert_report_binding,
    _assert_report_native_execution_provenance,
    _assert_report_runtime_manifest,
    _assert_runtime_reload_proof,
    _read_referenced_json_snapshot,
)
from invarlock.clean_pruning_selection_common import (
    CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
    CleanPruningSelectionBundleSnapshot,
    CleanPruningSelectionEvidenceError,
    _finite,
    _identity,
    _scope,
    strict_json_object_snapshot,
)
from invarlock.clean_pruning_selection_contract import (
    validate_clean_pruning_execution_receipt,
    verify_clean_pruning_selection_bundle,
)


def _verify_candidate_artifacts(
    entry: Mapping[str, object],
    *,
    evidence_root: Path,
    globally_referenced_paths: set[str],
) -> dict[str, bytes]:
    """Authenticate all retained evidence for every candidate in one receipt."""

    selected = cast(Mapping[str, object], entry["selected_entry"])
    receipt = cast(Mapping[str, object], selected["selection_receipt"])
    original_model_key = cast(str, entry["original_model_key"])
    baseline_identity = cast(Mapping[str, str], receipt["baseline_identity"])
    selection_config = cast(Mapping[str, object], receipt["selection_config"])
    selection_config_sha256 = cast(str, receipt["selection_config_sha256"])
    retained: dict[str, bytes] = {}
    for candidate in cast(Sequence[Mapping[str, object]], receipt["candidates"]):
        candidate_id = cast(str, candidate["candidate_id"])
        pruning = cast(Mapping[str, object], candidate["pruning"])
        evaluation = cast(Mapping[str, object], candidate["evaluation"])
        execution_ref = cast(Mapping[str, object], evaluation["execution"])
        replay_ref = cast(Mapping[str, object], evaluation["replay"])
        runtime_ref = cast(Mapping[str, object], evaluation["runtime"])
        reports = cast(Sequence[Mapping[str, object]], evaluation["reports"])
        references: list[tuple[str, Mapping[str, object]]] = [
            ("execution receipt", execution_ref),
            ("pruning replay", replay_ref),
            ("runtime reload proof", runtime_ref),
        ]
        for repeat_index, report_run in enumerate(reports):
            references.extend(
                (
                    (
                        f"report repeat {repeat_index}",
                        cast(Mapping[str, object], report_run["report"]),
                    ),
                    (
                        f"runtime manifest repeat {repeat_index}",
                        cast(Mapping[str, object], report_run["runtime_manifest"]),
                    ),
                )
            )
        paths = [cast(str, reference["path"]) for _, reference in references]
        if len(paths) != len(set(paths)):
            raise CleanPruningSelectionEvidenceError(
                "candidate evidence references reuse one sidecar path"
            )
        if any(path in globally_referenced_paths for path in paths):
            raise CleanPruningSelectionEvidenceError(
                "clean pruning candidates must not reuse retained evidence sidecars"
            )
        globally_referenced_paths.update(paths)
        payloads: dict[str, dict[str, object]] = {}
        for name, reference in references:
            raw, payload = _read_referenced_json_snapshot(
                reference,
                evidence_root=evidence_root,
                label=f"candidate {candidate_id} {name} sidecar",
            )
            payloads[name] = payload
            retained[cast(str, reference["path"])] = raw
        execution_sha256 = cast(str, execution_ref["sha256"])
        validate_clean_pruning_execution_receipt(
            payloads["execution receipt"],
            expected_model_key=original_model_key,
            expected_candidate_id=candidate_id,
            expected_pruning=pruning,
            expected_baseline_identity=baseline_identity,
            expected_selection_config=selection_config,
        )
        measured_losses: list[float] = []
        for repeat_index, report_run in enumerate(reports):
            report_ref = cast(Mapping[str, object], report_run["report"])
            manifest_ref = cast(Mapping[str, object], report_run["runtime_manifest"])
            report_name = f"report repeat {repeat_index}"
            manifest_name = f"runtime manifest repeat {repeat_index}"
            report = payloads[report_name]
            artifact_identity = cast(Mapping[str, str], report_ref["artifact_identity"])
            _assert_report_native_execution_provenance(
                report,
                execution_receipt_sha256=execution_sha256,
                selection_config=selection_config,
                original_model_key=original_model_key,
                candidate_id=candidate_id,
                pruning=pruning,
                baseline_identity=baseline_identity,
                repeat_index=repeat_index,
            )
            measured_losses.append(
                _assert_report_binding(
                    report,
                    selection_config_sha256=selection_config_sha256,
                    execution_receipt_sha256=execution_sha256,
                    original_model_key=original_model_key,
                    candidate_id=candidate_id,
                    pruning=pruning,
                    baseline_identity=baseline_identity,
                    artifact_identity=artifact_identity,
                    repeat_index=repeat_index,
                )
            )
            _assert_report_runtime_manifest(
                report_bytes=retained[cast(str, report_ref["path"])],
                report=report,
                manifest=payloads[manifest_name],
                report_reference=report_ref,
                manifest_reference=manifest_ref,
                execution_receipt_sha256=execution_sha256,
                selection_config_sha256=selection_config_sha256,
                original_model_key=original_model_key,
                candidate_id=candidate_id,
                pruning=pruning,
                baseline_identity=baseline_identity,
                repeat_index=repeat_index,
            )
        measured_quality_loss = math.fsum(measured_losses) / len(measured_losses)
        declared_quality_loss = _finite(
            cast(Mapping[str, object], evaluation["metrics"])["quality_loss"],
            label="candidate quality_loss",
        )
        if not math.isclose(
            declared_quality_loss,
            measured_quality_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CleanPruningSelectionEvidenceError(
                "candidate metric does not match retained strict reports"
            )
        artifact_identity = cast(
            Mapping[str, str],
            cast(Mapping[str, object], reports[0]["report"])["artifact_identity"],
        )
        _assert_pruning_replay(
            payloads["pruning replay"],
            pruning=pruning,
            baseline_identity=baseline_identity,
            artifact_identity=artifact_identity,
        )
        _assert_runtime_reload_proof(
            payloads["runtime reload proof"], artifact_identity=artifact_identity
        )
    return retained


def snapshot_clean_pruning_selection_bundle_file(
    bundle_path: Path, *, evidence_root: Path | None = None
) -> CleanPruningSelectionBundleSnapshot:
    """Authenticate bundle and all candidate sidecars from immutable snapshots."""

    bundle_bytes, raw_bundle = strict_json_object_snapshot(
        bundle_path, label="clean pruning selection bundle"
    )
    bundle = verify_clean_pruning_selection_bundle(raw_bundle)
    root = evidence_root if evidence_root is not None else bundle_path.parent
    retained: dict[str, bytes] = {}
    globally_referenced_paths: set[str] = set()
    for entry in cast(Sequence[Mapping[str, object]], bundle["entries"]):
        for path, raw in _verify_candidate_artifacts(
            entry,
            evidence_root=root,
            globally_referenced_paths=globally_referenced_paths,
        ).items():
            if path in retained:
                raise CleanPruningSelectionEvidenceError(
                    "clean pruning bundle reuses a candidate sidecar"
                )
            retained[path] = raw
    return CleanPruningSelectionBundleSnapshot(
        bundle=bundle,
        bundle_bytes=bundle_bytes,
        sidecar_bytes=retained,
    )


def verify_clean_pruning_selection_bundle_file(
    bundle_path: Path, *, evidence_root: Path | None = None
) -> dict[str, object]:
    """Verify all receipt, replay, report, runtime, and digest bindings."""

    return snapshot_clean_pruning_selection_bundle_file(
        bundle_path, evidence_root=evidence_root
    ).bundle


def _snapshot_tree_inventory(snapshot_root: Path) -> tuple[set[str], set[str]]:
    """Enumerate a staging tree without accepting symlinks or special files."""

    try:
        root_mode = snapshot_root.lstat().st_mode
    except OSError as exc:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection snapshot root is missing"
        ) from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection snapshot root must be a regular directory"
        )
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, relative: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise CleanPruningSelectionEvidenceError(
                "clean pruning selection snapshot tree cannot be enumerated"
            ) from exc
        for child in children:
            child_relative = relative / child.name
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise CleanPruningSelectionEvidenceError(
                    "clean pruning selection snapshot entry is unavailable"
                ) from exc
            display = child_relative.as_posix()
            if stat.S_ISLNK(mode):
                raise CleanPruningSelectionEvidenceError(
                    f"clean pruning selection snapshot must not contain symlink {display}"
                )
            if stat.S_ISDIR(mode):
                directories.add(display)
                visit(child, child_relative)
            elif stat.S_ISREG(mode):
                files.add(display)
            else:
                raise CleanPruningSelectionEvidenceError(
                    "clean pruning selection snapshot contains non-regular entry "
                    f"{display}"
                )

    visit(snapshot_root, Path())
    return files, directories


def verify_clean_pruning_selection_snapshot_tree(
    snapshot_root: Path,
) -> CleanPruningSelectionBundleSnapshot:
    """Verify the exact staged ``metadata/clean_pruning_selection`` bridge.

    The stage root must contain ``bundle.json`` and only the sidecars referenced
    by that bundle, at the same safe relative paths.  Callers should atomically
    publish the bytes returned by :func:`snapshot_clean_pruning_selection_bundle_file`
    into this layout, then call this function before accepting the final pack.
    This keeps selection evidence outside the pruning replay schema while
    rejecting stale files, copied extras, and every symlinked staging path.
    """

    bundle_path = snapshot_root / CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME
    snapshot = snapshot_clean_pruning_selection_bundle_file(
        bundle_path, evidence_root=snapshot_root
    )
    expected_files = {
        CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
        *snapshot.sidecar_bytes,
    }
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files, actual_directories = _snapshot_tree_inventory(snapshot_root)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extras = sorted(actual_files - expected_files)
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection snapshot file inventory mismatch "
            f"(missing={missing}, extras={extras})"
        )
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extras = sorted(actual_directories - expected_directories)
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection snapshot directory inventory mismatch "
            f"(missing={missing}, extras={extras})"
        )
    return snapshot


def selected_clean_pruning_entry_for(
    bundle: Mapping[str, object], *, model_key: str, requested_scope: str = ""
) -> dict[str, object]:
    """Return one verified selected pruning entry for a model and optional scope."""

    verified = verify_clean_pruning_selection_bundle(bundle)
    matches = [
        entry
        for entry in cast(Sequence[dict[str, object]], verified["entries"])
        if entry["original_model_key"] == model_key
    ]
    if len(matches) != 1:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection bundle has no unique matching model entry"
        )
    entry = matches[0]
    selected = cast(Mapping[str, object], entry["selected_entry"])
    if requested_scope and selected["scope"] != _scope(
        requested_scope, label="requested_scope"
    ):
        raise CleanPruningSelectionEvidenceError(
            "requested scope does not match the selected pruning candidate"
        )
    return entry


def selected_clean_pruning_artifact_identity_for(
    bundle: Mapping[str, object], *, model_key: str
) -> dict[str, str]:
    """Return the promoted candidate tree identity from a verified v1 receipt.

    A final clean-pruning pack must promote this exact candidate artifact.  An
    exact re-materialization is acceptable only if its checkpoint-tree identity
    equals this value; changing selection metadata alone cannot authorize a
    different output tree.
    """

    entry = selected_clean_pruning_entry_for(bundle, model_key=model_key)
    selected = cast(Mapping[str, object], entry["selected_entry"])
    receipt = cast(Mapping[str, object], selected["selection_receipt"])
    evaluation = cast(Mapping[str, object], receipt["selected_evaluation"])
    replay = cast(Mapping[str, object], evaluation["replay"])
    return _identity(
        replay["artifact_identity"], label="selected pruning replay.artifact_identity"
    )


def referenced_clean_pruning_candidate_paths(bundle: Mapping[str, object]) -> list[str]:
    """Return all safe retained candidate paths for snapshot-based staging."""

    verified = verify_clean_pruning_selection_bundle(bundle)
    paths: set[str] = set()
    for entry in cast(Sequence[Mapping[str, object]], verified["entries"]):
        selected = cast(Mapping[str, object], entry["selected_entry"])
        receipt = cast(Mapping[str, object], selected["selection_receipt"])
        for candidate in cast(Sequence[Mapping[str, object]], receipt["candidates"]):
            evaluation = cast(Mapping[str, object], candidate["evaluation"])
            for name in ("execution", "replay", "runtime"):
                paths.add(
                    cast(str, cast(Mapping[str, object], evaluation[name])["path"])
                )
            for report_run in cast(
                Sequence[Mapping[str, object]], evaluation["reports"]
            ):
                paths.add(
                    cast(
                        str,
                        cast(Mapping[str, object], report_run["report"])["path"],
                    )
                )
                paths.add(
                    cast(
                        str,
                        cast(Mapping[str, object], report_run["runtime_manifest"])[
                            "path"
                        ],
                    )
                )
    return sorted(paths)
