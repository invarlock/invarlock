"""Immutable clean-selection bundle and referenced-sidecar snapshots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from invarlock.clean_selection.artifacts import (
    _assert_candidate_replay_runtime,
    _assert_eligible_report,
    _assert_report_native_execution_provenance,
    _assert_report_runtime_manifest,
    _execution_receipt,
    _path_below,
)
from invarlock.clean_selection.bundle import verify_selection_bundle
from invarlock.clean_selection.common import (
    CleanSelectionEvidenceError,
    SelectionBundleSnapshot,
    _finite,
    _scope,
    sha256_prefixed,
    strict_json_object_snapshot,
)


def _read_referenced_json_snapshot(
    reference: Mapping[str, object],
    *,
    evidence_root: Path,
    label: str,
) -> tuple[bytes, dict[str, object]]:
    """Read one retained sidecar once and authenticate those exact bytes."""

    path = _path_below(
        evidence_root,
        cast(str, reference["path"]),
        label=label,
    )
    raw, payload = strict_json_object_snapshot(path, label=label)
    if sha256_prefixed(raw) != reference["sha256"]:
        raise CleanSelectionEvidenceError(f"{label} digest mismatch")
    return raw, payload


def _verify_candidate_artifacts(
    entry: Mapping[str, object], evidence_root: Path
) -> dict[str, bytes]:
    receipt = cast(
        Mapping[str, object],
        cast(Mapping[str, object], entry["selected_entry"])["selection_receipt"],
    )
    model_key = cast(str, entry["original_model_key"])
    baseline = cast(Mapping[str, str], receipt["baseline_identity"])
    config = cast(Mapping[str, object], receipt["selection_config"])
    config_digest = cast(str, receipt["selection_config_sha256"])
    retained_bytes: dict[str, bytes] = {}
    for candidate in cast(Sequence[Mapping[str, object]], receipt["candidates"]):
        candidate_id = cast(str, candidate["candidate_id"])
        transformation = cast(Mapping[str, object], candidate["transformation"])
        evaluation = cast(Mapping[str, object], candidate["evaluation"])
        execution_ref = cast(Mapping[str, object], evaluation["execution"])
        reports = cast(Sequence[Mapping[str, object]], evaluation["reports"])
        replay_ref = cast(Mapping[str, object], evaluation["replay"])
        runtime_ref = cast(Mapping[str, object], evaluation["runtime"])
        refs: list[tuple[str, Mapping[str, object]]] = [
            ("execution receipt", execution_ref),
            ("replay", replay_ref),
            ("runtime reload proof", runtime_ref),
        ]
        for repeat_index, report_run in enumerate(reports):
            refs.extend(
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
        paths = [cast(str, reference["path"]) for _, reference in refs]
        if len(paths) != len(set(paths)):
            raise CleanSelectionEvidenceError(
                "candidate evidence references reuse one sidecar path"
            )
        payloads: dict[str, dict[str, object]] = {}
        for name, reference in refs:
            raw, payload = _read_referenced_json_snapshot(
                reference,
                evidence_root=evidence_root,
                label=f"candidate {candidate_id} {name} sidecar",
            )
            payloads[name] = payload
            retained_bytes[cast(str, reference["path"])] = raw
        execution_payload = payloads["execution receipt"]
        execution_sha = cast(str, execution_ref["sha256"])
        _execution_receipt(
            execution_payload,
            expected_model_key=model_key,
            expected_candidate_id=candidate_id,
            expected_transformation=transformation,
            expected_baseline_identity=baseline,
            expected_selection_config=config,
        )
        measured_losses: list[float] = []
        for repeat_index, report_run in enumerate(reports):
            report_ref = cast(Mapping[str, object], report_run["report"])
            manifest_ref = cast(Mapping[str, object], report_run["runtime_manifest"])
            report_name = f"report repeat {repeat_index}"
            manifest_name = f"runtime manifest repeat {repeat_index}"
            artifact = cast(Mapping[str, str], report_ref["artifact_identity"])
            report_payload = payloads[report_name]
            _assert_report_native_execution_provenance(
                report_payload,
                execution_receipt_sha256=execution_sha,
                selection_config=config,
                original_model_key=model_key,
                candidate_id=candidate_id,
                transformation=transformation,
                baseline_identity=baseline,
                repeat_index=repeat_index,
            )
            measured_losses.append(
                _assert_eligible_report(
                    report_payload,
                    model_key=model_key,
                    candidate_id=candidate_id,
                    transformation=transformation,
                    baseline_identity=baseline,
                    artifact_identity=artifact,
                    selection_config_sha256=config_digest,
                    execution_receipt_sha256=execution_sha,
                    selection_config=config,
                    repeat_index=repeat_index,
                )
            )
            _assert_report_runtime_manifest(
                report_bytes=retained_bytes[cast(str, report_ref["path"])],
                report=report_payload,
                manifest=payloads[manifest_name],
                report_reference=report_ref,
                manifest_reference=manifest_ref,
                execution_receipt_sha256=execution_sha,
                selection_config_sha256=config_digest,
                model_key=model_key,
                candidate_id=candidate_id,
                transformation=transformation,
                baseline_identity=baseline,
                repeat_index=repeat_index,
            )
        if not measured_losses:
            raise CleanSelectionEvidenceError("candidate retains no evaluator reports")
        measured_loss = math.fsum(measured_losses) / len(measured_losses)
        declared_loss = cast(Mapping[str, object], evaluation["metrics"])[
            "quality_loss"
        ]
        if not math.isclose(
            _finite(declared_loss, label="candidate quality_loss"),
            measured_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CleanSelectionEvidenceError(
                "candidate metric does not match retained report"
            )
        _assert_candidate_replay_runtime(
            payloads["replay"],
            payloads["runtime reload proof"],
            transformation=transformation,
            baseline_identity=baseline,
            artifact_identity=cast(
                Mapping[str, str],
                cast(Mapping[str, object], reports[0]["report"])["artifact_identity"],
            ),
        )
    return retained_bytes


def snapshot_selection_bundle_file(
    bundle_path: Path, *, evidence_root: Path | None = None
) -> SelectionBundleSnapshot:
    """Authenticate all bundle evidence from single-read immutable snapshots."""

    bundle_bytes, raw_bundle = strict_json_object_snapshot(
        bundle_path, label="selection bundle"
    )
    bundle = verify_selection_bundle(raw_bundle)
    root = evidence_root if evidence_root is not None else bundle_path.parent
    sidecars: dict[str, bytes] = {}
    for entry in cast(Sequence[Mapping[str, object]], bundle["entries"]):
        for relative, raw in _verify_candidate_artifacts(entry, root).items():
            if relative in sidecars and sidecars[relative] != raw:
                raise CleanSelectionEvidenceError(
                    "selection bundle reuses conflicting candidate evidence"
                )
            sidecars[relative] = raw
    return SelectionBundleSnapshot(
        bundle=bundle,
        bundle_bytes=bundle_bytes,
        sidecar_bytes=sidecars,
    )


def verify_selection_bundle_file(
    bundle_path: Path, *, evidence_root: Path | None = None
) -> dict[str, object]:
    """Verify structure, sidecar references, identities, eligibility, and winner."""

    return snapshot_selection_bundle_file(
        bundle_path, evidence_root=evidence_root
    ).bundle


def selected_entry_for(
    bundle: Mapping[str, object],
    *,
    model_key: str,
    edit_type: str,
    requested_scope: str = "",
) -> dict[str, object]:
    """Return one exact selected v1 entry after structural recomputation."""

    verified = verify_selection_bundle(bundle)
    requested = (
        _scope(requested_scope, label="requested_scope") if requested_scope else ""
    )
    matches: list[dict[str, object]] = []
    for entry in cast(Sequence[dict[str, object]], verified["entries"]):
        selected = cast(Mapping[str, object], entry["selected_entry"])
        if (
            entry["original_model_key"] == model_key
            and selected["edit_type"] == edit_type
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise CleanSelectionEvidenceError(
            "selection bundle has no unique matching selected entry"
        )
    result = matches[0]
    selected = cast(Mapping[str, object], result["selected_entry"])
    if requested and selected["scope"] != requested:
        raise CleanSelectionEvidenceError(
            "requested scope does not match the selected candidate"
        )
    return result


def referenced_candidate_paths(bundle: Mapping[str, object]) -> list[str]:
    """Return sorted, unique safe sidecar paths for bounded staging."""

    verified = verify_selection_bundle(bundle)
    result: set[str] = set()
    for entry in cast(Sequence[Mapping[str, object]], verified["entries"]):
        selected = cast(Mapping[str, object], entry["selected_entry"])
        receipt = cast(Mapping[str, object], selected["selection_receipt"])
        for candidate in cast(Sequence[Mapping[str, object]], receipt["candidates"]):
            evaluation = cast(Mapping[str, object], candidate["evaluation"])
            for name in ("execution", "replay", "runtime"):
                reference = cast(Mapping[str, object], evaluation[name])
                result.add(cast(str, reference["path"]))
            for report_run in cast(
                Sequence[Mapping[str, object]], evaluation["reports"]
            ):
                for name in ("report", "runtime_manifest"):
                    reference = cast(Mapping[str, object], report_run[name])
                    result.add(cast(str, reference["path"]))
    return sorted(result)
