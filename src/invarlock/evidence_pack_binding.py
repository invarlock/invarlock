"""Semantic binding between evidence-pack verdicts and canonical reports.

This module owns the path-safety, digest, and identifier checks that prevent a
``final_verdict.json`` from making claims about a report other than the one(s)
actually shipped in an evidence pack.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.evidence_pack_contracts.probes import (
    PROBE_FILENAMES,
    ProbeValidationError,
    load_probe_file,
    validate_probe_binding,
)
from invarlock.evidence_pack_snapshot import PackSnapshot

_load_json = evidence_pack_integrity_mod._load_json
_json_load_error_types = evidence_pack_integrity_mod._json_load_error_types


def _normalize_binding_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return digest


def _normalize_verdict_report_path(value: Any) -> str | None:
    """Normalize supported source/pack report paths to a pack-relative path."""
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or "\\" in raw:
        return None
    raw_parts = raw.split("/")
    if any(part in {"", ".."} for part in raw_parts):
        return None
    while raw_parts and raw_parts[0] == ".":
        raw_parts.pop(0)
    if not raw_parts:
        return None
    path = PurePosixPath(*raw_parts)
    if path.is_absolute():
        return None
    parts = path.parts
    if parts[0] == "reports":
        normalized = path
    elif len(parts) >= 3 and parts[1] == "reports":
        # Historical verdict-generator records refer to the run tree as
        # <model>/reports/<scenario>/evaluation.report.json; portable packs use
        # reports/<model>/<scenario>/evaluation.report.json.
        normalized = PurePosixPath("reports", parts[0], *parts[2:])
    else:
        return None
    if normalized.name != "evaluation.report.json":
        return None
    return normalized.as_posix()


def _report_run_id(payload: dict[str, Any]) -> str | None:
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    meta = payload.get("meta")
    if isinstance(meta, dict):
        run_id = meta.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
    return None


def _report_id(payload: dict[str, Any]) -> str | None:
    report_id = payload.get("report_id")
    if isinstance(report_id, str) and report_id.strip():
        return report_id.strip()
    meta = payload.get("meta")
    if isinstance(meta, dict):
        report_id = meta.get("report_id")
        if isinstance(report_id, str) and report_id.strip():
            return report_id.strip()
    return _report_run_id(payload)


def _canonical_report_binding(
    *, path: Path, relative_path: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, [f"Canonical report is not valid JSON ({relative_path}): {exc}"]
    if not isinstance(payload, dict):
        return None, [f"Canonical report must be a JSON object: {relative_path}"]
    probe_bindings: list[dict[str, str]] = []
    for filename in PROBE_FILENAMES:
        probe_path = path.parent / filename
        if not probe_path.exists():
            continue
        relative_probe = (PurePosixPath(relative_path).parent / filename).as_posix()
        try:
            probe_payload = load_probe_file(probe_path)
            validate_probe_binding(
                probe_payload.get("binding"),
                payload,
                "sha256:" + evidence_pack_integrity_mod._sha256_path_hex(path),
            )
        except ProbeValidationError as exc:
            errors.append(f"Verdict probe is invalid ({relative_probe}): {exc}")
            continue
        probe_bindings.append(
            {
                "path": relative_probe,
                "sha256": evidence_pack_integrity_mod._sha256_path_hex(probe_path),
            }
        )
    return (
        {
            "path": relative_path,
            "report_sha256": evidence_pack_integrity_mod._sha256_path_hex(path),
            "run_id": _report_run_id(payload),
            "report_id": _report_id(payload),
            "probe_bindings": probe_bindings,
        },
        errors,
    )


def _validate_binding_item(
    item: dict[str, Any],
    *,
    label: str,
    reports_by_path: dict[str, dict[str, Any]],
    single_report_path: str | None = None,
    require_path: bool,
    require_digest: bool,
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    raw_path = item.get("path")
    raw_report_path = item.get("report_path")
    if (
        raw_path is not None
        and raw_report_path is not None
        and raw_path != raw_report_path
    ):
        errors.append(f"{label} path and report_path disagree.")
        return None, errors
    path_value = raw_path if raw_path is not None else raw_report_path
    normalized_path = _normalize_verdict_report_path(path_value)
    if path_value is None and single_report_path is not None:
        normalized_path = single_report_path
    elif path_value is None and require_path:
        errors.append(f"{label} requires a canonical report path.")
        return None, errors
    elif path_value is not None and normalized_path is None:
        errors.append(f"{label} contains an invalid report path: {path_value!r}.")
        return None, errors

    if normalized_path is None:
        errors.append(f"{label} cannot be associated with a canonical report.")
        return None, errors
    report = reports_by_path.get(normalized_path)
    if report is None:
        errors.append(
            f"{label} references a report not present in the pack: {normalized_path}."
        )
        return normalized_path, errors

    raw_digest = item.get("report_sha256")
    if raw_digest is None:
        if require_digest:
            errors.append(f"{label} requires report_sha256.")
    else:
        digest = _normalize_binding_digest(raw_digest)
        if digest is None:
            errors.append(f"{label} report_sha256 must be a SHA-256 digest.")
        elif digest != report["report_sha256"]:
            errors.append(
                f"{label} report_sha256 does not match {normalized_path} "
                f"(recorded {digest}, actual {report['report_sha256']})."
            )

    for field in ("run_id", "report_id"):
        if field not in item:
            continue
        recorded_id = item.get(field)
        if not isinstance(recorded_id, str) or not recorded_id.strip():
            errors.append(f"{label} {field} must be a non-empty string.")
            continue
        actual_id = report.get(field)
        if actual_id is None:
            errors.append(
                f"{label} {field} cannot be authenticated because "
                f"{normalized_path} does not contain {field}."
            )
        elif recorded_id.strip() != actual_id:
            errors.append(
                f"{label} {field} does not match {normalized_path} "
                f"(recorded {recorded_id.strip()!r}, actual {actual_id!r})."
            )
    claimed_probes = item.get("probe_bindings", [])
    if (
        label.startswith("Final verdict report_bindings[") or "probe_bindings" in item
    ) and claimed_probes != report.get("probe_bindings", []):
        errors.append(f"{label} probe_bindings do not match packed probe evidence.")
    return normalized_path, errors


def _verify_final_verdict_payload_report_binding(
    verdict: Any,
    reports: list[dict[str, Any]],
    *,
    require_binding: bool,
) -> list[str]:
    if not isinstance(verdict, dict):
        return ["Final verdict must be a JSON object."]
    errors: list[str] = []
    reports_by_path = {str(report["path"]): report for report in reports}
    if len(reports_by_path) != len(reports):
        errors.append("Canonical report paths are not unique.")
        return errors
    report_paths = set(reports_by_path)
    if not reports:
        return ["Final verdict exists but the pack contains no canonical reports."]

    binding_value = verdict.get("report_bindings")
    has_bindings = binding_value is not None
    if has_bindings and not isinstance(binding_value, list):
        errors.append("Final verdict report_bindings must be a list.")
    elif isinstance(binding_value, list):
        if not binding_value:
            errors.append("Final verdict report_bindings must not be empty.")
        bound_paths: list[str] = []
        for index, binding in enumerate(binding_value):
            label = f"Final verdict report_bindings[{index}]"
            if not isinstance(binding, dict):
                errors.append(f"{label} must be a JSON object.")
                continue
            bound_path, binding_errors = _validate_binding_item(
                binding,
                label=label,
                reports_by_path=reports_by_path,
                require_path=True,
                require_digest=True,
            )
            errors.extend(binding_errors)
            if bound_path is not None:
                if bound_path in bound_paths:
                    errors.append(
                        f"Final verdict report_bindings contains duplicate path: {bound_path}."
                    )
                bound_paths.append(bound_path)
        missing = sorted(report_paths - set(bound_paths))
        extra = sorted(set(bound_paths) - report_paths)
        if missing:
            errors.append(
                "Final verdict report_bindings does not cover canonical reports: "
                + ", ".join(missing)
                + "."
            )
        if extra:
            errors.append(
                "Final verdict report_bindings contains non-canonical reports: "
                + ", ".join(extra)
                + "."
            )

    single_fields = ("report_sha256", "report_path", "run_id", "report_id")
    present_single_fields = [field for field in single_fields if field in verdict]
    if len(reports) == 1:
        if require_binding and not has_bindings and "report_sha256" not in verdict:
            errors.append(
                "Final verdict for one report requires report_sha256 or report_bindings."
            )
        if present_single_fields:
            _, top_level_errors = _validate_binding_item(
                verdict,
                label="Final verdict",
                reports_by_path=reports_by_path,
                single_report_path=next(iter(report_paths)),
                require_path=False,
                require_digest=require_binding and not has_bindings,
            )
            errors.extend(top_level_errors)
    elif present_single_fields:
        errors.append(
            "Final verdict single-report fields are ambiguous for multiple reports: "
            + ", ".join(present_single_fields)
            + "."
        )
    if len(reports) > 1 and require_binding and not has_bindings:
        errors.append(
            "Final verdict for multiple reports requires exact report_bindings coverage."
        )

    records_value = verdict.get("records")
    if records_value is not None:
        if not isinstance(records_value, list):
            errors.append("Final verdict records must be a list.")
        else:
            record_paths: set[str] = set()
            for index, record in enumerate(records_value):
                label = f"Final verdict records[{index}]"
                if not isinstance(record, dict):
                    errors.append(f"{label} must be a JSON object.")
                    continue
                record_path, record_errors = _validate_binding_item(
                    record,
                    label=label,
                    reports_by_path=reports_by_path,
                    require_path=True,
                    require_digest=require_binding,
                )
                errors.extend(record_errors)
                if record_path is not None:
                    if record_path in record_paths:
                        errors.append(
                            f"Final verdict records contains duplicate report path: {record_path}."
                        )
                    record_paths.add(record_path)
    return errors


def _pack_report_binding_declaration(pack_dir: Path) -> tuple[bool, str | None]:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        return False, None
    try:
        manifest = _load_json(manifest_path)
    except _json_load_error_types() as exc:
        return False, f"Manifest report-binding declaration is not valid JSON: {exc}"
    if not isinstance(manifest, dict):
        return False, None
    verification = manifest.get("verification")
    if not isinstance(verification, dict):
        return False, None
    assurance = verification.get("report_assurance")
    return (
        isinstance(assurance, str) and assurance.strip().lower() == "strict",
        None,
    )


def _pack_declares_strict_report_binding(pack_dir: Path) -> bool:
    declaration, _ = _pack_report_binding_declaration(pack_dir)
    return declaration


def _binding_file_safety_errors(pack_dir: Path, path: Path, *, label: str) -> list[str]:
    try:
        relative = path.relative_to(pack_dir)
    except ValueError:
        return [f"{label} path is outside the pack: {path}."]

    current = pack_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return [f"{label} path must not contain symlinks: {relative.as_posix()}."]
    try:
        resolved_root = pack_dir.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError):
        return [
            f"{label} path must resolve to a regular file within the pack: "
            f"{relative.as_posix()}."
        ]
    if not resolved_path.is_file():
        return [f"{label} path is not a regular file: {relative.as_posix()}."]
    return []


def _discover_binding_files(
    pack_dir: Path,
    *,
    subtree: str,
    filename: str,
    label: str,
) -> tuple[list[Path], list[str]]:
    root = pack_dir / subtree
    if not os.path.lexists(root):
        return [], []
    if pack_dir.is_symlink() or root.is_symlink():
        return [], [f"{label} search tree must not be a symlink: {subtree}."]
    if not root.is_dir():
        return [], [f"{label} search tree must be a directory: {subtree}."]

    matches: list[Path] = []
    errors: list[str] = []
    for current_dir, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_dir)
        safe_dirnames: list[str] = []
        for dirname in dirnames:
            candidate = current_path / dirname
            if candidate.is_symlink():
                relative = candidate.relative_to(pack_dir).as_posix()
                errors.append(
                    f"{label} search tree must not contain symlinks: {relative}."
                )
            else:
                safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames
        if filename not in filenames:
            continue
        candidate = current_path / filename
        file_errors = _binding_file_safety_errors(
            pack_dir,
            candidate,
            label=label,
        )
        errors.extend(file_errors)
        if not file_errors:
            matches.append(candidate)
    return sorted(matches), errors


def _verify_final_verdict_report_binding_snapshot(
    pack_dir: Path, *, require_binding: bool = False
) -> list[str]:
    """Verify that final-verdict claims bind exactly to canonical packed reports."""
    declared_strict, declaration_error = _pack_report_binding_declaration(pack_dir)
    if declaration_error is not None:
        return [declaration_error]
    require_binding = require_binding or declared_strict
    verdict_paths, verdict_path_errors = _discover_binding_files(
        pack_dir,
        subtree="results",
        filename="final_verdict.json",
        label="Final verdict",
    )
    report_paths, report_path_errors = _discover_binding_files(
        pack_dir,
        subtree="reports",
        filename="evaluation.report.json",
        label="Canonical report",
    )
    path_errors = [*verdict_path_errors, *report_path_errors]
    if path_errors:
        return path_errors
    verdict_candidates = set(verdict_paths)
    if not verdict_candidates and not report_paths:
        return []
    if not verdict_candidates:
        return ["Canonical reports exist but final_verdict.json is missing."]
    if len(verdict_candidates) != 1:
        candidates = ", ".join(
            path.relative_to(pack_dir).as_posix() for path in sorted(verdict_candidates)
        )
        return [f"Pack contains multiple final verdicts: {candidates}."]
    verdict_path = next(iter(verdict_candidates))
    try:
        verdict = _load_json(verdict_path)
    except _json_load_error_types() as exc:
        return [f"Final verdict is not valid JSON: {exc}"]

    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for report_path in report_paths:
        relative_path = report_path.relative_to(pack_dir).as_posix()
        report, report_errors = _canonical_report_binding(
            path=report_path,
            relative_path=relative_path,
        )
        errors.extend(report_errors)
        if report is not None:
            reports.append(report)
    if errors:
        return errors
    return _verify_final_verdict_payload_report_binding(
        verdict,
        reports,
        require_binding=require_binding,
    )


def verify_final_verdict_report_binding(
    pack_dir: Path, *, require_binding: bool = False
) -> list[str]:
    """Verify report bindings against one immutable pack snapshot.

    This function is also a public direct helper, so it cannot assume its
    caller already established the package verifier's snapshot boundary.
    Hashes, parsed reports, the verdict, and the manifest therefore all come
    from one captured tree.  Source mutation during or after verification is
    reported as an integrity error instead of allowing mixed generations.
    """

    if not pack_dir.exists():
        return []
    # Preserve the focused path-safety diagnostics before capture.  No report
    # or verdict semantics are consumed during this preflight.
    _, verdict_path_errors = _discover_binding_files(
        pack_dir,
        subtree="results",
        filename="final_verdict.json",
        label="Final verdict",
    )
    _, report_path_errors = _discover_binding_files(
        pack_dir,
        subtree="reports",
        filename="evaluation.report.json",
        label="Canonical report",
    )
    if verdict_path_errors or report_path_errors:
        return [*verdict_path_errors, *report_path_errors]

    snapshot, capture_errors = PackSnapshot.capture(
        pack_dir,
        validate_structural_json=False,
    )
    if snapshot is None:
        return capture_errors
    with snapshot.files.materialized() as snapshot_root:
        errors = _verify_final_verdict_report_binding_snapshot(
            snapshot_root,
            require_binding=require_binding,
        )
        materialized_errors = snapshot.files.materialized_stability_errors(
            snapshot_root
        )
    return [*errors, *materialized_errors, *snapshot.stability_errors()]
