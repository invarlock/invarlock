"""Signed baseline-material bindings for evidence-pack report verification.

Strict report verification needs the independently produced baseline report, not
only the baseline values copied into a subject report.  This module validates
the manifest declaration which binds those baseline bytes to the subject report
paths that consume them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from invarlock import evidence_pack_integrity as integrity
from invarlock.core.assurance_contract import resolve_report_assurance_mode

BASELINES_MANIFEST_FIELD = "verification_baselines"
BASELINES_ROOT = "baselines"
BASELINE_REPORT_FILENAME = "evaluation.report.json"


@dataclass(frozen=True)
class BaselineMaterialVerification:
    """Validated subject-to-baseline paths inside one evidence pack."""

    baseline_by_report: dict[Path, Path]
    errors: tuple[str, ...]
    required: bool


def _normalize_relative_path(
    value: Any,
    *,
    root: str,
    filename: str,
) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or "\\" in raw or raw.startswith("/"):
        return None
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    path = PurePosixPath(*parts)
    if path.is_absolute() or len(path.parts) < 3:
        return None
    if path.parts[0] != root or path.name != filename:
        return None
    return path.as_posix()


def _path_symlink_error(
    pack_dir: Path, relative_path: str, *, label: str
) -> str | None:
    candidate = pack_dir
    for part in PurePosixPath(relative_path).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return f"{label} must not be a symlink or traverse one: {relative_path}"
    return None


def _canonical_report_paths(pack_dir: Path) -> set[str]:
    reports_root = pack_dir / "reports"
    if not reports_root.is_dir() or reports_root.is_symlink():
        return set()
    return {
        path.relative_to(pack_dir).as_posix()
        for path in reports_root.rglob(BASELINE_REPORT_FILENAME)
        if path.is_file() and not path.is_symlink()
    }


def _strict_report_paths(pack_dir: Path, report_paths: set[str]) -> set[str]:
    strict_paths: set[str] = set()
    for relative_path in report_paths:
        try:
            payload = integrity._load_json(pack_dir / relative_path)
        except integrity._json_load_error_types():
            continue
        if (
            isinstance(payload, dict)
            and resolve_report_assurance_mode(payload) == "strict"
        ):
            strict_paths.add(relative_path)
    return strict_paths


def _manifest_requires_baselines(
    manifest: dict[str, Any],
    *,
    report_assurance: str,
    strict_report_paths: set[str],
) -> bool:
    verification = manifest.get("verification")
    declared_assurance = (
        verification.get("report_assurance") if isinstance(verification, dict) else None
    )
    return bool(
        report_assurance == "strict"
        or declared_assurance == "strict"
        or (report_assurance == "report" and strict_report_paths)
    )


def _checksum_entries_by_path(pack_dir: Path) -> dict[str, list[str]]:
    entries, _errors = integrity.parse_checksums(pack_dir)
    by_path: dict[str, list[str]] = {}
    for digest, raw_path in entries:
        path = integrity.canonicalize_checksum_path(raw_path)
        by_path.setdefault(path, []).append(digest)
    return by_path


def _baseline_tree_files_and_symlink_errors(
    pack_dir: Path,
) -> tuple[set[str], list[str]]:
    root = pack_dir / BASELINES_ROOT
    if not root.exists() and not root.is_symlink():
        return set(), []
    if root.is_symlink():
        return set(), [f"{BASELINES_ROOT}/ must not be a symlink."]
    files: set[str] = set()
    errors: list[str] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(pack_dir).as_posix()
        if path.is_symlink():
            errors.append(
                f"Baseline material tree must not contain symlinks: {relative_path}"
            )
        elif path.is_file():
            files.add(relative_path)
    return files, errors


def _metric_kind_and_final(payload: dict[str, Any]) -> tuple[str, float] | None:
    metric: Any = payload.get("primary_metric")
    if not isinstance(metric, dict):
        metrics = payload.get("metrics")
        metric = metrics.get("primary_metric") if isinstance(metrics, dict) else None
    if not isinstance(metric, dict):
        return None
    kind = metric.get("kind")
    final = metric.get("final")
    if (
        not isinstance(kind, str)
        or not kind.strip()
        or isinstance(final, bool)
        or not isinstance(final, int | float)
        or not math.isfinite(float(final))
    ):
        return None
    return kind.strip().lower(), float(final)


def _subject_baseline_metric(payload: dict[str, Any]) -> tuple[str, float] | None:
    baseline_ref = payload.get("baseline_ref")
    if not isinstance(baseline_ref, dict):
        return None
    return _metric_kind_and_final(baseline_ref)


def _semantic_binding_errors(
    *,
    report_path: Path,
    baseline_path: Path,
    relative_report_path: str,
) -> list[str]:
    errors: list[str] = []
    try:
        report_payload = integrity._load_json(report_path)
    except integrity._json_load_error_types() as exc:
        return [
            f"Baseline-bound subject report is not valid JSON ({relative_report_path}): {exc}"
        ]
    try:
        baseline_payload = integrity._load_json(baseline_path)
    except integrity._json_load_error_types() as exc:
        return [
            f"Verification baseline is not valid JSON ({baseline_path.name}): {exc}"
        ]
    if not isinstance(report_payload, dict):
        return [
            f"Baseline-bound subject report must be a JSON object: {relative_report_path}"
        ]
    if not isinstance(baseline_payload, dict):
        return [f"Verification baseline must be a JSON object: {baseline_path}"]

    baseline_metric = _metric_kind_and_final(baseline_payload)
    subject_reference = _subject_baseline_metric(report_payload)
    if baseline_metric is None:
        errors.append(
            f"Verification baseline lacks a finite primary metric: {baseline_path}"
        )
    if subject_reference is None:
        errors.append(
            "Baseline-bound subject report lacks baseline_ref.primary_metric: "
            f"{relative_report_path}"
        )
    if baseline_metric is not None and subject_reference is not None:
        baseline_kind, baseline_final = baseline_metric
        subject_kind, subject_final = subject_reference
        if baseline_kind != subject_kind:
            errors.append(
                "Verification baseline metric kind does not match subject baseline_ref "
                f"for {relative_report_path}: {baseline_kind!r} != {subject_kind!r}."
            )
        if not math.isclose(
            baseline_final,
            subject_final,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                "Verification baseline final value does not match subject baseline_ref "
                f"for {relative_report_path}: {baseline_final:.12f} != "
                f"{subject_final:.12f}."
            )

    # Reuse the verifier's strict baseline contract so skip-verify never turns a
    # malformed or schedule-mismatched raw baseline into trusted pack metadata.
    from invarlock.reporting.verify_baseline import (
        append_strict_baseline_contract_errors,
    )
    from invarlock.reporting.verify_bootstrap import (
        append_strict_ppl_bootstrap_replay_errors,
    )
    from invarlock.reporting.verify_strict_schedule import (
        _append_strict_supplied_baseline_binding_errors,
    )

    strict_errors: list[str] = []
    append_strict_baseline_contract_errors(
        strict_errors,
        report=report_payload,
        baseline_payload=baseline_payload,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    _append_strict_supplied_baseline_binding_errors(
        strict_errors,
        cert_obj=report_payload,
        baseline_payload=baseline_payload,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    append_strict_ppl_bootstrap_replay_errors(
        strict_errors,
        report=report_payload,
        baseline_payload=baseline_payload,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    errors.extend(
        f"Verification baseline does not match {relative_report_path}: {error}"
        for error in strict_errors
    )
    return errors


def verify_baseline_materials(
    pack_dir: Path,
    *,
    report_assurance: str,
) -> BaselineMaterialVerification:
    """Validate and resolve signed baseline declarations without trusting paths."""

    errors: list[str] = []
    try:
        manifest = integrity._load_json(pack_dir / "manifest.json")
    except integrity._json_load_error_types() as exc:
        return BaselineMaterialVerification(
            baseline_by_report={},
            errors=(f"manifest is not valid JSON: {exc}",),
            required=False,
        )
    if not isinstance(manifest, dict):
        return BaselineMaterialVerification(
            baseline_by_report={},
            errors=("manifest must decode to a JSON object",),
            required=False,
        )

    canonical_reports = _canonical_report_paths(pack_dir)
    strict_reports = _strict_report_paths(pack_dir, canonical_reports)
    required = _manifest_requires_baselines(
        manifest,
        report_assurance=report_assurance,
        strict_report_paths=strict_reports,
    )
    declarations = manifest.get(BASELINES_MANIFEST_FIELD)
    actual_baseline_files, tree_errors = _baseline_tree_files_and_symlink_errors(
        pack_dir
    )
    errors.extend(tree_errors)

    if declarations is None:
        if required:
            errors.append(
                "Strict evidence-pack verification requires signed verification_baselines."
            )
        if actual_baseline_files:
            errors.append(
                "Pack contains undeclared baseline material: "
                + ", ".join(sorted(actual_baseline_files))
                + "."
            )
        return BaselineMaterialVerification(
            baseline_by_report={}, errors=tuple(errors), required=required
        )
    if not isinstance(declarations, list) or not declarations:
        errors.append("manifest verification_baselines must be a non-empty list.")
        return BaselineMaterialVerification(
            baseline_by_report={}, errors=tuple(errors), required=required
        )

    checksum_entries = _checksum_entries_by_path(pack_dir)
    declared_paths: set[str] = set()
    mapped_report_paths: set[str] = set()
    baseline_by_report: dict[Path, Path] = {}
    seen_names: set[str] = set()

    for index, declaration in enumerate(declarations):
        label = f"manifest verification_baselines[{index}]"
        if not isinstance(declaration, dict):
            errors.append(f"{label} must be an object.")
            continue
        name = declaration.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label}.name must be a non-empty string.")
        elif name in seen_names:
            errors.append(f"{label}.name is duplicated: {name!r}.")
        else:
            seen_names.add(name)

        relative_baseline_path = _normalize_relative_path(
            declaration.get("path"),
            root=BASELINES_ROOT,
            filename=BASELINE_REPORT_FILENAME,
        )
        if relative_baseline_path is None:
            errors.append(
                f"{label}.path must be a canonical {BASELINES_ROOT}/.../"
                f"{BASELINE_REPORT_FILENAME} path."
            )
            continue
        if relative_baseline_path in declared_paths:
            errors.append(
                f"{label}.path is declared more than once: {relative_baseline_path}."
            )
            continue
        declared_paths.add(relative_baseline_path)

        symlink_error = _path_symlink_error(
            pack_dir, relative_baseline_path, label=label
        )
        if symlink_error is not None:
            errors.append(symlink_error)
            continue
        baseline_path = pack_dir / relative_baseline_path
        if not baseline_path.is_file():
            errors.append(f"{label}.path is missing: {relative_baseline_path}.")
            continue

        digest = declaration.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            errors.append(f"{label}.digest must be a sha256:... string.")
            continue
        expected_hex = digest.removeprefix("sha256:")
        if len(expected_hex) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hex
        ):
            errors.append(f"{label}.digest must be a lowercase sha256:... string.")
            continue
        actual_hex = integrity._sha256_path_hex(baseline_path)
        if actual_hex != expected_hex:
            errors.append(
                f"{label}.digest mismatch for {relative_baseline_path} "
                f"(expected sha256:{expected_hex}, got sha256:{actual_hex})."
            )
        bound_checksums = checksum_entries.get(relative_baseline_path, [])
        if len(bound_checksums) != 1:
            errors.append(
                f"{label}.path must have exactly one checksums.sha256 entry: "
                f"{relative_baseline_path}."
            )
        elif bound_checksums[0] != expected_hex:
            errors.append(
                f"{label}.digest is not bound by checksums.sha256 for "
                f"{relative_baseline_path}."
            )

        try:
            baseline_payload = integrity._load_json(baseline_path)
        except integrity._json_load_error_types() as exc:
            errors.append(
                f"Verification baseline is not valid JSON ({relative_baseline_path}): {exc}"
            )
            baseline_payload = None
        if baseline_payload is not None and not isinstance(baseline_payload, dict):
            errors.append(
                f"Verification baseline must decode to a JSON object: "
                f"{relative_baseline_path}."
            )

        report_paths = declaration.get("report_paths")
        if not isinstance(report_paths, list) or not report_paths:
            errors.append(f"{label}.report_paths must be a non-empty list.")
            continue
        for report_index, raw_report_path in enumerate(report_paths):
            relative_report_path = _normalize_relative_path(
                raw_report_path,
                root="reports",
                filename=BASELINE_REPORT_FILENAME,
            )
            report_label = f"{label}.report_paths[{report_index}]"
            if relative_report_path is None:
                errors.append(f"{report_label} is not a canonical report path.")
                continue
            if relative_report_path not in canonical_reports:
                errors.append(
                    f"{report_label} is not present as a canonical report: "
                    f"{relative_report_path}."
                )
                continue
            if relative_report_path in mapped_report_paths:
                errors.append(
                    f"Canonical report has more than one verification baseline: "
                    f"{relative_report_path}."
                )
                continue
            mapped_report_paths.add(relative_report_path)
            report_path = pack_dir / relative_report_path
            baseline_by_report[report_path.resolve()] = baseline_path.resolve()
            errors.extend(
                _semantic_binding_errors(
                    report_path=report_path,
                    baseline_path=baseline_path,
                    relative_report_path=relative_report_path,
                )
            )

    extra_baselines = sorted(actual_baseline_files - declared_paths)
    if extra_baselines:
        errors.append(
            "Pack contains undeclared baseline material: "
            + ", ".join(extra_baselines)
            + "."
        )
    verification = manifest.get("verification")
    manifest_strict = (
        isinstance(verification, dict)
        and verification.get("report_assurance") == "strict"
    )
    required_report_paths = (
        canonical_reports
        if report_assurance == "strict" or manifest_strict
        else strict_reports
        if report_assurance == "report"
        else set()
    )
    missing_required_reports = sorted(required_report_paths - mapped_report_paths)
    if missing_required_reports:
        errors.append(
            "Signed verification baselines do not cover required subject reports: "
            + ", ".join(missing_required_reports)
            + "."
        )

    return BaselineMaterialVerification(
        baseline_by_report=baseline_by_report,
        errors=tuple(errors),
        required=required,
    )


def baseline_manifest_entry(
    *,
    name: str,
    relative_path: str,
    baseline_path: Path,
    report_paths: list[str],
) -> dict[str, Any]:
    """Build one signed manifest entry for already-copied baseline bytes."""

    return {
        "name": name,
        "path": relative_path,
        "digest": integrity._sha256_file(baseline_path),
        "report_paths": list(report_paths),
    }


def discover_staged_baseline_materials(
    pack_dir: Path,
    *,
    report_assurance: str,
) -> BaselineMaterialVerification:
    """Match run-pack baseline files to reports before a manifest is written."""

    canonical_reports = _canonical_report_paths(pack_dir)
    strict_reports = _strict_report_paths(pack_dir, canonical_reports)
    required_reports = (
        canonical_reports
        if report_assurance == "strict"
        else strict_reports
        if report_assurance == "report"
        else set()
    )
    baseline_files, errors = _baseline_tree_files_and_symlink_errors(pack_dir)
    candidates = sorted(baseline_files)
    baseline_by_report: dict[Path, Path] = {}

    for relative_report_path in sorted(canonical_reports):
        report_parts = PurePosixPath(relative_report_path).parts
        model_component = report_parts[1] if len(report_parts) > 2 else None
        matches: dict[str, Path] = {}
        for relative_baseline_path in candidates:
            baseline_parts = PurePosixPath(relative_baseline_path).parts
            if (
                model_component is not None
                and len(baseline_parts) > 2
                and baseline_parts[1] != model_component
            ):
                continue
            report_path = pack_dir / relative_report_path
            baseline_path = pack_dir / relative_baseline_path
            if _semantic_binding_errors(
                report_path=report_path,
                baseline_path=baseline_path,
                relative_report_path=relative_report_path,
            ):
                continue
            digest = integrity._sha256_path_hex(baseline_path)
            matches.setdefault(digest, baseline_path)
        if len(matches) == 1:
            baseline_by_report[(pack_dir / relative_report_path).resolve()] = next(
                iter(matches.values())
            ).resolve()
        elif relative_report_path in required_reports:
            if matches:
                errors.append(
                    "Required subject report matches multiple distinct staged baselines: "
                    f"{relative_report_path}."
                )
            else:
                errors.append(
                    "Required subject report has no matching staged raw baseline: "
                    f"{relative_report_path}."
                )

    return BaselineMaterialVerification(
        baseline_by_report=baseline_by_report,
        errors=tuple(errors),
        required=bool(required_reports),
    )


def verify_build_baseline(
    *,
    baseline_path: Path,
    report_paths: list[Path],
) -> list[str]:
    """Validate an external build input before any output directory is created."""

    errors: list[str] = []
    baseline_resolved = baseline_path.resolve(strict=False)
    try:
        baseline_payload = integrity._load_json(baseline_path)
    except integrity._json_load_error_types():
        baseline_payload = None
    for index, report_path in enumerate(report_paths, start=1):
        if report_path.resolve(strict=False) == baseline_resolved:
            errors.append(
                "Verification baseline must be a file distinct from every subject "
                f"report (report input {index})."
            )
            continue
        try:
            report_payload = integrity._load_json(report_path)
        except integrity._json_load_error_types():
            report_payload = None
        if not (
            isinstance(report_payload, dict)
            and isinstance(baseline_payload, dict)
            and _subject_baseline_metric(report_payload) is not None
            and _metric_kind_and_final(baseline_payload) is not None
        ):
            # The normal report verifier owns malformed-report diagnostics.  This
            # preflight adds independent-baseline semantics once both metric
            # identities are available.
            continue
        errors.extend(
            _semantic_binding_errors(
                report_path=report_path,
                baseline_path=baseline_path,
                relative_report_path=f"report input {index}",
            )
        )
    return errors


def baseline_manifest_entries_from_mapping(
    pack_dir: Path,
    baseline_by_report: dict[Path, Path],
) -> list[dict[str, Any]]:
    """Collapse a validated mapping into deterministic signed declarations."""

    reports_by_baseline: dict[Path, list[str]] = {}
    for report_path, baseline_path in baseline_by_report.items():
        reports_by_baseline.setdefault(baseline_path.resolve(), []).append(
            report_path.resolve().relative_to(pack_dir.resolve()).as_posix()
        )
    entries: list[dict[str, Any]] = []
    for index, (baseline_path, report_paths) in enumerate(
        sorted(reports_by_baseline.items(), key=lambda item: str(item[0])),
        start=1,
    ):
        entries.append(
            baseline_manifest_entry(
                name=f"baseline-{index:03d}",
                relative_path=baseline_path.relative_to(pack_dir.resolve()).as_posix(),
                baseline_path=baseline_path,
                report_paths=sorted(report_paths),
            )
        )
    return entries


__all__ = [
    "BASELINE_REPORT_FILENAME",
    "BASELINES_MANIFEST_FIELD",
    "BASELINES_ROOT",
    "BaselineMaterialVerification",
    "baseline_manifest_entries_from_mapping",
    "baseline_manifest_entry",
    "discover_staged_baseline_materials",
    "verify_build_baseline",
    "verify_baseline_materials",
]
