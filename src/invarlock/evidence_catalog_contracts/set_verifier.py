"""Exact catalog evidence-pack set verification and receipt generation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from invarlock.evidence_catalog import (
    EVIDENCE_PACK_SET_RECEIPT_FORMAT,
    EvidenceCatalog,
    EvidenceCatalogError,
    load_evidence_catalog,
)
from invarlock.evidence_catalog_contracts.primitives import (
    entry_digest,
)
from invarlock.evidence_catalog_contracts.primitives import (
    sha256_bytes as _sha256_bytes,
)
from invarlock.evidence_pack import verify_evidence_pack
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_snapshot import PackSnapshot
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus

_CATALOG_PATH = "metadata/catalog.json"
_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        try:
            existing = read_regular_file_bytes(path, label="evidence-pack set receipt")
        except StrictJsonError as exc:
            raise EvidenceCatalogError(f"receipt cannot be reused: {exc}") from exc
        if existing == serialized.encode("utf-8"):
            return
        raise EvidenceCatalogError("receipt already exists with different content")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _preflight_pack_set_paths(*, pack_dirs: Sequence[Path], receipt_path: Path) -> None:
    """Reject ambiguous pack topology and any receipt that could mutate a pack."""

    resolved_packs = [path.resolve(strict=False) for path in pack_dirs]
    if len(set(resolved_packs)) != len(resolved_packs):
        raise EvidenceCatalogError("pack directories must be unique")
    for index, pack_path in enumerate(resolved_packs):
        for other_path in resolved_packs[index + 1 :]:
            if pack_path in other_path.parents or other_path in pack_path.parents:
                raise EvidenceCatalogError("pack directories must not be nested")
    resolved_receipt = receipt_path.resolve(strict=False)
    if any(
        resolved_receipt == pack_path or pack_path in resolved_receipt.parents
        for pack_path in resolved_packs
    ):
        raise EvidenceCatalogError("receipt must be outside every pack directory")


def _pack_provenance(
    pack_dir: Path,
) -> tuple[dict[str, object], dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        manifest_bytes, manifest = read_json_object_snapshot(
            pack_dir / "manifest.json", label="pack manifest"
        )
    except StrictJsonError as exc:
        return (
            {
                "pack_digest": None,
                "manifest_digest": None,
                "report_digests": [],
                "source_commit": None,
                "source_bundle_digest": None,
                "runtime_image_digest": None,
            },
            None,
            [f"pack manifest cannot be loaded: {exc}"],
        )
    try:
        checksums = read_regular_file_bytes(
            pack_dir / "checksums.sha256", label="pack checksums"
        )
        pack_digest: str | None = _sha256_bytes(checksums)
    except StrictJsonError as exc:
        pack_digest = None
        errors.append(f"pack checksums cannot be loaded: {exc}")
    report_digests: list[dict[str, str]] = []
    source_commit: object = None
    source_bundle_digest: object = None
    try:
        _source_raw, source = read_json_object_snapshot(
            pack_dir / "metadata" / "source_repo.json",
            label="pack source provenance",
        )
        source_commit = source.get("commit")
        source_bundle_digest = source.get("source_bundle_sha256")
    except StrictJsonError:
        errors.append("pack source provenance cannot be loaded")
    runtime_image_digests: set[str] = set()
    reports_root = pack_dir / "reports"
    if reports_root.is_dir():
        for report_path in sorted(reports_root.glob("**/evaluation.report.json")):
            try:
                report_bytes = read_regular_file_bytes(report_path, label="pack report")
            except StrictJsonError as exc:
                errors.append(f"pack report cannot be loaded: {exc}")
                continue
            report_digests.append(
                {
                    "path": report_path.relative_to(pack_dir).as_posix(),
                    "digest": _sha256_bytes(report_bytes),
                }
            )
        for runtime_path in sorted(reports_root.glob("**/runtime.manifest.json")):
            try:
                _runtime_raw, runtime = read_json_object_snapshot(
                    runtime_path, label="pack runtime manifest"
                )
            except StrictJsonError:
                errors.append("pack runtime manifest cannot be loaded")
                continue
            runtime_section = runtime.get("runtime")
            image_digest = (
                runtime_section.get("image_digest")
                if isinstance(runtime_section, Mapping)
                else None
            )
            if isinstance(image_digest, str):
                runtime_image_digests.add(image_digest)
    runtime_image_digest: str | None = None
    if len(runtime_image_digests) == 1:
        runtime_image_digest = next(iter(runtime_image_digests))
    else:
        errors.append("pack runtime image identity is not singular")
    return (
        {
            "pack_digest": pack_digest,
            "manifest_digest": _sha256_bytes(manifest_bytes),
            "report_digests": report_digests,
            "source_commit": source_commit,
            "source_bundle_digest": source_bundle_digest,
            "runtime_image_digest": runtime_image_digest,
        },
        manifest,
        errors,
    )


def _pack_binding_errors(
    pack_dir: Path, *, catalog: EvidenceCatalog, manifest: Mapping[str, object] | None
) -> tuple[str | None, list[str]]:
    if not isinstance(manifest, Mapping):
        return None, ["pack manifest must be an object"]
    binding = manifest.get("catalog")
    if not isinstance(binding, Mapping):
        return None, ["pack manifest has no catalog binding"]
    lane_id = binding.get("entry_id")
    if not isinstance(lane_id, str) or _IDENTIFIER_RE.fullmatch(lane_id) is None:
        return None, ["catalog entry_id has an invalid format"]
    errors: list[str] = []
    catalog_path = binding.get("path")
    if catalog_path != _CATALOG_PATH:
        errors.append("catalog path must be metadata/catalog.json")
    embedded_catalog_path = pack_dir / _CATALOG_PATH
    try:
        embedded = load_evidence_catalog(embedded_catalog_path)
    except EvidenceCatalogError as exc:
        errors.append(f"embedded catalog is invalid: {exc}")
        embedded = None
    if binding.get("digest") != catalog.digest:
        errors.append("catalog digest does not match supplied catalog")
    if embedded is not None and embedded.digest != catalog.digest:
        errors.append("embedded catalog digest does not match supplied catalog")
    entry = catalog.entries.get(lane_id)
    if entry is None:
        errors.append(f"catalog entry_id is not in supplied catalog: {lane_id}")
    elif binding.get("entry_digest") != entry_digest(entry):
        errors.append("catalog entry digest does not match supplied catalog")
    return lane_id, errors


def verify_evidence_pack_set(
    *,
    catalog_path: Path,
    pack_dirs: Sequence[Path],
    receipt_path: Path,
    expected_catalog_digest: str,
    expected_source_commit: str,
    expected_source_bundle_digest: str,
    expected_runtime_image_digest: str,
    expected_fingerprint: str | None = None,
    trust_store_path: Path | None = None,
    policy_pack_path: Path | None = None,
) -> EvidencePackResult:
    """Verify every supplied pack and require exact catalog coverage."""

    _preflight_pack_set_paths(pack_dirs=pack_dirs, receipt_path=receipt_path)
    payload: dict[str, object]
    anchor_errors: list[str] = []
    if _SHA256_RE.fullmatch(expected_catalog_digest) is None:
        anchor_errors.append("independent_catalog_anchor_required")
    if _COMMIT_RE.fullmatch(expected_source_commit) is None:
        anchor_errors.append("independent_source_commit_anchor_required")
    if _SHA256_RE.fullmatch(expected_source_bundle_digest) is None:
        anchor_errors.append("independent_source_bundle_anchor_required")
    if _SHA256_RE.fullmatch(expected_runtime_image_digest) is None:
        anchor_errors.append("independent_runtime_image_anchor_required")
    if anchor_errors:
        payload = {
            "format_version": EVIDENCE_PACK_SET_RECEIPT_FORMAT,
            "ok": False,
            "catalog_digest": None,
            "source_commit": None,
            "source_bundle_digest": None,
            "runtime_image_digest": None,
            "expected_entry_ids": [],
            "observed_entry_ids": [],
            "duplicate_entry_ids": [],
            "missing_entry_ids": [],
            "extra_entry_ids": [],
            "packs": [],
            "errors": anchor_errors,
        }
        _write_receipt(receipt_path, payload)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.USAGE)
    if expected_fingerprint is None and trust_store_path is None:
        payload = {
            "format_version": EVIDENCE_PACK_SET_RECEIPT_FORMAT,
            "ok": False,
            "catalog_digest": None,
            "source_commit": expected_source_commit,
            "source_bundle_digest": expected_source_bundle_digest,
            "runtime_image_digest": expected_runtime_image_digest,
            "expected_entry_ids": [],
            "observed_entry_ids": [],
            "duplicate_entry_ids": [],
            "missing_entry_ids": [],
            "extra_entry_ids": [],
            "packs": [],
            "errors": ["independent_trust_anchor_required"],
        }
        _write_receipt(receipt_path, payload)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.USAGE)

    errors: list[str] = []
    try:
        catalog = load_evidence_catalog(catalog_path)
    except EvidenceCatalogError:
        payload = {
            "format_version": EVIDENCE_PACK_SET_RECEIPT_FORMAT,
            "ok": False,
            "catalog_digest": None,
            "source_commit": expected_source_commit,
            "source_bundle_digest": expected_source_bundle_digest,
            "runtime_image_digest": expected_runtime_image_digest,
            "expected_entry_ids": [],
            "observed_entry_ids": [],
            "duplicate_entry_ids": [],
            "missing_entry_ids": [],
            "extra_entry_ids": [],
            "packs": [],
            # Receipts are intended to be shareable.  Do not copy loader text
            # here because it can include a caller-controlled absolute path.
            "errors": ["catalog_invalid"],
        }
        _write_receipt(receipt_path, payload)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)
    if catalog.digest != expected_catalog_digest:
        payload = {
            "format_version": EVIDENCE_PACK_SET_RECEIPT_FORMAT,
            "ok": False,
            "catalog_digest": catalog.digest,
            "source_commit": expected_source_commit,
            "source_bundle_digest": expected_source_bundle_digest,
            "runtime_image_digest": expected_runtime_image_digest,
            "expected_entry_ids": sorted(catalog.entries),
            "observed_entry_ids": [],
            "duplicate_entry_ids": [],
            "missing_entry_ids": [],
            "extra_entry_ids": [],
            "packs": [],
            "errors": ["catalog_digest_mismatch"],
        }
        _write_receipt(receipt_path, payload)
        return EvidencePackResult(payload=payload, status=EvidencePackStatus.FORMAT)

    observed: list[str] = []
    pack_payloads: list[dict[str, object]] = []
    provenance: dict[str, object]
    for pack_dir in pack_dirs:
        snapshot, snapshot_errors = PackSnapshot.capture(pack_dir)
        if snapshot is None:
            provenance = {
                "pack_digest": None,
                "manifest_digest": None,
                "report_digests": [],
                "source_commit": None,
                "source_bundle_digest": None,
                "runtime_image_digest": None,
            }
            manifest = None
            provenance_errors = list(snapshot_errors)
            binding_errors = ["pack could not be snapshotted"]
            lane_id = None
            verification = EvidencePackResult(
                payload={"ok": False, "authenticity": "unpinned"},
                status=EvidencePackStatus.INTEGRITY,
            )
        else:
            with snapshot.files.materialized() as snapshot_root:
                provenance, manifest, provenance_errors = _pack_provenance(
                    snapshot_root
                )
                lane_id, binding_errors = _pack_binding_errors(
                    snapshot_root, catalog=catalog, manifest=manifest
                )
                entry = catalog.entries.get(lane_id) if lane_id is not None else None
                execution = (
                    entry.get("execution") if isinstance(entry, Mapping) else None
                )
                pack_profile = (
                    execution.get("profile") if isinstance(execution, Mapping) else None
                )
                verification = verify_evidence_pack(
                    snapshot_root,
                    strict=True,
                    profile=str(pack_profile or "invalid"),
                    report_assurance="strict",
                    expected_fingerprint=expected_fingerprint,
                    trust_store_path=trust_store_path,
                    expected_catalog_digest=expected_catalog_digest,
                    expected_runtime_image_digest=expected_runtime_image_digest,
                    policy_pack_path=policy_pack_path,
                )
                materialized_errors = snapshot.files.materialized_stability_errors(
                    snapshot_root
                )
            snapshot_stability_errors = [
                *materialized_errors,
                *snapshot.stability_errors(),
            ]
            if snapshot_stability_errors:
                provenance_errors.extend(snapshot_stability_errors)
        if lane_id is not None:
            observed.append(lane_id)
        pack_error_codes: list[str] = []
        if provenance_errors:
            errors.append("pack_provenance_invalid")
            pack_error_codes.append("pack_provenance_invalid")
        if binding_errors:
            errors.append("catalog_binding_invalid")
            pack_error_codes.append("catalog_binding_invalid")
        if not verification.payload.get("ok", False):
            errors.append("strict pack verification failed")
            pack_error_codes.append("strict_pack_verification_failed")
        authenticity = verification.payload.get("authenticity")
        if authenticity != "pinned":
            errors.append("pack signer is not pinned")
            pack_error_codes.append("pack_signer_not_pinned")
        if provenance.get("source_commit") != expected_source_commit:
            errors.append("source commit mismatch")
            pack_error_codes.append("source_commit_mismatch")
        if provenance.get("source_bundle_digest") != expected_source_bundle_digest:
            errors.append("source bundle mismatch")
            pack_error_codes.append("source_bundle_mismatch")
        if provenance.get("runtime_image_digest") != expected_runtime_image_digest:
            errors.append("runtime image mismatch")
            pack_error_codes.append("runtime_image_mismatch")
        pack_payloads.append(
            {
                "entry_id": lane_id,
                **provenance,
                "ok": not pack_error_codes,
                "status": verification.status.value,
                "authenticity": authenticity,
                "signer_fingerprint": verification.payload.get("signer_fingerprint"),
                "errors": pack_error_codes,
            }
        )

    expected = set(catalog.entries)
    observed_set = set(observed)
    duplicate_ids = sorted(
        {lane_id for lane_id in observed if observed.count(lane_id) > 1}
    )
    missing_ids = sorted(expected - observed_set)
    extra_ids = sorted(observed_set - expected)
    if duplicate_ids:
        errors.append("duplicate catalog entries: " + ", ".join(duplicate_ids))
    if missing_ids:
        errors.append("missing catalog entries: " + ", ".join(missing_ids))
    if extra_ids:
        errors.append("unknown catalog entries: " + ", ".join(extra_ids))
    payload = {
        "format_version": EVIDENCE_PACK_SET_RECEIPT_FORMAT,
        "ok": not errors,
        "catalog_digest": catalog.digest,
        "source_commit": expected_source_commit,
        "source_bundle_digest": expected_source_bundle_digest,
        "runtime_image_digest": expected_runtime_image_digest,
        "expected_entry_ids": sorted(expected),
        "observed_entry_ids": sorted(observed),
        "duplicate_entry_ids": duplicate_ids,
        "missing_entry_ids": missing_ids,
        "extra_entry_ids": extra_ids,
        "packs": sorted(
            pack_payloads,
            key=lambda item: (
                str(item.get("entry_id") or ""),
                str(item.get("pack_digest") or ""),
                str(item.get("manifest_digest") or ""),
            ),
        ),
        "errors": errors,
    }
    _write_receipt(receipt_path, payload)
    status = EvidencePackStatus.OK if not errors else EvidencePackStatus.FORMAT
    return EvidencePackResult(payload=payload, status=status)


__all__ = ["verify_evidence_pack_set"]
