#!/usr/bin/env python3
"""Independently validate the signed receipt produced by qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.core.registry import CoreRegistry
from invarlock.core.scorer_extension import (
    ScorerExtensionBinding,
    decode_scorer_binding,
)
from invarlock.evidence_pack_contract import EVIDENCE_PATHS
from invarlock.evidence_pack_integrity import (
    public_key_fingerprint,
    verify_checksums,
    verify_manifest_binds_checksums_payload,
    verify_no_extra_files,
)
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.evidence_pack_snapshot import PackSnapshot
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.runtime_provider_evidence import (
    MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
    decode_runtime_provider_receipt,
)
from invarlock.trust_inputs import load_trust_inputs

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CUDA_DEVICE = re.compile(r"^cuda(?::[0-9]+)?$")


def _acceptance_compatibility(
    *, metric: object, scorer_extension: object
) -> dict[str, str]:
    if (metric is None) == (scorer_extension is None):
        raise ValueError(
            "comparison must select exactly one built-in metric or scorer extension"
        )
    if metric is not None:
        if not isinstance(metric, str) or metric not in {
            "exact_match",
            "normalized_nll_per_utf8_byte",
        }:
            raise ValueError("comparison built-in acceptance metric is invalid")
        return {"kind": "builtin_metric", "metric": metric}
    try:
        binding = decode_scorer_binding(scorer_extension)
    except ValueError as exc:
        raise ValueError("comparison scorer acceptance binding is invalid") from exc
    return _scorer_compatibility(binding)


def _scorer_compatibility(binding: ScorerExtensionBinding) -> dict[str, str]:
    return {
        "configuration_sha256": binding.configuration_sha256,
        "descriptor_sha256": binding.descriptor_sha256,
        "kind": "scorer_extension",
        "scorer_id": binding.scorer_id,
        "scorer_version": binding.scorer_version,
    }


def _runtime_device_class(value: str) -> str:
    if value == "cpu":
        return "cpu"
    if _CUDA_DEVICE.fullmatch(value) is not None:
        return "cuda"
    raise ValueError("expected runtime device must be cpu, cuda, or cuda:<index>")


def _request_compatibility(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"{label} comparison is missing")
    providers: dict[str, str] = {}
    for side in ("baseline", "subject"):
        side_payload = comparison.get(side)
        runtime = (
            side_payload.get("runtime") if isinstance(side_payload, dict) else None
        )
        provider = runtime.get("provider") if isinstance(runtime, dict) else None
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"{label} {side} provider identity is missing")
        providers[side] = provider
    task = comparison.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{label} task identity is missing")
    return {
        "acceptance": _acceptance_compatibility(
            metric=comparison.get("metric"),
            scorer_extension=comparison.get("scorer_extension"),
        ),
        "providers": providers,
        "task": task,
    }


def _authenticated_canary_compatibility(
    evidence: Path, *, expected_manifest_digest: str
) -> dict[str, object]:
    """Read provider/task identity only after its checksum chain is verified."""

    snapshot, capture_errors = PackSnapshot.capture(
        evidence, validate_structural_json=False
    )
    if snapshot is None:
        raise ValueError(
            "canary evidence snapshot failed: " + "; ".join(capture_errors)
        )
    manifest_entry = snapshot.files.entry("manifest.json")
    if (
        manifest_entry is None
        or f"sha256:{manifest_entry.sha256}" != expected_manifest_digest
    ):
        snapshot.files.cleanup()
        raise ValueError("canary manifest changed after signed receipt verification")
    errors: list[str] = []
    compatibility: dict[str, object] | None = None
    materialized_errors: list[str] = []
    try:
        with snapshot.files.materialized() as snapshot_root:
            manifest_bytes = read_regular_file_bytes(
                snapshot_root / "manifest.json",
                label="canary manifest",
                max_bytes=1024 * 1024,
            )
            checksums_bytes = read_regular_file_bytes(
                snapshot_root / "checksums.sha256",
                label="canary checksums",
                max_bytes=1024 * 1024,
            )
            manifest = parse_json_bytes(manifest_bytes, label="canary manifest")
            errors.extend(
                verify_manifest_binds_checksums_payload(manifest, checksums_bytes)
            )
            checksum_errors, covered = verify_checksums(snapshot_root)
            errors.extend(checksum_errors)
            extra_errors, _warnings = verify_no_extra_files(
                snapshot_root, covered_paths=covered, strict=True
            )
            errors.extend(extra_errors)
            if "request.json" not in covered:
                errors.append("canary request.json is not covered by checksums.sha256")
            if not errors:
                request_bytes = read_regular_file_bytes(
                    snapshot_root / "request.json",
                    label="canary request",
                    max_bytes=1024 * 1024,
                )
                compatibility = _request_compatibility(
                    parse_json_bytes(request_bytes, label="canary request"),
                    label="canary request",
                )
                request_providers = compatibility["providers"]
                assert isinstance(request_providers, dict)
                provider_receipts: dict[str, str] = {}
                device_classes: dict[str, str] = {}
                for role in ("baseline", "subject"):
                    relative = EVIDENCE_PATHS[f"{role}_provider_receipt"]
                    if relative not in covered:
                        errors.append(
                            f"canary {relative} is not covered by checksums.sha256"
                        )
                        continue
                    try:
                        receipt = decode_runtime_provider_receipt(
                            read_regular_file_bytes(
                                snapshot_root / relative,
                                label=f"canary {role} runtime provider receipt",
                                max_bytes=MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
                            )
                        )
                    except ValueError as exc:
                        errors.append(
                            f"canary {role} runtime provider receipt is invalid: {exc}"
                        )
                        continue
                    provider_receipts[role] = receipt.plugin.name
                    device_classes[role] = receipt.device.device_kind
                    if request_providers.get(role) != receipt.plugin.name:
                        errors.append(
                            f"canary {role} provider receipt does not match request"
                        )
                if not errors:
                    compatibility["providers"] = provider_receipts
                    compatibility["device_classes"] = device_classes
            materialized_errors = snapshot.files.materialized_stability_errors(
                snapshot_root
            )
    except RuntimeError as exc:
        errors.append(str(exc))
    errors.extend(materialized_errors)
    errors.extend(snapshot.stability_errors())
    if errors or compatibility is None:
        raise ValueError("canary evidence integrity failed: " + "; ".join(errors))
    return compatibility


def _expected_compatibility(
    request: Path, *, request_root: Path, runtime_device: str
) -> dict[str, object]:
    loaded = load_evaluation_request(
        request,
        request_root=request_root,
        provider_resolver=CoreRegistry().get_runtime_provider,
    )
    comparison = loaded.comparison
    acceptance = (
        {"kind": "builtin_metric", "metric": comparison.metric}
        if comparison.scorer_extension is None
        else _scorer_compatibility(comparison.scorer_extension)
    )
    device_class = _runtime_device_class(runtime_device)
    return {
        "acceptance": acceptance,
        "device_classes": {"baseline": device_class, "subject": device_class},
        "providers": {
            "baseline": comparison.baseline.runtime.provider,
            "subject": comparison.subject.runtime.provider,
        },
        "task": comparison.task,
    }


def validate(
    *,
    receipt: Path,
    evidence: Path,
    trust_profile: Path,
    verifier_public_key: Path | None = None,
    expected_runtime_image_digest: str | None = None,
    expected_request: Path | None = None,
    expected_request_root: Path | None = None,
    expected_runtime_device: str | None = None,
) -> dict[str, object]:
    """Verify one captured receipt against the independently loaded trust unit."""

    receipt_bytes = read_regular_file_bytes(
        receipt,
        label="qualification verification receipt",
        max_bytes=1024 * 1024,
    )
    descriptor, snapshot_name = tempfile.mkstemp(
        prefix=".invarlock-receipt-check-",
        suffix=".json",
    )
    snapshot = Path(snapshot_name)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(receipt_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        public_key_bytes = (
            read_regular_file_bytes(
                verifier_public_key,
                label="qualification verifier public key",
                max_bytes=64 * 1024,
            )
            if verifier_public_key is not None
            else None
        )
        trust = load_trust_inputs(
            trust_profile,
            verifier_key_bytes_override=public_key_bytes,
        )
        if verifier_public_key is None:
            key = serialization.load_pem_private_key(
                trust.verifier_signing_key_bytes,
                password=None,
            )
            if not isinstance(key, ed25519.Ed25519PrivateKey):
                raise ValueError("qualification verifier signing key must be Ed25519")
            public_key = key.public_key()
        else:
            public_key = serialization.load_pem_public_key(
                trust.verifier_signing_key_bytes
            )
            if not isinstance(public_key, ed25519.Ed25519PublicKey):
                raise ValueError("qualification verifier public key must be Ed25519")
        verifier_fingerprint = public_key_fingerprint(public_key)
        result = verify_signed_verification_receipt(
            snapshot,
            evidence,
            policy_path=trust.policy_path,
            expected_artifact_digests=dict(trust.expected_artifact_digests),
            expected_schedule_digest=trust.expected_schedule_digest,
            expected_runtime_digests=dict(trust.expected_runtime_digests),
            expected_pack_signer_fingerprint=trust.expected_signer_fingerprint,
            expected_verifier_identity=trust.verifier_identity,
            expected_verifier_fingerprint=verifier_fingerprint,
            expected_trust_profile_digest=trust.profile_digest,
            require_signed=True,
        )
        if not result.ok or not result.signed or result.statement is None:
            detail = "; ".join(result.errors) or "receipt verification failed"
            raise ValueError(detail)
        manifest_digest = result.statement.get("pack_manifest_digest")
        if not isinstance(manifest_digest, str):
            raise ValueError("verified receipt manifest identity is missing")
        checked: dict[str, object] = {
            "format_version": "invarlock/qualification-receipt-check-v1",
            "ok": True,
            "pack_manifest_digest": manifest_digest,
            "receipt_sha256": f"sha256:{hashlib.sha256(receipt_bytes).hexdigest()}",
            "verifier_fingerprint": verifier_fingerprint,
        }
        compatibility_inputs = (
            expected_request,
            expected_request_root,
            expected_runtime_device,
        )
        if any(value is not None for value in compatibility_inputs) and not all(
            value is not None for value in compatibility_inputs
        ):
            raise ValueError(
                "expected request, request root, and runtime device must be supplied "
                "together"
            )
        if expected_runtime_image_digest is not None:
            if _DIGEST.fullmatch(expected_runtime_image_digest) is None:
                raise ValueError(
                    "expected runtime image digest must be a lowercase sha256 digest"
                )
            anchors = result.statement.get("anchors")
            runtime_digests = (
                anchors.get("runtime_digests") if isinstance(anchors, dict) else None
            )
            expected = {
                "baseline": expected_runtime_image_digest,
                "subject": expected_runtime_image_digest,
            }
            if runtime_digests != expected:
                raise ValueError(
                    "verified receipt runtime digests do not match the exact "
                    "qualification image"
                )
            verdict = result.statement.get("verdict")
            required_verdict = {
                "ok": True,
                "integrity_ok": True,
                "policy_verdict": "pass",
                "verification_status": 0,
            }
            if verdict != required_verdict:
                raise ValueError(
                    "verified canary receipt does not record a strict passing verdict"
                )
            checked["runtime_image_digest"] = expected_runtime_image_digest
        if (
            expected_request is not None
            and expected_request_root is not None
            and expected_runtime_device is not None
        ):
            observed_compatibility = _authenticated_canary_compatibility(
                evidence, expected_manifest_digest=manifest_digest
            )
            expected_compatibility = _expected_compatibility(
                expected_request,
                request_root=expected_request_root,
                runtime_device=expected_runtime_device,
            )
            if observed_compatibility != expected_compatibility:
                raise ValueError(
                    "verified canary provider/task/acceptance/device compatibility "
                    "does not match the target qualification request"
                )
            checked["compatibility"] = observed_compatibility
        return checked
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            snapshot.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trust-profile", type=Path, required=True)
    parser.add_argument(
        "--verifier-public-key",
        type=Path,
        help=(
            "Replay a completed receipt with this captured Ed25519 public key "
            "instead of requiring the destroyed verifier signing key."
        ),
    )
    parser.add_argument("--expected-runtime-image-digest")
    parser.add_argument("--expected-request", type=Path)
    parser.add_argument("--expected-request-root", type=Path)
    parser.add_argument("--expected-runtime-device")
    arguments = parser.parse_args(argv)
    try:
        result = validate(
            receipt=arguments.receipt,
            evidence=arguments.evidence,
            trust_profile=arguments.trust_profile,
            verifier_public_key=arguments.verifier_public_key,
            expected_runtime_image_digest=arguments.expected_runtime_image_digest,
            expected_request=arguments.expected_request,
            expected_request_root=arguments.expected_request_root,
            expected_runtime_device=arguments.expected_runtime_device,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
