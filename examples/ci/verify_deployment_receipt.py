#!/usr/bin/env python3
"""Fail closed unless a signed InvarLock receipt matches deployment anchors."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_receipt import verify_signed_verification_receipt

INPUT_FORMAT = "invarlock/deployment-approval-inputs-v1"
OUTPUT_FORMAT = "invarlock/deployment-approval-v1"
MAX_INPUT_BYTES = 1024 * 1024
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_FIELDS = {
    "artifact_digests",
    "evidence_signer_fingerprint",
    "format",
    "policy_sha256",
    "runtime_digests",
    "schedule_digest",
    "verifier_fingerprint",
    "verifier_identity",
}


class DeploymentApprovalError(ValueError):
    """Raised when independently managed deployment inputs do not authorize."""


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise DeploymentApprovalError(f"{label} must be a sha256 digest")
    return value


def _side_digests(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"baseline", "subject"}:
        raise DeploymentApprovalError(
            f"{label} must contain exactly baseline and subject"
        )
    return {
        side: _digest(value[side], label=f"{label}.{side}")
        for side in ("baseline", "subject")
    }


def load_approval_inputs(path: Path) -> dict[str, Any]:
    """Load a small, strict verifier-owned approval-input document."""

    try:
        payload = read_regular_file_bytes(
            path,
            label="deployment approval inputs",
            max_bytes=MAX_INPUT_BYTES,
        )
        value = parse_json_bytes(payload, label="deployment approval inputs")
    except StrictJsonError as exc:
        raise DeploymentApprovalError(str(exc)) from exc
    if not isinstance(value, dict) or set(value) != EXPECTED_FIELDS:
        raise DeploymentApprovalError("deployment approval input fields are invalid")
    if value.get("format") != INPUT_FORMAT:
        raise DeploymentApprovalError("deployment approval input format is invalid")
    identity = value.get("verifier_identity")
    if not isinstance(identity, str) or not identity.strip():
        raise DeploymentApprovalError("verifier identity must be non-empty")
    result = cast(dict[str, Any], value)
    result["artifact_digests"] = _side_digests(
        value["artifact_digests"], label="artifact digests"
    )
    result["runtime_digests"] = _side_digests(
        value["runtime_digests"], label="runtime digests"
    )
    for field in (
        "evidence_signer_fingerprint",
        "policy_sha256",
        "schedule_digest",
        "verifier_fingerprint",
    ):
        result[field] = _digest(value[field], label=field.replace("_", " "))
    return result


def approve(
    *,
    approval_inputs_path: Path,
    evidence_path: Path,
    policy_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Verify one receipt against independent anchors and return a deploy record."""

    inputs = load_approval_inputs(approval_inputs_path)
    try:
        policy = read_regular_file_bytes(
            policy_path,
            label="deployment acceptance policy",
            max_bytes=4 * 1024 * 1024,
        )
    except StrictJsonError as exc:
        raise DeploymentApprovalError(str(exc)) from exc
    policy_sha256 = "sha256:" + hashlib.sha256(policy).hexdigest()
    if policy_sha256 != inputs["policy_sha256"]:
        raise DeploymentApprovalError("policy digest does not match approval inputs")

    verified = verify_signed_verification_receipt(
        receipt_path,
        evidence_path,
        policy_path=policy_path,
        expected_artifact_digests=inputs["artifact_digests"],
        expected_schedule_digest=inputs["schedule_digest"],
        expected_runtime_digests=inputs["runtime_digests"],
        expected_pack_signer_fingerprint=inputs["evidence_signer_fingerprint"],
        expected_verifier_identity=inputs["verifier_identity"],
        expected_verifier_fingerprint=inputs["verifier_fingerprint"],
    )
    if not verified.ok or verified.statement is None:
        diagnostic = "; ".join(verified.errors) or "signed receipt was not accepted"
        raise DeploymentApprovalError(diagnostic)
    verdict = verified.statement.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("ok") is not True:
        raise DeploymentApprovalError("signed receipt does not authorize deployment")
    manifest_digest = verified.statement.get("pack_manifest_digest")
    return {
        "accepted": True,
        "artifact_digests": inputs["artifact_digests"],
        "format": OUTPUT_FORMAT,
        "pack_manifest_digest": _digest(
            manifest_digest, label="receipt pack manifest digest"
        ),
        "policy_sha256": policy_sha256,
        "runtime_digests": inputs["runtime_digests"],
        "schedule_digest": inputs["schedule_digest"],
        "verifier_fingerprint": verified.verifier_fingerprint,
        "verifier_identity": inputs["verifier_identity"],
    }


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise DeploymentApprovalError(
            f"deployment approval output already exists: {path}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-inputs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = approve(
            approval_inputs_path=args.approval_inputs,
            evidence_path=args.evidence,
            policy_path=args.policy,
            receipt_path=args.receipt,
        )
        payload = canonical_json_bytes(result)
        if args.output is not None:
            _write_new(args.output, payload)
        sys.stdout.write(payload.decode("utf-8"))
    except (DeploymentApprovalError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
