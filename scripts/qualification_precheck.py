#!/usr/bin/env python3
"""Reject invalid verifier inputs before an expensive qualification run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.trust_inputs import TrustInputsError, load_trust_inputs

_MAX_PREFLIGHT_BYTES = 256 * 1024


def _receipt_destination(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name in {"", ".", ".."}:
        raise ValueError("receipt destination must name a file")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("receipt parent must be an existing directory") from exc
    if not parent.is_dir():
        raise ValueError("receipt parent must be an existing directory")
    destination = parent / candidate.name
    if destination.exists() or destination.is_symlink():
        raise ValueError("receipt destination already exists")
    return destination


def validate(
    *,
    preflight: object,
    trust_profile: Path,
    receipt: Path,
) -> dict[str, object]:
    if not isinstance(preflight, dict) or preflight.get("ok") is not True:
        raise ValueError("evaluation preflight result is not successful")
    trust = load_trust_inputs(trust_profile)
    destination = _receipt_destination(receipt)

    if preflight.get("schedule_digest") != trust.expected_schedule_digest:
        raise ValueError("trust profile schedule digest does not match preflight")
    expected_policy = f"sha256:{hashlib.sha256(trust.policy_bytes).hexdigest()}"
    if preflight.get("policy_digest") != expected_policy:
        raise ValueError("trust profile policy does not match preflight")
    observed_artifacts = preflight.get("artifact_digests")
    if not isinstance(observed_artifacts, dict):
        raise ValueError("preflight artifact digests are missing")
    for role in ("baseline", "subject"):
        if observed_artifacts.get(role) != trust.expected_artifact_digests[role]:
            raise ValueError(
                f"trust profile {role} artifact digest does not match preflight"
            )
    if (
        preflight.get("evidence_signer_fingerprint")
        != trust.expected_signer_fingerprint
    ):
        raise ValueError(
            "trust profile evidence signer fingerprint does not match preflight"
        )
    request_digest = preflight.get("request_digest")
    if (
        not isinstance(request_digest, str)
        or len(request_digest) != 71
        or not request_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in request_digest[7:])
    ):
        raise ValueError("preflight normalized request digest is invalid")
    observed_runtimes = preflight.get("runtime_image_digests")
    if not isinstance(observed_runtimes, dict):
        raise ValueError("preflight runtime image digests are missing")
    for role in ("baseline", "subject"):
        if observed_runtimes.get(role) != trust.expected_runtime_digests[role]:
            raise ValueError(
                f"trust profile {role} runtime digest does not match preflight"
            )
    try:
        verifier_key = serialization.load_pem_private_key(
            trust.verifier_signing_key_bytes,
            password=None,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("trust profile verifier signing key is invalid") from exc
    if not isinstance(verifier_key, ed25519.Ed25519PrivateKey):
        raise ValueError("trust profile verifier signing key must be Ed25519")
    verifier_fingerprint = public_key_fingerprint(verifier_key.public_key())
    return {
        "format_version": "invarlock/qualification-precheck-v1",
        "ok": True,
        "receipt": str(destination),
        "policy_digest": expected_policy,
        "artifact_digests": dict(trust.expected_artifact_digests),
        "evidence_signer_fingerprint": trust.expected_signer_fingerprint,
        "request_digest": request_digest,
        "runtime_digests": dict(trust.expected_runtime_digests),
        "schedule_digest": trust.expected_schedule_digest,
        "trust_profile_digest": trust.profile_digest,
        "verifier_fingerprint": verifier_fingerprint,
        "verifier_identity": trust.verifier_identity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-profile", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    raw = sys.stdin.buffer.read(_MAX_PREFLIGHT_BYTES + 1)
    if len(raw) > _MAX_PREFLIGHT_BYTES:
        parser.error("evaluation preflight result is too large")
    try:
        preflight = json.loads(raw)
        result = validate(
            preflight=preflight,
            trust_profile=arguments.trust_profile,
            receipt=arguments.receipt,
        )
    except (json.JSONDecodeError, TrustInputsError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
