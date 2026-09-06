"""Optional signed handoff of a replayable pipeline comparison."""

from __future__ import annotations

import base64
import copy
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline.comparison import compare_runs
from invarlock.pipeline.contracts import PipelineError, digest, validate

DOMAIN = b"invarlock/pipeline-evidence-v1\x00"


def create_evidence(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
    signing_key: Ed25519PrivateKey | None = None,
) -> dict[str, Any]:
    """Bind the complete inputs and comparison; unsigned local use is explicit."""
    if signing_key is not None and not isinstance(signing_key, Ed25519PrivateKey):
        raise PipelineError("signing key must be Ed25519")
    for value, name in ((baseline, "run"), (candidate, "run"), (policy, "policy")):
        validate(value, name)
    baseline, candidate, policy = copy.deepcopy((baseline, candidate, policy))
    payload = {
        "format": "invarlock/pipeline-evidence-v1",
        "baseline": baseline,
        "candidate": candidate,
        "policy": policy,
        "comparison": compare_runs(baseline, candidate, policy),
    }
    signature = None
    if signing_key is not None:
        signature = {
            "algorithm": "Ed25519",
            "value": base64.b64encode(
                signing_key.sign(DOMAIN + canonical_json_bytes(payload))
            ).decode("ascii"),
        }
    result = {**payload, "signature": signature}
    validate(result, "evidence")
    return result


def verify_evidence(
    evidence: dict[str, Any],
    *,
    public_key: Ed25519PublicKey,
    expected_baseline: str,
    expected_candidate: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Verify using recipient-owned key, complete-run digests and policy bytes."""
    if not isinstance(public_key, Ed25519PublicKey):
        raise PipelineError("verification key must be Ed25519")
    validate(evidence, "evidence")
    if evidence["signature"] is None:
        raise PipelineError(
            "unsigned local comparison cannot be independently authenticated"
        )
    if (
        digest(evidence["baseline"]) != expected_baseline
        or digest(evidence["candidate"]) != expected_candidate
    ):
        raise PipelineError("run digest differs from the independently expected run")
    if canonical_json_bytes(evidence["policy"]) != canonical_json_bytes(policy):
        raise PipelineError(
            "policy differs from the recipient's independently supplied policy"
        )
    payload = {k: v for k, v in evidence.items() if k != "signature"}
    try:
        public_key.verify(
            base64.b64decode(evidence["signature"]["value"], validate=True),
            DOMAIN + canonical_json_bytes(payload),
        )
    except (InvalidSignature, ValueError) as exc:
        raise PipelineError("pipeline evidence signature is invalid") from exc
    replayed = compare_runs(evidence["baseline"], evidence["candidate"], policy)
    if canonical_json_bytes(replayed) != canonical_json_bytes(evidence["comparison"]):
        raise PipelineError(
            "signed comparison differs from independent arithmetic replay"
        )
    return replayed
