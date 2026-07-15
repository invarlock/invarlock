"""Directed paired replay for runtime behavioral evidence."""

from __future__ import annotations

from pathlib import Path

from invarlock.core.runtime_provider import load_runtime_behavioral_schedule
from invarlock.reporting.validation.runtime_behavioral_claim import (
    verify_runtime_behavioral_claim,
)
from invarlock.runtime_behavioral_claim_receipt import (
    build_runtime_behavioral_claim_receipt,
    canonical_runtime_behavioral_claim_receipt_json,
    verify_runtime_behavioral_claim_receipt,
)

from .contracts import RuntimeBehaviorError, RuntimePairVerification
from .io import atomic_write_new, read_policy_pack_bounded
from .side import load_side_bundle


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def verify_pair(
    *,
    baseline_directory: Path,
    subject_directory: Path,
    schedule_path: Path,
    policy_pack_path: Path,
    receipt_path: Path,
) -> RuntimePairVerification:
    """Replay two directed bundles and publish a positive receipt."""

    baseline_input = Path(baseline_directory)
    subject_input = Path(subject_directory)
    if baseline_input.is_symlink() or subject_input.is_symlink():
        raise RuntimeBehaviorError("baseline and subject side bundles must be real")
    try:
        baseline_root = baseline_input.resolve(strict=True)
        subject_root = subject_input.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBehaviorError(
            "baseline and subject side bundles must exist"
        ) from exc
    if _directory_identity(baseline_root) == _directory_identity(subject_root):
        raise RuntimeBehaviorError("baseline and subject side bundles must be distinct")

    schedule = load_runtime_behavioral_schedule(Path(schedule_path))
    policy = read_policy_pack_bounded(Path(policy_pack_path))
    baseline = load_side_bundle(
        baseline_root,
        role="baseline",
        schedule=schedule,
        policy_pack=policy,
    )
    subject = load_side_bundle(
        subject_root,
        role="subject",
        schedule=schedule,
        policy_pack=policy,
    )
    verification = verify_runtime_behavioral_claim(
        baseline_capabilities=baseline.evidence.capabilities,
        subject_capabilities=subject.evidence.capabilities,
        baseline_artifact_identity=baseline.evidence.artifact_identity,
        subject_artifact_identity=subject.evidence.artifact_identity,
        baseline_receipt=baseline.evidence.receipt,
        subject_receipt=subject.evidence.receipt,
        baseline_observation=baseline.evidence.scoring_observation,
        subject_observation=subject.evidence.scoring_observation,
        schedule=schedule,
        policy_pack=policy,
    )
    if not verification.ok:
        raise RuntimeBehaviorError(
            "paired runtime behavioral claim failed: " + "; ".join(verification.errors)
        )
    receipt = build_runtime_behavioral_claim_receipt(
        baseline=baseline.bindings,
        subject=subject.bindings,
        verification=verification,
    )
    verify_runtime_behavioral_claim_receipt(
        receipt,
        expected_baseline=baseline.bindings,
        expected_subject=subject.bindings,
        expected_verification=verification,
    )
    destination = Path(receipt_path)
    resolved_destination = destination.resolve()
    for side in (baseline_root, subject_root):
        try:
            resolved_destination.relative_to(side)
        except ValueError:
            continue
        raise RuntimeBehaviorError(
            "paired receipt must not modify an input side bundle"
        )
    atomic_write_new(
        destination,
        canonical_runtime_behavioral_claim_receipt_json(receipt),
    )
    return RuntimePairVerification(
        verification=verification,
        receipt=receipt,
        receipt_path=destination.resolve(),
    )


__all__ = ["verify_pair"]
