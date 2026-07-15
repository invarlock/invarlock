# ruff: noqa: E402  # Direct script execution must establish package import roots first.
"""Produce a verifier-bound v1 artifact-replay proof for a training subject.

This is deliberately a post-training producer.  It does not train a model,
replay optimizer steps, or manufacture an artifact.  Instead it requires an
already-published subject and invokes the independently owned training artifact
verifier, whose checks include a fresh saved-artifact reload and deterministic
finite inference.  The resulting sidecar contains only portable identities and
facts already bound by that verifier.

The proof does not attest historical optimizer execution. Its profile-specific
backend field is a constrained producer declaration; offline validation
establishes artifact state and reload behavior instead.

The output is intentionally kept outside the checkpoint tree.  A proof must
not change the identity of the artifact it claims to verify.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[4]

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_sha256,
)
from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot
from invarlock.training_evidence import (
    LORA_MERGE_PROOF_SCHEMA,
    TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND,
    TRAINING_ARTIFACT_REPLAY_SCHEMA,
    TRAINING_EVIDENCE_PROOF_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
    TrainingEvidenceProofError,
    canonical_producer_declared_training_backend,
    require_valid_training_evidence_proof,
    with_training_evidence_proof_digest,
)

if __package__ not in {None, ""}:
    from .training_artifact_verifier import verify_training_artifact
    from .training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        LoraTrainingProfile,
        TrainingProfile,
        TrainingProfileError,
        load_training_profile,
    )
    from .training_receipt import TrainingReceiptError, require_valid_training_receipt
    from .training_runtime import TrainingRuntimeError
else:  # pragma: no cover - direct script-path loading
    from scripts.evidence_packs.python.editing.training_artifact_verifier import (
        verify_training_artifact,
    )
    from scripts.evidence_packs.python.editing.training_contract import (
        DEFAULT_TRAINING_PROFILES_PATH,
        LoraTrainingProfile,
        TrainingProfile,
        TrainingProfileError,
        load_training_profile,
    )
    from scripts.evidence_packs.python.editing.training_receipt import (
        TrainingReceiptError,
        require_valid_training_receipt,
    )
    from scripts.evidence_packs.python.editing.training_runtime import (
        TrainingRuntimeError,
    )


TRAINING_RECEIPT_FILENAME = "training_receipt.json"


class TrainingEvidenceProofProducerError(RuntimeError):
    """Raised when an existing training artifact cannot support a proof."""


def _require_regular_directory(path: Path, *, label: str) -> Path:
    """Resolve one non-symlinked directory before reading its checkpoint tree."""

    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise TrainingEvidenceProofProducerError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        raise TrainingEvidenceProofProducerError(f"{label} must be a regular directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise TrainingEvidenceProofProducerError(f"{label} is unavailable") from exc


def _subject_identity(subject_dir: Path) -> dict[str, str]:
    try:
        digest = checkpoint_tree_sha256(subject_dir)
    except (CheckpointIdentityError, OSError, ValueError) as exc:
        raise TrainingEvidenceProofProducerError(
            "subject artifact identity is unavailable"
        ) from exc
    return {"kind": "local_checkpoint_tree", "sha256": digest}


def _baseline_identity(profile: TrainingProfile) -> dict[str, str]:
    """Return the immutable upstream baseline declared by a typed profile."""

    return {"kind": "remote_revision", "revision": profile.model_revision}


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingEvidenceProofProducerError(f"{label} is unavailable")
    return value


def _receipt_snapshot(subject_dir: Path) -> tuple[bytes, dict[str, Any]]:
    receipt_path = subject_dir / TRAINING_RECEIPT_FILENAME
    try:
        snapshot = read_json_object_snapshot(receipt_path, label="training receipt")
        return cast(tuple[bytes, dict[str, Any]], snapshot)
    except StrictJsonError as exc:
        raise TrainingEvidenceProofProducerError(
            "training receipt is unavailable or not strict JSON"
        ) from exc


def _require_receipt_for_profile(
    receipt: Mapping[str, object], *, profile: TrainingProfile
) -> dict[str, Any]:
    """Authenticate the exact receipt snapshot before its independent replay."""

    try:
        validated = require_valid_training_receipt(receipt, profile=profile)
    except TrainingReceiptError as exc:
        raise TrainingEvidenceProofProducerError(
            "training receipt does not bind the requested immutable profile"
        ) from exc
    if validated.get("schema") != TRAINING_RECEIPT_SCHEMA:
        raise TrainingEvidenceProofProducerError(
            "training receipt has an unknown schema"
        )
    if (
        validated.get("profile_id") != profile.profile_id
        or validated.get("profile_sha256") != profile.profile_sha256
        or validated.get("edit_type") != profile.edit_type
    ):
        raise TrainingEvidenceProofProducerError(
            "training receipt does not bind the requested immutable profile"
        )
    return cast(dict[str, Any], validated)


def _artifact_replay(
    *,
    receipt: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> dict[str, object]:
    hashes = _required_mapping(receipt.get("hashes"), label="training receipt hashes")
    changes = _required_mapping(
        receipt.get("changes"), label="training receipt changes"
    )
    model = _required_mapping(receipt.get("model"), label="training receipt model")
    baseline_load = _required_mapping(
        model.get("baseline_load"), label="training receipt baseline load"
    )
    return {
        "schema": TRAINING_ARTIFACT_REPLAY_SCHEMA,
        "passed": True,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "baseline_identity": dict(baseline_identity),
        "artifact_identity": dict(artifact_identity),
        "baseline_tree_sha256": hashes.get("baseline_tree_sha256"),
        "subject_tree_sha256": hashes.get("subject_tree_sha256"),
        "baseline_state_sha256": hashes.get("baseline_state_sha256"),
        "post_training_state_sha256": hashes.get("post_training_state_sha256"),
        "reloaded_subject_state_sha256": hashes.get("reloaded_subject_state_sha256"),
        "delta_sha256": hashes.get("delta_sha256"),
        "changed_tensors": changes.get("changed_tensors"),
        "changed_params": changes.get("changed_params"),
        "total_params": changes.get("total_params"),
        "max_abs_delta": changes.get("max_abs_delta"),
        "baseline_load_diagnostics_sha256": baseline_load.get("diagnostics_sha256"),
        "loss_function": baseline_load.get("loss_function"),
        "saved_artifact_verified": True,
        "reloaded_artifact_verified": True,
    }


def _runtime_reload(
    *, receipt: Mapping[str, object], artifact_identity: Mapping[str, str]
) -> dict[str, object]:
    hashes = _required_mapping(receipt.get("hashes"), label="training receipt hashes")
    smoke = _required_mapping(
        receipt.get("reload_smoke"), label="training receipt reload evidence"
    )
    runtime = _required_mapping(
        receipt.get("runtime"), label="training receipt runtime"
    )
    return {
        "schema": TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
        "passed": True,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "artifact_identity": dict(artifact_identity),
        "subject_state_sha256": hashes.get("post_training_state_sha256"),
        "reload_runs": smoke.get("repeat_runs"),
        "input_sha256": smoke.get("input_sha256"),
        "logits_sha256": smoke.get("logits_sha256"),
        "logits_shape": smoke.get("logits_shape"),
        "all_logits_finite": True,
        "repeat_deterministic": True,
        "device": runtime.get("device"),
    }


def _lora_merge_proof(
    subject_dir: Path, receipt: Mapping[str, object]
) -> dict[str, object]:
    """Bind the saved adapter tree alongside the independently merged subject."""

    lora = _required_mapping(receipt.get("lora"), label="LoRA training receipt")
    adapter_dir = _require_regular_directory(
        subject_dir / "adapter", label="serialized LoRA adapter"
    )
    adapter_identity = _subject_identity(adapter_dir)
    return {
        "schema": LORA_MERGE_PROOF_SCHEMA,
        "adapter_identity": adapter_identity,
        "adapter_tree_sha256": lora.get("adapter_tree_sha256"),
        "profile_lora_config_sha256": lora.get("profile_lora_config_sha256"),
        "serialized_adapter_config_sha256": lora.get(
            "serialized_adapter_config_sha256"
        ),
        "initial_adapter_state_sha256": lora.get("initial_adapter_state_sha256"),
        "trained_adapter_state_sha256": lora.get("trained_adapter_state_sha256"),
        "serialized_adapter_state_sha256": lora.get("serialized_adapter_state_sha256"),
        "base_state_before_adapter_sha256": lora.get(
            "base_state_before_adapter_sha256"
        ),
        "base_state_after_training_sha256": lora.get(
            "base_state_after_training_sha256"
        ),
        "base_state_manifest_sha256": lora.get("base_state_manifest_sha256"),
        "base_state_manifest_before_adapter_sha256": lora.get(
            "base_state_manifest_before_adapter_sha256"
        ),
        "base_state_manifest_after_training_sha256": lora.get(
            "base_state_manifest_after_training_sha256"
        ),
        "state_evidence_policy": lora.get("state_evidence_policy"),
        "expected_merge_target_names_sha256": lora.get(
            "expected_merge_target_names_sha256"
        ),
        "merge_target_names": lora.get("merge_target_names"),
        "observed_merged_changed_names_sha256": lora.get(
            "observed_merged_changed_names_sha256"
        ),
        "merged_changed_tensor_count": lora.get("merged_changed_tensor_count"),
        "merge_scope_exact": lora.get("merge_scope_exact"),
        "merged_state_sha256": lora.get("merged_state_sha256"),
        "adapter_optimizer_steps": lora.get("adapter_optimizer_steps"),
        "trainable_parameter_count": lora.get("trainable_parameter_count"),
        "adapter_modules_before_merge": lora.get("adapter_modules_before_merge"),
        "adapter_modules_after_merge": lora.get("adapter_modules_after_merge"),
        "merge_method": lora.get("merge_method"),
        "adapter_training_performed": lora.get("adapter_training_performed"),
        "adapter_merge_performed": lora.get("adapter_merge_performed"),
    }


def _producer_declared_training_backend(profile: TrainingProfile) -> str:
    """Return the constrained producer label without attesting history."""

    return canonical_producer_declared_training_backend(profile.edit_type)


def _build_training_evidence_proof(
    *,
    profile: TrainingProfile,
    receipt: Mapping[str, object],
    subject_dir: Path,
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> dict[str, object]:
    """Build a closed v1 envelope for replayed artifact facts.

    The backend label is kept separate as a producer declaration, not an
    independently verified account of historical execution.
    """

    proof: dict[str, object] = {
        "schema": TRAINING_EVIDENCE_PROOF_SCHEMA,
        "edit_type": profile.edit_type,
        "provenance": {
            "kind": TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND,
            "producer_declared_training_backend": (
                _producer_declared_training_backend(profile)
            ),
        },
        "training_receipt": {
            "schema": receipt.get("schema"),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "profile_id": receipt.get("profile_id"),
            "profile_sha256": receipt.get("profile_sha256"),
            "edit_type": receipt.get("edit_type"),
            "dataset_provider": receipt.get("dataset_provider"),
        },
        "baseline_identity": dict(baseline_identity),
        "artifact_identity": dict(artifact_identity),
        "artifact_replay": _artifact_replay(
            receipt=receipt,
            baseline_identity=baseline_identity,
            artifact_identity=artifact_identity,
        ),
        "runtime_reload": _runtime_reload(
            receipt=receipt,
            artifact_identity=artifact_identity,
        ),
    }
    if isinstance(profile, LoraTrainingProfile):
        proof["lora_merge"] = _lora_merge_proof(subject_dir, receipt)
    return cast(dict[str, object], with_training_evidence_proof_digest(proof))


def _resolved_output_outside_subject(subject_dir: Path, output_path: Path) -> Path:
    """Resolve a sidecar target while forbidding any checkpoint-tree mutation."""

    candidate = output_path.expanduser().absolute()
    if candidate.name in {"", ".", ".."}:
        raise TrainingEvidenceProofProducerError("proof output path is invalid")
    try:
        if candidate.is_symlink():
            raise TrainingEvidenceProofProducerError(
                "proof output must not be a symlink"
            )
    except OSError as exc:
        raise TrainingEvidenceProofProducerError(
            "proof output path is unavailable"
        ) from exc
    try:
        # ``strict=False`` is intentional: output parents are created only
        # after all replay checks pass.  This still resolves every existing
        # ancestor and normalizes a nested path before any mutation occurs.
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise TrainingEvidenceProofProducerError(
            "proof output path is unavailable"
        ) from exc
    try:
        resolved.relative_to(subject_dir)
    except ValueError:
        return resolved
    raise TrainingEvidenceProofProducerError(
        "proof output must be outside the subject artifact tree"
    )


def _read_exact_existing_proof(
    path: Path,
    *,
    receipt: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
    expected: Mapping[str, object],
) -> bool:
    """Return whether an already-published sidecar is the exact proof needed."""

    try:
        _, existing = read_json_object_snapshot(path, label="existing training proof")
    except StrictJsonError as exc:
        raise TrainingEvidenceProofProducerError(
            "existing training proof is unavailable or not strict JSON"
        ) from exc
    try:
        validated = require_valid_training_evidence_proof(
            existing,
            receipt,
            expected_edit_type=receipt.get("edit_type")
            if isinstance(receipt.get("edit_type"), str)
            else None,
            expected_baseline_identity=baseline_identity,
            expected_artifact_identity=artifact_identity,
        )
    except TrainingEvidenceProofError as exc:
        raise TrainingEvidenceProofProducerError(
            "existing training proof fails closed validation"
        ) from exc
    return bool(validated == dict(expected))


def _publish_proof_idempotently(
    path: Path,
    *,
    subject_dir: Path,
    proof: Mapping[str, object],
    receipt: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> None:
    """Atomically create a proof, accepting only an exact prior publication."""

    if path.exists() or path.is_symlink():
        if _read_exact_existing_proof(
            path,
            receipt=receipt,
            baseline_identity=baseline_identity,
            artifact_identity=artifact_identity,
            expected=proof,
        ):
            return
        raise TrainingEvidenceProofProducerError(
            "refusing to overwrite a different training evidence proof"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        published_path = (path.parent.resolve(strict=True) / path.name).resolve(
            strict=False
        )
    except OSError as exc:
        raise TrainingEvidenceProofProducerError(
            "proof output path is unavailable"
        ) from exc
    try:
        published_path.relative_to(subject_dir)
    except ValueError:
        pass
    else:
        raise TrainingEvidenceProofProducerError(
            "proof output must be outside the subject artifact tree"
        )
    if published_path != path:
        raise TrainingEvidenceProofProducerError(
            "proof output path changed before write"
        )

    encoded = (
        json.dumps(dict(proof), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            if _read_exact_existing_proof(
                path,
                receipt=receipt,
                baseline_identity=baseline_identity,
                artifact_identity=artifact_identity,
                expected=proof,
            ):
                return
            raise TrainingEvidenceProofProducerError(
                "refusing to overwrite a different training evidence proof"
            ) from None
        except OSError as exc:
            raise TrainingEvidenceProofProducerError(
                "could not atomically publish training evidence proof"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def produce_training_evidence_proof(
    *,
    profile_id: str,
    subject_dir: Path,
    output_path: Path,
    profiles_path: Path = DEFAULT_TRAINING_PROFILES_PATH,
    repo_root: Path | None = None,
    local_files_only: bool = True,
) -> dict[str, object]:
    """Independently replay an existing training artifact and publish its proof.

    The function is deterministic and idempotent: it never replaces a prior
    sidecar, and a rerun succeeds only if that sidecar is the exact same proof
    for the currently verified receipt and artifact identity.
    """

    subject_root = _require_regular_directory(subject_dir, label="training subject")
    output = _resolved_output_outside_subject(subject_root, output_path)
    verifier_root = repo_root if repo_root is not None else _REPO_ROOT
    try:
        profile = load_training_profile(
            profile_id,
            profiles_path=profiles_path,
            repo_root=verifier_root,
        )
    except TrainingProfileError as exc:
        raise TrainingEvidenceProofProducerError(
            "requested training profile is unavailable or invalid"
        ) from exc

    receipt_before_bytes, receipt_before = _receipt_snapshot(subject_root)
    verified_receipt = _require_receipt_for_profile(receipt_before, profile=profile)
    identity_before = _subject_identity(subject_root)
    try:
        independently_verified = verify_training_artifact(
            profile,
            subject_root,
            repo_root=verifier_root,
            local_files_only=local_files_only,
        )
    except (TrainingRuntimeError, TrainingReceiptError, OSError, ValueError) as exc:
        raise TrainingEvidenceProofProducerError(
            "independent training artifact verification did not succeed"
        ) from exc
    receipt_after_bytes, receipt_after = _receipt_snapshot(subject_root)
    identity_after = _subject_identity(subject_root)
    if receipt_before_bytes != receipt_after_bytes or receipt_before != receipt_after:
        raise TrainingEvidenceProofProducerError(
            "training receipt changed during independent verification"
        )
    if identity_before != identity_after:
        raise TrainingEvidenceProofProducerError(
            "subject artifact changed during independent verification"
        )
    if independently_verified != verified_receipt:
        raise TrainingEvidenceProofProducerError(
            "independent verifier result does not match the receipt snapshot"
        )

    baseline = _baseline_identity(profile)
    try:
        proof = _build_training_evidence_proof(
            profile=profile,
            receipt=verified_receipt,
            subject_dir=subject_root,
            baseline_identity=baseline,
            artifact_identity=identity_after,
        )
    except TrainingEvidenceProofError as exc:
        raise TrainingEvidenceProofProducerError(
            "generated training evidence proof cannot be canonicalized"
        ) from exc
    if _subject_identity(subject_root) != identity_after:
        raise TrainingEvidenceProofProducerError(
            "subject artifact changed while building the training evidence proof"
        )
    try:
        validated = require_valid_training_evidence_proof(
            proof,
            verified_receipt,
            expected_edit_type=profile.edit_type,
            expected_baseline_identity=baseline,
            expected_artifact_identity=identity_after,
        )
    except TrainingEvidenceProofError as exc:
        raise TrainingEvidenceProofProducerError(
            "generated training evidence proof fails closed validation"
        ) from exc
    _publish_proof_idempotently(
        output,
        subject_dir=subject_root,
        proof=validated,
        receipt=verified_receipt,
        baseline_identity=baseline,
        artifact_identity=identity_after,
    )
    return cast(dict[str, object], validated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--subject", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--profiles-path",
        type=Path,
        default=DEFAULT_TRAINING_PROFILES_PATH,
        help="immutable training profile document (default: repository profile set)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="repository root used to authenticate the immutable training data",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit the independent verifier to fetch a pinned baseline if absent",
    )
    args = parser.parse_args(argv)
    try:
        proof = produce_training_evidence_proof(
            profile_id=args.profile_id,
            subject_dir=args.subject,
            output_path=args.out,
            profiles_path=args.profiles_path,
            repo_root=args.repo_root,
            local_files_only=not args.allow_network,
        )
    except TrainingEvidenceProofProducerError as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(proof, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "TRAINING_RECEIPT_FILENAME",
    "TrainingEvidenceProofProducerError",
    "produce_training_evidence_proof",
]
