#!/usr/bin/env python3
"""Execute one pinned evaluator transaction over two Qwen3 sides."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

try:
    from examples.integrations.trust_material import (
        create_trust_material,
        read_external_file,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from trust_material import (  # type: ignore[no-redef]
        create_trust_material,
        read_external_file,
    )
try:
    from examples.integrations.launch import inspect_evaluator_image
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from launch import inspect_evaluator_image  # type: ignore[no-redef]
from invarlock import __version__ as INVARLOCK_VERSION

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderPluginIdentity,
    canonical_runtime_behavioral_schedule_json,
    load_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evaluation_oci import OciEvaluationError
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.runtime_import_authoring import (
    load_external_scoring_records_jsonl,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

from . import adapters
from .config import (
    BATCH_SIZE,
    EVALUATORS,
    EXPECTED_MODEL_ARTIFACTS,
    EXPECTED_MODEL_TREE_DIGESTS,
    EXPECTED_TOKENIZER_DIGESTS,
    MAX_GENERATION_TOKENS,
    MAX_WORKER_ARTIFACT_BYTES,
    PER_RECORD_TIMEOUT_SECONDS,
    RUN_FIELDS,
    SAMPLE_FIELDS,
    SEED,
    WORKER_TIMEOUT_SECONDS,
    BridgeError,
    evaluator_id,
    execution_config,
    task_config,
    worker_timeout_seconds,
)
from .corpora import (
    CorpusProfile,
    corpus_profile,
    corpus_provenance,
    profile_for_dataset,
    profile_for_descriptor,
)

try:
    from examples.integrations.evaluator_transaction.worker import (
        run_evaluator_worker,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    try:
        from evaluator_transaction.worker import run_evaluator_worker
    except ModuleNotFoundError as nested_exc:
        if nested_exc.name not in {
            "evaluator_transaction",
            "evaluator_transaction.worker",
        }:
            raise
        from evaluator_transaction_worker import (  # type: ignore[no-redef]
            run_evaluator_worker,
        )

IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_WORKER_ARTIFACT_BYTES,
) -> bytes:
    """Read one regular file through a stable, no-follow file descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise BridgeError("secure evaluator artifact loading is unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise BridgeError(
            f"{label} could not be opened without following links"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BridgeError(f"{label} must be a regular file")
        if before.st_size > max_bytes:
            raise BridgeError(f"{label} exceeds its size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BridgeError(f"{label} exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (  # noqa: E731 - compact stable projection
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise BridgeError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def evaluator_lock_digest(selected: str, *, container: bool = False) -> str:
    package = EVALUATORS[selected]
    path = Path(package["container_lock"] if container else package["lock"])
    if not container:
        path = REPOSITORY_ROOT / path
    return (
        f"sha256:{digest(_read_regular_file(path, label=f'{selected} evaluator lock'))}"
    )


def _runtime_image_from_environment() -> str:
    value = os.environ.get("INVARLOCK_RUNTIME_IMAGE_ID", "")
    if IMAGE_ID.fullmatch(value) is None:
        raise BridgeError("the worker must receive the inspected runtime image digest")
    return value


def _external_ed25519_key(path: Path, *, label: str) -> ed25519.Ed25519PrivateKey:
    try:
        payload = read_external_file(path, label=label)
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"{label} is not an Ed25519 private key") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise BridgeError(f"{label} is not an Ed25519 private key")
    return key


def _external_ed25519_public_key(path: Path, *, label: str) -> ed25519.Ed25519PublicKey:
    try:
        payload = read_external_file(path, label=label)
        key = serialization.load_pem_public_key(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise BridgeError(f"{label} is not an Ed25519 public key") from exc
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise BridgeError(f"{label} is not an Ed25519 public key")
    return key


def _require_distinct_signers(
    evidence_key: ed25519.Ed25519PrivateKey,
    verifier_key: ed25519.Ed25519PrivateKey,
    builder_key: ed25519.Ed25519PublicKey | None = None,
) -> None:
    fingerprints = [
        public_key_fingerprint(evidence_key.public_key()),
        public_key_fingerprint(verifier_key.public_key()),
    ]
    if builder_key is not None:
        fingerprints.append(public_key_fingerprint(builder_key))
    if len(fingerprints) != len(set(fingerprints)):
        raise BridgeError(
            "evidence, verifier, and builder signing keys must be distinct"
        )


def _inspect_runtime_image(
    engine: str,
    image: str,
    selected: str,
    lock_digest: str,
    *,
    source_commit: str,
    base_image_id: str,
    build_attestation: Path,
    builder_public_key: ed25519.Ed25519PublicKey,
) -> None:
    if engine not in {"docker", "podman"}:
        raise BridgeError("container engine must be docker or podman")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise BridgeError("source commit must be a full lowercase Git commit")
    if IMAGE_ID.fullmatch(base_image_id) is None:
        raise BridgeError("base image identity must be an immutable image digest")
    try:
        inspect_evaluator_image(
            engine=engine,
            image=image,
            repository=REPOSITORY_ROOT,
            attestation_path=build_attestation,
            evaluator=selected,
            evaluator_version=EVALUATORS[selected]["version"],
            lock_sha256=lock_digest,
            expected_entrypoint=(
                "python",
                "-m",
                "evaluator_transaction.cli",
                "worker",
            ),
            source_commit=source_commit,
            base_image_id=base_image_id,
            builder_public_key=builder_public_key,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BridgeError(
            "evaluator build attestation did not authenticate the image"
        ) from exc


def _run_local_cli(command: list[str]) -> str:
    """Run one completion CLI with bounded diagnostics and a hard deadline."""
    try:
        completed = run_bounded_command(
            command,
            capture_output=True,
            check=True,
            timeout_seconds=WORKER_TIMEOUT_SECONDS,
            stdout_limit=4 * 1024 * 1024,
            stderr_limit=256 * 1024,
            label="evaluator completion command",
        )
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or exc.output or "").strip()
        raise BridgeError(
            diagnostic or f"command exited with status {exc.returncode}"
        ) from exc
    except RuntimeError as exc:
        raise BridgeError(str(exc)) from exc
    return completed.stdout or ""


def worker(role: str, model: Path, dataset: Path, output: Path) -> None:
    selected = evaluator_id()
    package = EVALUATORS[selected]
    observed = importlib.metadata.version(package["distribution"])
    if observed != package["version"]:
        raise BridgeError(
            f"the runtime must contain {package['distribution']} {package['version']}"
        )
    runtime_image_digest = _runtime_image_from_environment()
    lock_digest = evaluator_lock_digest(selected, container=True)
    if os.environ.get("INVARLOCK_EVALUATOR_LOCK_SHA256") != lock_digest:
        raise BridgeError("the evaluator lock does not match the inspected image label")
    if (
        output.exists()
        or output.is_symlink()
        or not model.is_dir()
        or dataset.is_symlink()
        or not dataset.is_file()
    ):
        raise BridgeError("worker inputs must exist and output must be new")
    output.mkdir(parents=True)
    model_digest = checkpoint_tree_sha256(model)
    dataset_bytes = _read_regular_file(dataset, label="worker dataset")
    profile_key = os.environ.get("INVARLOCK_CORPUS_PROFILE")
    profile = corpus_profile(profile_key) if profile_key is not None else None
    if profile is not None:
        try:
            if profile_for_dataset(dataset_bytes) != profile:
                raise BridgeError("worker corpus profile does not match the dataset")
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    dataset_digest = digest(dataset_bytes)
    config = task_config("/records.jsonl", selected)
    config_path = output / "task.json"
    config_path.write_bytes(canonical_json_bytes(config))
    config_digest = digest(_read_regular_file(config_path, label="task configuration"))
    generated, scored = adapters._run_upstream_evaluator(model, dataset_bytes, selected)
    if len(generated) != len(scored):
        raise BridgeError("upstream scorer did not return one result per record")
    if profile is not None and len(generated) != profile.record_count:
        raise BridgeError("upstream evaluator returned an incomplete corpus")
    samples: list[dict[str, Any]] = []
    for sample, (score, detail) in zip(generated, scored, strict=True):
        samples.append(
            {
                "record_id": sample["id"],
                "prompt": sample["prompt"],
                "target": sample["expected"],
                "output": sample["output"],
                "input_sha256": digest(sample["prompt"].encode()),
                "target_sha256": digest(sample["expected"].encode()),
                "output_sha256": digest(sample["output"].encode()),
                "reported_score": float(score),
                "score_detail": detail,
                "status": "ok",
            }
        )
    samples_path = output / "samples.jsonl"
    samples_path.write_bytes(b"".join(canonical_json_bytes(item) for item in samples))
    if (
        checkpoint_tree_sha256(model) != model_digest
        or digest(_read_regular_file(dataset, label="worker dataset")) != dataset_digest
    ):
        raise BridgeError("model or dataset changed during evaluator execution")
    execution = execution_config(selected)
    manifest = {
        "format": "invarlock/evaluator-run-v1",
        "role": role,
        "evaluator": selected,
        "evaluator_version": package["version"],
        "task_config": config,
        "task_config_sha256": digest(canonical_json_bytes(config)),
        "execution_config": execution,
        "execution_config_sha256": digest(canonical_json_bytes(execution)),
        "samples": samples_path.name,
        "samples_sha256": digest(_read_regular_file(samples_path, label="samples")),
        "model_tree_sha256": model_digest,
        "dataset_sha256": dataset_digest,
        "evaluator_lock_sha256": lock_digest,
        "runtime_image_digest": runtime_image_digest,
        "record_count": len(samples),
        "stable_id_field": "record_id",
    }
    if config_digest != manifest["task_config_sha256"]:
        raise BridgeError("task configuration digest was not canonicalized")
    (output / "run-manifest.json").write_bytes(canonical_json_bytes(manifest))


def mount_source(path: Path, *, label: str) -> str:
    """Return a stable, representable host path for a restricted bind mount."""

    if path.is_symlink():
        raise BridgeError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BridgeError(f"{label} could not be resolved") from exc
    rendered = str(resolved)
    if any(character in rendered for character in (",", "\n", "\r", "\x00")):
        raise BridgeError(f"{label} cannot be represented in an OCI mount")
    return rendered


def _run_verified_worker(
    *,
    engine: str,
    image: str,
    selected: str,
    role: str,
    prepared: Path,
    output: Path,
    lock_digest: str,
    profile: CorpusProfile | None = None,
) -> None:
    """Re-run the evaluator in the inspected image for this transaction.

    Prepared evaluator output is deliberately not an authority.  Completion
    creates a fresh output directory and only consumes the run produced by the
    inspected image during this call.
    """

    if engine not in {"docker", "podman"}:
        raise BridgeError("container engine must be docker or podman")
    if role not in {"baseline", "subject"}:
        raise BridgeError("evaluator worker role is invalid")
    model = prepared / f"evaluation/models/{role}"
    dataset = prepared / "evaluation/inputs/records.jsonl"
    if not model.is_dir() or model.is_symlink() or not dataset.is_file():
        raise BridgeError("prepared evaluator inputs are missing or unsafe")
    if dataset.is_symlink() or output.exists() or output.is_symlink():
        raise BridgeError("transaction evaluator output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_name = output.name
    if output_name in {"", ".", ".."} or "/" in output_name:
        raise BridgeError("transaction evaluator output name is invalid")
    try:
        completed = run_evaluator_worker(
            engine=engine,
            image=image,
            entrypoint=(
                "python",
                "-m",
                "evaluator_transaction.cli",
                "worker",
            ),
            worker_arguments=(
                "--role",
                role,
                "--model",
                "/model",
                "--dataset",
                "/records.jsonl",
                "--output",
                f"/outputs/{output_name}",
            ),
            model_source=model,
            dataset_source=dataset,
            output=output,
            environment={
                "INVARLOCK_RUNTIME_IMAGE_ID": image,
                "INVARLOCK_EVALUATOR_LOCK_SHA256": lock_digest,
                **(
                    {"INVARLOCK_CORPUS_PROFILE": profile.key}
                    if profile is not None
                    else {}
                ),
            },
            timeout_seconds=worker_timeout_seconds(profile or corpus_profile("quick")),
        )
    except OciEvaluationError as exc:
        raise BridgeError(
            f"{selected} worker control failed for {role}: {exc}"
        ) from exc
    if completed.returncode:
        raise BridgeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"{selected} worker failed for {role}"
        )


def load_run(
    path: Path,
    role: str,
    selected: str,
    profile: CorpusProfile | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        run = json.loads(_read_regular_file(path, label=f"{role} run manifest"))
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{role} run provenance is missing") from exc
    if not isinstance(run, dict) or set(run) != RUN_FIELDS:
        raise BridgeError(f"{role} run provenance is incomplete")
    if (
        run["format"] != "invarlock/evaluator-run-v1"
        or run["role"] != role
        or run["evaluator"] != selected
        or run["evaluator_version"] != EVALUATORS[selected]["version"]
        or run["stable_id_field"] != "record_id"
        or not isinstance(run["samples"], str)
        or not isinstance(run["samples_sha256"], str)
        or not isinstance(run["record_count"], int)
        or isinstance(run["record_count"], bool)
        or run["record_count"] != (profile or corpus_profile("quick")).record_count
        or IMAGE_ID.fullmatch(run["model_tree_sha256"]) is None
        or IMAGE_ID.fullmatch(f"sha256:{run['dataset_sha256']}") is None
        or IMAGE_ID.fullmatch(run["evaluator_lock_sha256"]) is None
        or IMAGE_ID.fullmatch(run["runtime_image_digest"]) is None
        or run["evaluator_lock_sha256"]
        != evaluator_lock_digest(selected, container=False)
        or run["task_config"] != task_config("/records.jsonl", selected)
        or run["task_config_sha256"] != digest(canonical_json_bytes(run["task_config"]))
        or run["execution_config"] != execution_config(selected)
        or run["execution_config_sha256"]
        != digest(canonical_json_bytes(run["execution_config"]))
    ):
        raise BridgeError(f"{role} run provenance is invalid")
    if run["samples"] != "samples.jsonl":
        raise BridgeError(f"{role} per-record output path is not canonical")
    samples = path.parent / run["samples"]
    try:
        sample_bytes = _read_regular_file(samples, label=f"{role} evaluator samples")
    except BridgeError:
        raise
    if (
        digest(sample_bytes) != run["samples_sha256"]
        or len(sample_bytes.splitlines()) != run["record_count"]
    ):
        raise BridgeError(f"{role} per-record output was tampered")
    return cast(dict[str, Any], run), sample_bytes


def load_canonical_samples(sample_bytes: bytes, *, role: str) -> list[dict[str, Any]]:
    """Decode the exact upstream JSONL snapshot for signed provenance."""

    samples: list[dict[str, Any]] = []
    for index, raw in enumerate(sample_bytes.splitlines(), 1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"{role} evaluator sample {index} is not JSON") from exc
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(value, newline=False) != raw
        ):
            raise BridgeError(f"{role} evaluator sample {index} is not canonical JSON")
        samples.append(value)
    return samples


def adapt(samples: Path | bytes, schedule: Any, destination: Path) -> None:
    sample_bytes = (
        _read_regular_file(samples, label="evaluator samples")
        if isinstance(samples, Path)
        else samples
    )
    if not isinstance(sample_bytes, bytes):
        raise BridgeError("evaluator samples must be bytes or a regular file")
    lines = sample_bytes.splitlines()
    if len(lines) != len(schedule.records):
        raise BridgeError("one evaluator sample is required for every schedule record")
    output: list[dict[str, object]] = []
    for index, (raw, expected) in enumerate(
        zip(lines, schedule.records, strict=True), 1
    ):
        sample = json.loads(raw)
        if not isinstance(sample, dict) or not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError(f"sample {index} lacks complete per-record facts")
        if any(
            not isinstance(sample[field], str)
            for field in ("record_id", "prompt", "target", "output", "status")
        ):
            raise BridgeError(f"sample {index} has invalid text facts")
        if (
            sample["record_id"] != expected.record_id
            or sample["prompt"] != expected.input_parts[0].text
            or sample["target"] != expected.expected_output
            or sample["input_sha256"] != digest(sample["prompt"].encode())
            or sample["target_sha256"] != digest(sample["target"].encode())
            or sample["output_sha256"] != digest(sample["output"].encode())
            or sample["status"] != "ok"
        ):
            raise BridgeError(
                f"sample {index} authenticated inputs or output are invalid"
            )
        reported = sample["reported_score"]
        if isinstance(reported, bool) or reported not in (0.0, 1.0):
            raise BridgeError(f"sample {index} has an invalid evaluator score")
        recomputed = 1.0 if sample["output"] == sample["target"] else 0.0
        if float(reported) != recomputed:
            raise BridgeError(f"sample {index} evaluator score disagrees with replay")
        output.append(
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "status": "ok",
                "output_text": sample["output"],
                "output_sha256": sample["output_sha256"],
            }
        )
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in output))
    load_external_scoring_records_jsonl(destination, schedule=schedule)


def imported(role: str) -> dict[str, str]:
    root = f"imports/{role}"
    return {
        key: f"{root}/{name}"
        for key, name in {
            "identity": "model-artifact.identity.json",
            "receipt": "runtime-provider.receipt.json",
            "observation": "runtime-scoring.observation.json",
            "run_report": "report.json",
            "runtime_manifest": "runtime.manifest.json",
            "runtime_config": "run.yaml",
        }.items()
    }


def validate_completed_outputs(evidence: Path, receipt: Path, report: Path) -> None:
    try:
        evaluation = json.loads(
            _read_regular_file(
                evidence / "reports/evaluation.report.json",
                label="evaluation report",
            )
        )
        signed = json.loads(_read_regular_file(receipt, label="verification receipt"))
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "the completed transaction is missing verified outputs"
        ) from exc
    statement = signed.get("statement") if isinstance(signed, dict) else None
    verdict = statement.get("verdict") if isinstance(statement, dict) else None
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("verdict") != "pass"
        or evaluation.get("metric") != "exact_match"
        or not isinstance(verdict, dict)
        or verdict.get("ok") is not True
        or verdict.get("integrity_ok") is not True
        or verdict.get("policy_verdict") != "pass"
        or not report.is_file()
    ):
        raise BridgeError("the completed transaction did not verify a passing result")


def _validate_completion_paths(
    root: Path,
    *,
    build_attestation: Path,
    trust_root: Path,
    key_paths: tuple[tuple[Path, str], ...],
) -> None:
    if build_attestation.is_symlink():
        raise BridgeError("build attestation must not be a symlink")
    try:
        build_attestation.relative_to(root)
    except ValueError:
        pass
    else:
        raise BridgeError("build attestation must remain outside the transaction")
    if trust_root.exists() or trust_root.is_symlink():
        raise BridgeError("trust root must be new and outside the transaction")
    try:
        trust_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise BridgeError("trust root must remain outside the transaction")
    for key_path, label in key_paths:
        try:
            key_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise BridgeError(f"{label} must remain outside the transaction")


def complete(
    root: Path,
    prepared: Path,
    image: str,
    selected: str,
    *,
    container_engine: str = "docker",
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    source_commit: str | None = None,
    base_image_id: str | None = None,
    build_attestation: Path | None = None,
    builder_public_key: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Author strict import inputs and execute evaluate, verify, and report."""

    if (
        selected not in EVALUATORS
        or IMAGE_ID.fullmatch(image) is None
        or evidence_signing_key is None
        or verifier_signing_key is None
        or trust_root is None
        or source_commit is None
        or base_image_id is None
        or build_attestation is None
        or builder_public_key is None
    ):
        raise BridgeError(
            "an inspected image and caller-owned evidence, verifier, and trust roots "
            "builder public key, and build attestation are required"
        )
    if root.exists() or root.is_symlink():
        raise BridgeError("transaction workspace must be new")
    if not prepared.is_dir() or prepared.is_symlink():
        raise BridgeError("prepared workspace must be a real directory")
    _validate_completion_paths(
        root,
        build_attestation=build_attestation,
        trust_root=trust_root,
        key_paths=(
            (evidence_signing_key, "evidence signing key"),
            (verifier_signing_key, "verifier signing key"),
            (builder_public_key, "builder public key"),
        ),
    )
    evidence_key = _external_ed25519_key(
        evidence_signing_key, label="evidence signing key"
    )
    verifier_key = _external_ed25519_key(
        verifier_signing_key, label="verifier signing key"
    )
    builder_key = _external_ed25519_public_key(
        builder_public_key, label="builder public key"
    )
    _require_distinct_signers(evidence_key, verifier_key, builder_key)
    lock_digest = evaluator_lock_digest(selected)
    _inspect_runtime_image(
        container_engine,
        image,
        selected,
        lock_digest,
        source_commit=source_commit,
        base_image_id=base_image_id,
        build_attestation=build_attestation,
        builder_public_key=builder_key,
    )
    prepared_image = (
        _read_regular_file(
            prepared / "runtime-image-id.txt", label="prepared runtime image identity"
        )
        .decode("ascii")
        .strip()
    )
    if prepared_image != image:
        raise BridgeError("prepared runtime image identity does not match inspection")
    try:
        request0 = yaml.safe_load(
            _read_regular_file(
                prepared / "evaluation/request.yaml", label="prepared request"
            )
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BridgeError("prepared request could not be read as YAML") from exc
    comparison0 = request0.get("comparison") if isinstance(request0, dict) else None
    if (
        not isinstance(comparison0, dict)
        or set(comparison0)
        != {"baseline", "subject", "dataset", "policy", "task", "metric"}
        or comparison0.get("metric") != "exact_match"
        or comparison0.get("policy") != "inputs/acceptance.json"
        or comparison0.get("task") != "text_causal"
    ):
        raise BridgeError("prepared request is not the fixed exact-match transaction")
    dataset0 = prepared / "evaluation/inputs/records.jsonl"
    dataset = comparison0.get("dataset")
    if not isinstance(dataset, dict):
        raise BridgeError("prepared request lacks the authenticated dataset")
    raw_dataset = _read_regular_file(dataset0, label="prepared dataset")
    try:
        profile = profile_for_dataset(raw_dataset)
        described_profile = profile_for_descriptor(dataset)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    if described_profile != profile or dataset["sha256"] != digest(raw_dataset):
        raise BridgeError("prepared dataset does not match the request digest")
    corpus_profile_path = prepared / "evaluation/inputs/corpus-profile.json"
    try:
        prepared_corpus_provenance = json.loads(
            _read_regular_file(corpus_profile_path, label="prepared corpus profile")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError("prepared corpus profile is invalid") from exc
    if prepared_corpus_provenance != corpus_provenance(profile):
        raise BridgeError("prepared corpus provenance does not match the dataset")
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset0,
            sha256=digest(raw_dataset),
            format="jsonl",
            name=dataset["name"],
            split=dataset["split"],
            input_field=dataset["input_field"],
            expected_output_field=dataset["expected_output_field"],
            id_field=dataset["id_field"],
        ),
        raw_dataset,
    )
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    (root / "imports").mkdir()
    (root / "upstream").mkdir()
    (root / "verifier").mkdir()
    schedule_path = root / "inputs/schedule.json"
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    schedule = load_runtime_behavioral_schedule(schedule_path)
    policy = _read_regular_file(
        prepared / "evaluation/inputs/acceptance.json", label="prepared policy"
    )
    expected_policy = profile.acceptance_policy()
    if json.loads(policy) != expected_policy:
        raise BridgeError("prepared exact-match policy is not the fixed example policy")
    for role in ("baseline", "subject"):
        output = root / f"upstream/{role}/result"
        _run_verified_worker(
            engine=container_engine,
            image=image,
            selected=selected,
            role=role,
            prepared=prepared,
            output=output,
            lock_digest=lock_digest,
            profile=profile,
        )
    runs = {
        role: load_run(
            root / f"upstream/{role}/result/run-manifest.json",
            role,
            selected,
            profile,
        )
        for role in ("baseline", "subject")
    }
    if (
        runs["baseline"][0]["task_config_sha256"]
        != runs["subject"][0]["task_config_sha256"]
        or runs["baseline"][0]["execution_config_sha256"]
        != runs["subject"][0]["execution_config_sha256"]
    ):
        raise BridgeError(
            "baseline and subject used different evaluator configurations"
        )
    if any(
        run["runtime_image_digest"] != image
        or run["evaluator_lock_sha256"] != lock_digest
        or run["dataset_sha256"] != dataset["sha256"]
        for run, _samples in runs.values()
    ):
        raise BridgeError(
            "fresh worker runs are not bound to the inspected image, lock, and dataset"
        )
    (root / "inputs/acceptance.json").write_bytes(policy)
    provenance = canonical_json_bytes(
        {
            "format": "invarlock/evaluator-provenance-v1",
            "evaluator": selected,
            "evaluator_lock_sha256": lock_digest,
            "runtime_image_digest": image,
            "source_commit": source_commit,
            "base_image_id": base_image_id,
            "corpus_profile": prepared_corpus_provenance,
            "task_config": runs["baseline"][0]["task_config"],
            "task_config_sha256": runs["baseline"][0]["task_config_sha256"],
            "execution_config": runs["baseline"][0]["execution_config"],
            "execution_config_sha256": runs["baseline"][0]["execution_config_sha256"],
            "runs": {
                role: {
                    "manifest": runs[role][0],
                    "samples": load_canonical_samples(runs[role][1], role=role),
                }
                for role in ("baseline", "subject")
            },
        }
    )
    (root / "inputs/evaluator-provenance.json").write_bytes(provenance)
    provider = HFTransformersProvider()
    sides: dict[str, Any] = {}
    anchors: dict[str, str] = {}
    for role in ("baseline", "subject"):
        records_path = root / f"imports/{role}-records.jsonl"
        adapt(runs[role][1], schedule, records_path)
        original = comparison0[role]
        if (
            not isinstance(original, dict)
            or original.get("artifact") != EXPECTED_MODEL_ARTIFACTS[role]
            or not isinstance(original.get("runtime"), dict)
            or original["runtime"].get("provider") != "hf_transformers"
            or set(original["runtime"]) != {"provider", "settings"}
        ):
            raise BridgeError(f"{role} is not the canonical pinned Qwen3 model")
        settings = original["runtime"]["settings"]
        if not isinstance(settings, dict):
            raise BridgeError(f"{role} runtime settings are not canonical")
        expected_settings = {
            "batch_size": BATCH_SIZE,
            "checkpoint_tree_sha256": EXPECTED_MODEL_TREE_DIGESTS[role],
            "context_length": profile.context_length,
            "max_output_tokens": MAX_GENERATION_TOKENS,
            "offline": True,
            "seed": SEED,
            "timeout_seconds": PER_RECORD_TIMEOUT_SECONDS,
            "tokenizer_metadata_sha256": EXPECTED_TOKENIZER_DIGESTS[role],
        }
        if set(settings) != set(expected_settings) or any(
            settings.get(key) != value for key, value in expected_settings.items()
        ):
            raise BridgeError(f"{role} runtime settings are not canonical")
        checkpoint = prepared / f"evaluation/models/{role}"
        if (checkpoint / "generation_config.json").exists() or (
            checkpoint / "generation_config.json"
        ).is_symlink():
            raise BridgeError(
                f"{role} snapshot does not leave generation defaults to the task"
            )
        try:
            from transformers import AutoTokenizer

            from invarlock.runtime_providers.hf_transformers import (
                hf_tokenizer_contract_sha256,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True, trust_remote_code=False
            )
            observed_tokenizer_digest = hf_tokenizer_contract_sha256(tokenizer)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise BridgeError(
                f"{role} tokenizer identity could not be authenticated"
            ) from exc
        if settings["tokenizer_metadata_sha256"] != observed_tokenizer_digest:
            raise BridgeError(
                f"{role} tokenizer identity does not match the checkpoint"
            )
        identity = provider.authenticate_artifact(
            ModelRuntimeSpec(
                "hf_transformers", original["artifact"]["model_id"], settings
            ),
            checkpoint,
        )
        if runs[role][0]["model_tree_sha256"] != settings.get("checkpoint_tree_sha256"):
            raise BridgeError(f"{role} run used a different authenticated checkpoint")
        execution = runs[role][0]["execution_config"]
        if (
            settings.get("seed") != execution["seed"]
            or settings.get("batch_size") != execution["batch_size"]
            or settings.get("max_output_tokens") != execution["max_generation_tokens"]
        ):
            raise BridgeError(
                f"{role} runtime settings do not match evaluator execution"
            )
        side = write_runtime_import_side(
            root / f"imports/{role}",
            role=cast(Literal["baseline", "subject"], role),
            schedule=schedule,
            policy_digest=sha256_digest(policy),
            artifact_identity=identity,
            records=load_external_scoring_records_jsonl(
                records_path, schedule=schedule
            ),
            plugin=RuntimeProviderPluginIdentity(
                "hf_transformers", "invarlock", INVARLOCK_VERSION
            ),
            backend=RuntimeBackendIdentity(
                f"{selected}-hf",
                EVALUATORS[selected]["version"],
                digest(provenance),
                None,
                runs[role][0]["task_config_sha256"],
            ),
            capabilities=provider.capabilities(),
            execution_settings=RuntimeExecutionSettings(
                settings["seed"],
                settings["context_length"],
                settings["batch_size"],
                settings["max_output_tokens"],
                settings["timeout_seconds"],
                False,
            ),
            device=RuntimeDeviceFacts(
                str(execution["device"]), f"container-{execution['device']}"
            ),
            runtime_image_ref=image,
            runtime_image_digest=image,
            generated_at_utc=datetime.now(tz=UTC).isoformat(),
        )
        sides[role] = side
        anchors[role] = sha256_digest(side.provider_evidence.artifact_identity_bytes)
    write_runtime_import_paired_records(
        root / "imports/paired-records.json",
        schedule=schedule,
        metric="exact_match",
        baseline=sides["baseline"],
        subject=sides["subject"],
    )

    def side_descriptor(role: str) -> dict[str, Any]:
        original = comparison0[role]
        return {
            "artifact": {
                key: original["artifact"][key] for key in ("model_id", "locator")
            },
            "runtime": original["runtime"],
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side_descriptor("baseline"),
            "subject": side_descriptor("subject"),
            "dataset": "inputs/schedule.json",
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {
            "mode": "import",
            "records": "imports/paired-records.json",
            "schedule": "inputs/schedule.json",
            "baseline": imported("baseline"),
            "subject": imported("subject"),
        },
        "observations": [
            {
                "id": f"{selected}-provenance",
                "kind": "evaluator_provenance",
                "scope": "comparison",
                "path": "inputs/evaluator-provenance.json",
            }
        ],
        "output": {"evidence": "evidence"},
    }
    request_path = root / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False))
    evidence_fingerprint = public_key_fingerprint(evidence_key.public_key())
    trust = create_trust_material(
        transaction_root=root,
        evidence_key=evidence_signing_key,
        verifier_key_bytes=read_external_file(
            verifier_signing_key, label="verifier signing key"
        ),
        evidence_fingerprint=evidence_fingerprint,
        verifier_fingerprint=public_key_fingerprint(verifier_key.public_key()),
        trust_root=trust_root,
        policy_bytes=policy,
        verifier_identity=f"invarlock-example/{selected}-verifier",
        anchors={
            "baseline_artifact_digest": anchors["baseline"],
            "subject_artifact_digest": anchors["subject"],
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": image,
            "subject_runtime_digest": image,
            "evidence_signer_fingerprint": evidence_fingerprint,
        },
    )
    trust_path = trust.trusted_inputs
    evidence, receipt, report = (
        root / "evidence",
        root / "verifier/verification.receipt.json",
        root / "comparison-report.html",
    )
    commands = [
        [
            "evaluate",
            str(request_path),
            "--signing-key",
            str(evidence_signing_key),
            "--json",
        ],
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(trust_path),
            "--receipt",
            str(receipt),
            "--json",
        ],
        ["report", str(evidence), "--html", str(report)],
    ]
    for arguments in commands:
        _run_local_cli([sys.executable, "-m", "invarlock", *arguments])
    validate_completed_outputs(evidence, receipt, report)
    return evidence, receipt, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--role", choices=("baseline", "subject"), required=True)
    worker_parser.add_argument("--model", type=Path, required=True)
    worker_parser.add_argument("--dataset", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    bridge_parser = subparsers.add_parser("complete")
    bridge_parser.add_argument("--workspace", type=Path, required=True)
    bridge_parser.add_argument("--prepared", type=Path, required=True)
    bridge_parser.add_argument("--runtime-image", required=True)
    bridge_parser.add_argument("--evaluator", choices=tuple(EVALUATORS), required=True)
    bridge_parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    bridge_parser.add_argument("--evidence-signing-key", type=Path, required=True)
    bridge_parser.add_argument("--verifier-signing-key", type=Path, required=True)
    bridge_parser.add_argument("--trust-root", type=Path, required=True)
    bridge_parser.add_argument("--builder-public-key", type=Path, required=True)
    bridge_parser.add_argument("--source-commit", required=True)
    bridge_parser.add_argument("--base-image-id", required=True)
    bridge_parser.add_argument("--build-attestation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            worker(args.role, args.model, args.dataset, args.output)
        else:
            evidence, receipt, report = complete(
                Path(os.path.abspath(args.workspace.expanduser())),
                Path(os.path.abspath(args.prepared.expanduser())),
                args.runtime_image,
                args.evaluator,
                container_engine=args.container_engine,
                evidence_signing_key=Path(
                    os.path.abspath(args.evidence_signing_key.expanduser())
                ),
                verifier_signing_key=Path(
                    os.path.abspath(args.verifier_signing_key.expanduser())
                ),
                trust_root=Path(os.path.abspath(args.trust_root.expanduser())),
                source_commit=args.source_commit,
                base_image_id=args.base_image_id,
                builder_public_key=Path(
                    os.path.abspath(args.builder_public_key.expanduser())
                ),
                build_attestation=(
                    Path(os.path.abspath(args.build_attestation.expanduser()))
                    if args.build_attestation is not None
                    else None
                ),
            )
            print(f"Evidence: {evidence}\nReceipt: {receipt}\nReport: {report}")
    except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
