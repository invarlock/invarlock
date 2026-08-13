#!/usr/bin/env python3
"""Run LM Evaluation Harness and import its per-record output into InvarLock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]

try:
    from examples.integrations.trust_material import (
        create_trust_material,
        read_external_file,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    script_directory = Path(__file__).resolve().parent
    integration_directory = script_directory.parent
    for candidate in (script_directory, integration_directory):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
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
from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes
from invarlock.runtime_import_authoring import (
    load_external_scoring_records_jsonl,
    write_runtime_import_paired_records,
    write_runtime_import_side,
)
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

try:
    from examples.integrations.evaluator_transaction.worker import (
        run_evaluator_worker,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    try:
        from evaluator_transaction.worker import run_evaluator_worker
    except (
        ModuleNotFoundError
    ) as nested_exc:  # pragma: no cover - flat-script compatibility
        if nested_exc.name not in {
            "evaluator_transaction",
            "evaluator_transaction.worker",
        }:
            raise
        from evaluator_transaction_worker import (  # type: ignore[no-redef]
            run_evaluator_worker,
        )
try:
    from examples.integrations.evaluator_transaction.corpora import (
        CorpusProfile,
        corpus_profile,
        corpus_provenance,
        profile_for_dataset,
        profile_for_descriptor,
    )
    from examples.integrations.evaluator_transaction.model_profiles import (
        ModelProfile,
        model_profile,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from evaluator_transaction.corpora import (  # type: ignore[no-redef]
        CorpusProfile,
        corpus_profile,
        corpus_provenance,
        profile_for_dataset,
        profile_for_descriptor,
    )
    from evaluator_transaction.model_profiles import (  # type: ignore[no-redef]
        ModelProfile,
        model_profile,
    )

VERSION = "0.4.12+invarlock.nocache.1"
MAX_GENERATION_TOKENS = 1
HARNESS_SEED = 20_260_716
QUICK_CORPUS = corpus_profile("quick")
QUICK_MODELS = model_profile("quick")
HARNESS_BATCH_SIZE = QUICK_MODELS.batch_size
RECORD_COUNT = QUICK_CORPUS.record_count
MAX_WORKER_ARTIFACT_BYTES = 64 * 1024 * 1024
PER_RECORD_TIMEOUT_SECONDS = 300
WORKER_TIMEOUT_SECONDS = min(
    PER_RECORD_TIMEOUT_SECONDS * (RECORD_COUNT + 2), 24 * 60 * 60
)
CLI_TIMEOUT_SECONDS = 10 * 60
MINIMUM_SIDE_ACCURACY = QUICK_CORPUS.minimum_side_accuracy
TASK = "invarlock_exact_match"
DATASET_NAME = QUICK_CORPUS.dataset_name
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
DATASET_SHA256 = QUICK_CORPUS.dataset_sha256


def worker_timeout_seconds(profile: CorpusProfile) -> int:
    return min(PER_RECORD_TIMEOUT_SECONDS * (profile.record_count + 2), 24 * 60 * 60)


HARNESS_LOCK_PATH = Path("requirements/workflows/lm-evaluation-harness-py312.txt")
HARNESS_CUDA_LOCK_PATH = Path(
    "requirements/workflows/lm-evaluation-harness-py312-cu129.txt"
)


def _model_artifacts(profile: ModelProfile) -> dict[str, dict[str, str]]:
    return {
        snapshot.role: {
            "path": f"models/{snapshot.role}",
            "model_id": snapshot.repository,
            "locator": snapshot.locator,
        }
        for snapshot in profile.snapshots
    }


EXPECTED_MODEL_ARTIFACTS = _model_artifacts(QUICK_MODELS)
RUN_FIELDS = {
    "format",
    "role",
    "harness_version",
    "task_config",
    "task_config_sha256",
    "execution_config",
    "execution_config_sha256",
    "samples",
    "samples_sha256",
    "model_tree_sha256",
    "dataset_sha256",
    "runtime_image_digest",
    "record_count",
    "stable_id_field",
}
SAMPLE_FIELDS = {
    "doc",
    "target",
    "arguments",
    "filtered_resps",
    "filter",
    "doc_hash",
    "prompt_hash",
    "target_hash",
}


class BridgeError(ValueError):
    """The harness output cannot support verifier replay."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_bounded_command(
    command: list[str], *, timeout_seconds: int, label: str
) -> subprocess.CompletedProcess[str]:
    """Run one evaluator-side command with bounded diagnostics and a deadline."""
    try:
        completed = run_bounded_command(
            command,
            capture_output=True,
            check=False,
            timeout_seconds=timeout_seconds,
            stdout_limit=4 * 1024 * 1024,
            stderr_limit=256 * 1024,
            label=label,
        )
    except RuntimeError as exc:
        raise BridgeError(str(exc)) from exc
    return subprocess.CompletedProcess(
        command,
        completed.returncode,
        completed.stdout or "",
        completed.stderr or "",
    )


def _read_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_WORKER_ARTIFACT_BYTES,
) -> bytes:
    try:
        return read_regular_file_bytes(path, label=label, max_bytes=max_bytes)
    except StrictJsonError as exc:
        raise BridgeError(str(exc)) from exc


def _runtime_image_from_environment() -> str:
    # Direct unit/demo worker calls may omit the container binding; completion
    # always supplies and verifies the real image digest before accepting runs.
    value = os.environ.get("INVARLOCK_RUNTIME_IMAGE_ID", "sha256:" + "0" * 64)
    if IMAGE_ID.fullmatch(value) is None:
        raise BridgeError("the worker must receive the inspected runtime image digest")
    return value


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _inspect_runtime_image(
    engine: str,
    image: str,
    source_commit: str,
    base_image_id: str,
    build_attestation: Path,
    builder_public_key: ed25519.Ed25519PublicKey,
    profile: CorpusProfile = QUICK_CORPUS,
) -> None:
    if engine not in {"docker", "podman"}:
        raise BridgeError("container engine must be docker or podman")
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise BridgeError("source commit must be a full lowercase Git commit")
    if IMAGE_ID.fullmatch(base_image_id) is None:
        raise BridgeError("base image identity must be an immutable image digest")
    lock_digest = (
        "sha256:"
        + hashlib.sha256(
            (
                REPOSITORY_ROOT
                / (
                    HARNESS_CUDA_LOCK_PATH
                    if model_profile(profile.key).device == "cuda"
                    else HARNESS_LOCK_PATH
                )
            ).read_bytes()
        ).hexdigest()
    )
    try:
        inspect_evaluator_image(
            engine=engine,
            image=image,
            repository=REPOSITORY_ROOT,
            attestation_path=build_attestation,
            evaluator="lm-evaluation-harness",
            evaluator_version=VERSION,
            lock_sha256=lock_digest,
            expected_entrypoint=(
                "python",
                "/opt/invarlock/examples/lm-evaluation-harness-example.py",
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


def _external_ed25519_key(path: Path, *, label: str) -> ed25519.Ed25519PrivateKey:
    try:
        payload = read_external_file(path, label=label)
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
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


def mount_source(path: Path, *, label: str) -> str:
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


def _tokenizer_metadata_digest(checkpoint: Path) -> str:
    try:
        from transformers import AutoTokenizer

        from invarlock.runtime_providers.hf_transformers import (
            hf_tokenizer_contract_sha256,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        return hf_tokenizer_contract_sha256(tokenizer)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise BridgeError("tokenizer identity could not be authenticated") from exc


def _run_verified_worker(
    *,
    engine: str,
    image: str,
    role: str,
    prepared: Path,
    output: Path,
    profile: CorpusProfile | None = None,
    device: str = "cpu",
) -> None:
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
                "/opt/invarlock/examples/lm-evaluation-harness-example.py",
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
                **(
                    {"INVARLOCK_CORPUS_PROFILE": profile.key}
                    if profile is not None
                    else {}
                ),
            },
            timeout_seconds=worker_timeout_seconds(profile or corpus_profile("quick")),
            device=device,
        )
    except OciEvaluationError as exc:
        raise BridgeError(f"Harness worker control failed for {role}: {exc}") from exc
    if completed.returncode:
        raise BridgeError(
            completed.stderr or completed.stdout or "Harness worker failed"
        )


def task_config(dataset: str) -> dict[str, Any]:
    return {
        "task": TASK,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": dataset}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{expected}}",
        "generation_kwargs": {
            "do_sample": False,
            "max_gen_toks": MAX_GENERATION_TOKENS,
            "until": ["\n"],
        },
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
        "metadata": {"version": 1},
    }


def execution_config(profile: CorpusProfile | None = None) -> dict[str, Any]:
    """Return the complete fixed execution profile authenticated by the bridge."""

    selected_corpus = profile or corpus_profile(
        os.environ.get("INVARLOCK_CORPUS_PROFILE", "quick")
    )
    selected_models = model_profile(selected_corpus.key)
    return {
        "batch_size": selected_models.batch_size,
        "checkpoint_generation_config": "excluded",
        "device": selected_models.device,
        "dtype": selected_models.dtype,
        "harness_backend": "causal",
        "harness_model": "hf",
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "seed": HARNESS_SEED,
        "trust_remote_code": False,
    }


def worker(role: str, model: Path, dataset: Path, output: Path) -> None:
    """Run the real upstream CLI and retain its official samples JSONL."""

    if importlib.metadata.version("lm-eval") != VERSION:
        raise BridgeError(f"the runtime must contain lm-eval {VERSION}")
    runtime_image_digest = _runtime_image_from_environment()
    if (
        output.exists()
        or output.is_symlink()
        or not model.is_dir()
        or not dataset.is_file()
    ):
        raise BridgeError("worker inputs must exist and output must be new")
    generation_defaults = model / "generation_config.json"
    if generation_defaults.exists() or generation_defaults.is_symlink():
        raise BridgeError(
            "Harness model snapshot must leave generation defaults to the task"
        )
    output.mkdir(parents=True)
    model_tree_sha256 = checkpoint_tree_sha256(model)
    dataset_payload = _read_regular_file(dataset, label="Harness dataset")
    profile_key = os.environ.get("INVARLOCK_CORPUS_PROFILE")
    profile = corpus_profile(profile_key) if profile_key is not None else None
    if profile is not None:
        try:
            if profile_for_dataset(dataset_payload) != profile:
                raise BridgeError("worker corpus profile does not match the dataset")
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    dataset_sha256 = digest(dataset_payload)
    config = task_config(str(dataset))
    config_path = output / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_sha256 = digest(_read_regular_file(config_path, label="Harness task config"))
    raw = output / "upstream"
    execution = execution_config(profile)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        str(execution["harness_model"]),
        "--model_args",
        (
            f"pretrained={model},backend={execution['harness_backend']},"
            f"dtype={execution['dtype']},"
            f"trust_remote_code={execution['trust_remote_code']}"
        ),
        "--tasks",
        str(config_path),
        "--device",
        str(execution["device"]),
        "--batch_size",
        str(execution["batch_size"]),
        "--seed",
        str(execution["seed"]),
        "--log_samples",
        "--output_path",
        str(raw),
    ]
    completed = _run_bounded_command(
        command,
        timeout_seconds=worker_timeout_seconds(profile or corpus_profile("quick")),
        label="LM Evaluation Harness execution",
    )
    if completed.returncode:
        raise BridgeError("LM Evaluation Harness execution failed")
    if (
        checkpoint_tree_sha256(model) != model_tree_sha256
        or digest(_read_regular_file(dataset, label="Harness dataset"))
        != dataset_sha256
        or digest(_read_regular_file(config_path, label="Harness task config"))
        != config_sha256
    ):
        raise BridgeError("LM Evaluation Harness inputs changed during execution")
    samples = list(raw.rglob("samples_*.jsonl"))
    if len(samples) != 1:
        raise BridgeError("LM Evaluation Harness did not emit one per-record file")
    destination = output / "samples.jsonl"
    shutil.copy2(samples[0], destination)
    bound = config
    sample_bytes = _read_regular_file(destination, label="Harness samples")
    lines = sample_bytes.splitlines()
    if profile is not None and len(lines) != profile.record_count:
        raise BridgeError("LM Evaluation Harness returned an incomplete corpus")
    manifest = {
        "format": "invarlock/lm-evaluation-harness-run-v1",
        "role": role,
        "harness_version": VERSION,
        "task_config": bound,
        "task_config_sha256": digest(canonical_json_bytes(bound)),
        "execution_config": execution,
        "execution_config_sha256": digest(canonical_json_bytes(execution)),
        "samples": destination.name,
        "samples_sha256": digest(sample_bytes),
        "model_tree_sha256": model_tree_sha256,
        "dataset_sha256": dataset_sha256,
        "runtime_image_digest": runtime_image_digest,
        "record_count": len(lines),
        "stable_id_field": "id",
    }
    (output / "run-manifest.json").write_bytes(canonical_json_bytes(manifest))


def load_run(
    path: Path,
    role: str,
    image: str | None = None,
    profile: CorpusProfile | None = None,
) -> tuple[dict[str, Any], Path]:
    try:
        run = json.loads(_read_regular_file(path, label=f"{role} run provenance"))
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{role} run provenance is missing") from exc
    legacy_fields = RUN_FIELDS - {"runtime_image_digest"}
    if not isinstance(run, dict) or set(run) not in (RUN_FIELDS, legacy_fields):
        raise BridgeError(f"{role} run provenance is incomplete")
    if (
        run["format"] != "invarlock/lm-evaluation-harness-run-v1"
        or run["role"] != role
        or run["harness_version"] != VERSION
        or run["stable_id_field"] != "id"
        or IMAGE_ID.fullmatch(run["model_tree_sha256"]) is None
        or IMAGE_ID.fullmatch(f"sha256:{run['dataset_sha256']}") is None
        or (image is not None and run.get("runtime_image_digest") != image)
        or run["task_config"] != task_config("/records.jsonl")
        or run["task_config_sha256"] != digest(canonical_json_bytes(run["task_config"]))
        or run["execution_config"]
        != execution_config(profile or corpus_profile("quick"))
        or run["execution_config_sha256"]
        != digest(canonical_json_bytes(run["execution_config"]))
        or (profile is not None and run["record_count"] != profile.record_count)
    ):
        raise BridgeError(f"{role} run provenance is invalid")
    samples = path.parent / run["samples"]
    if (
        run["samples"] != "samples.jsonl"
        or not samples.is_file()
        or samples.is_symlink()
        or digest(_read_regular_file(samples, label=f"{role} Harness samples"))
        != run["samples_sha256"]
        or len(
            _read_regular_file(samples, label=f"{role} Harness samples").splitlines()
        )
        != run["record_count"]
    ):
        raise BridgeError(f"{role} per-record output was tampered")
    return cast(dict[str, Any], run), samples


def load_upstream_samples(path: Path, *, role: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _read_regular_file(path, label=f"{role} Harness samples").splitlines(), 1
    ):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"{role} Harness sample {index} is not JSON") from exc
        if not isinstance(value, dict):
            raise BridgeError(f"{role} Harness sample {index} is not an object")
        values.append(value)
    return values


def adapt(samples: Path, schedule: Any, destination: Path) -> None:
    """Map upstream records to the strict ABI; never import aggregate scores."""

    lines = _read_regular_file(samples, label="Harness samples").splitlines()
    if len(lines) != len(schedule.records):
        raise BridgeError("one Harness sample is required for every schedule record")
    output: list[dict[str, object]] = []
    for index, (raw, expected) in enumerate(
        zip(lines, schedule.records, strict=True), 1
    ):
        sample = json.loads(raw)
        if not isinstance(sample, dict):
            raise BridgeError(f"sample {index} is not an object")
        if "results" in sample and not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError("aggregate-only Harness results are not accepted")
        if not SAMPLE_FIELDS.issubset(sample):
            raise BridgeError(f"sample {index} lacks per-record facts")
        doc = sample["doc"]
        if not isinstance(doc, dict) or doc.get("id") != expected.record_id:
            raise BridgeError(f"sample {index} lacks a stable, ordered ID")
        arguments = sample["arguments"]
        request = arguments.get("gen_args_0") if isinstance(arguments, dict) else None
        prompt = request.get("arg_0") if isinstance(request, dict) else None
        generation = request.get("arg_1") if isinstance(request, dict) else None
        part = expected.input_parts[0] if len(expected.input_parts) == 1 else None
        if not isinstance(prompt, str) or part is None or prompt != part.text:
            raise BridgeError(f"sample {index} prompt does not match the schedule")
        if generation != task_config("/records.jsonl")["generation_kwargs"]:
            raise BridgeError(
                f"sample {index} generation settings do not match the task"
            )
        target = str(sample["target"])
        doc_bytes = json.dumps(doc, indent=2, default=str, ensure_ascii=False).encode()
        if (
            sample["filter"] != "none"
            or target != expected.expected_output
            or sample["doc_hash"] != digest(doc_bytes)
            or sample["prompt_hash"] != digest(prompt.encode())
            or sample["target_hash"] != digest(target.encode())
        ):
            raise BridgeError(f"sample {index} authenticated inputs were tampered")
        responses = sample["filtered_resps"]
        if (
            not isinstance(responses, list)
            or len(responses) != 1
            or not isinstance(responses[0], str)
        ):
            raise BridgeError(f"sample {index} lacks one model response")
        response = responses[0]
        output.append(
            {
                "record_id": expected.record_id,
                "input_sha256": expected.input_sha256,
                "status": "ok",
                "output_text": response,
                "output_sha256": digest(response.encode()),
            }
        )
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in output))
    load_external_scoring_records_jsonl(destination, schedule=schedule)


def imported(role: str) -> dict[str, str]:
    root = f"imports/{role}"
    names = {
        "identity": "model-artifact.identity.json",
        "receipt": "runtime-provider.receipt.json",
        "observation": "runtime-scoring.observation.json",
        "run_report": "report.json",
        "runtime_manifest": "runtime.manifest.json",
        "runtime_config": "run.yaml",
    }
    return {key: f"{root}/{name}" for key, name in names.items()}


def validate_completed_outputs(evidence: Path, receipt: Path, report: Path) -> None:
    """Require a passing signed transaction, not merely successful processes."""

    try:
        evaluation_report = json.loads(
            _read_regular_file(
                evidence / "reports/evaluation.report.json",
                label="evaluation report",
            )
        )
        verification_receipt = json.loads(
            _read_regular_file(receipt, label="verification receipt")
        )
    except (BridgeError, OSError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "the completed transaction is missing verified outputs"
        ) from exc
    if not isinstance(evaluation_report, dict) or not isinstance(
        verification_receipt, dict
    ):
        raise BridgeError("the completed transaction returned invalid outputs")
    statement = verification_receipt.get("statement")
    receipt_verdict = statement.get("verdict") if isinstance(statement, dict) else None
    comparison = evaluation_report.get("comparison")
    baseline = evaluation_report.get("baseline")
    subject = evaluation_report.get("subject")
    if (
        evaluation_report.get("verdict") != "pass"
        or evaluation_report.get("metric") != "exact_match"
        or not isinstance(comparison, dict)
        or isinstance(comparison.get("value"), bool)
        or not isinstance(comparison.get("value"), (int, float))
        or not isinstance(baseline, dict)
        or not isinstance(subject, dict)
        or isinstance(baseline.get("mean_score"), bool)
        or not isinstance(baseline.get("mean_score"), (int, float))
        or isinstance(subject.get("mean_score"), bool)
        or not isinstance(subject.get("mean_score"), (int, float))
        or not isinstance(receipt_verdict, dict)
        or receipt_verdict.get("ok") is not True
        or receipt_verdict.get("integrity_ok") is not True
        or receipt_verdict.get("policy_verdict") != "pass"
        or not report.is_file()
    ):
        raise BridgeError("the completed transaction did not verify a passing result")


def _validated_comparison(request: object) -> tuple[dict[str, Any], CorpusProfile]:
    comparison = request.get("comparison") if isinstance(request, dict) else None
    if not isinstance(comparison, dict) or comparison.get("metric") != "exact_match":
        raise BridgeError("prepared request is not the fixed exact-match transaction")
    if comparison.get("policy") != "inputs/acceptance.json":
        raise BridgeError("prepared request has an unexpected policy path")
    try:
        profile = profile_for_descriptor(comparison.get("dataset"))
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    selected_models = model_profile(profile.key)
    expected_artifacts = _model_artifacts(selected_models)
    expected_settings = {
        "batch_size": selected_models.batch_size,
        "context_length": profile.context_length,
        "max_output_tokens": MAX_GENERATION_TOKENS,
        "offline": True,
        "seed": HARNESS_SEED,
        "timeout_seconds": PER_RECORD_TIMEOUT_SECONDS,
    }
    for role in ("baseline", "subject"):
        side = comparison.get(role)
        if (
            not isinstance(side, dict)
            or set(side) != {"artifact", "runtime"}
            or side.get("artifact") != expected_artifacts[role]
            or not isinstance(side.get("runtime"), dict)
            or set(side["runtime"]) != {"provider", "settings"}
            or side["runtime"].get("provider") != "hf_transformers"
            or not isinstance(side["runtime"].get("settings"), dict)
        ):
            raise BridgeError(f"{role} is not the canonical pinned Qwen model")
        settings = side["runtime"]["settings"]
        checkpoint_digest = settings.get("checkpoint_tree_sha256")
        tokenizer_digest = settings.get("tokenizer_metadata_sha256")
        if (
            set(settings)
            != set(expected_settings)
            | {"checkpoint_tree_sha256", "tokenizer_metadata_sha256"}
            or any(
                settings.get(key) != value for key, value in expected_settings.items()
            )
            or not isinstance(checkpoint_digest, str)
            or IMAGE_ID.fullmatch(checkpoint_digest) is None
            or not isinstance(tokenizer_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", tokenizer_digest) is None
        ):
            raise BridgeError(f"{role} runtime settings are not canonical")
    return cast(dict[str, Any], comparison), profile


def _validate_workspace_roots(root: Path, prepared: Path) -> None:
    """Keep caller-selected transaction paths lexical and free of symlinks."""

    if root.exists() or root.is_symlink():
        raise BridgeError("transaction workspace must be new")
    if not prepared.is_dir() or prepared.is_symlink():
        raise BridgeError("prepared workspace must be a real directory")


def _authenticated_prepared_corpus(
    prepared: Path,
    comparison: dict[str, Any],
    profile: CorpusProfile,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    dataset = comparison.get("dataset")
    if not isinstance(dataset, dict):
        raise BridgeError("prepared request lacks the authenticated dataset")
    raw_dataset = _read_regular_file(
        prepared / "evaluation/inputs/records.jsonl", label="prepared dataset"
    )
    try:
        dataset_profile = profile_for_dataset(raw_dataset)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    if dataset_profile != profile or dataset["sha256"] != digest(raw_dataset):
        raise BridgeError("prepared dataset does not match the request digest")
    try:
        provenance = json.loads(
            _read_regular_file(
                prepared / "evaluation/inputs/corpus-profile.json",
                label="prepared corpus profile",
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError("prepared corpus profile is invalid") from exc
    if provenance != corpus_provenance(profile):
        raise BridgeError("prepared corpus provenance does not match the dataset")
    return cast(dict[str, Any], dataset), raw_dataset, cast(dict[str, Any], provenance)


def complete(
    root: Path,
    prepared: Path,
    image: str,
    *,
    container_engine: str = "docker",
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    source_commit: str | None = None,
    base_image_id: str | None = None,
    build_attestation: Path | None = None,
    builder_public_key: Path | None = None,
    device: str | None = None,
) -> tuple[Path, Path, Path]:
    """Author strict import inputs and execute evaluate, verify, and report."""

    if IMAGE_ID.fullmatch(image) is None:
        raise BridgeError("runtime image must be an immutable local sha256 ID")
    _validate_workspace_roots(root, prepared)
    if (
        evidence_signing_key is None
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
    for key_path, label in (
        (evidence_signing_key, "evidence signing key"),
        (verifier_signing_key, "verifier signing key"),
        (builder_public_key, "builder public key"),
    ):
        try:
            key_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise BridgeError(f"{label} must remain outside the transaction")
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
    request0 = yaml.safe_load(
        _read_regular_file(
            prepared / "evaluation/request.yaml", label="prepared request"
        )
    )
    comparison0, profile = _validated_comparison(request0)
    selected_models = model_profile(profile.key)
    device_selector = device or selected_models.device
    if selected_models.device == "cpu" and device_selector != "cpu":
        raise BridgeError("the quick Harness profile requires a CPU worker")
    if selected_models.device == "cuda" and not device_selector.startswith("cuda"):
        raise BridgeError("the flagship Harness profile requires a CUDA worker")
    _inspect_runtime_image(
        container_engine,
        image,
        source_commit,
        base_image_id,
        build_attestation,
        builder_public_key=builder_key,
        profile=profile,
    )
    dataset0 = prepared / "evaluation/inputs/records.jsonl"
    dataset, raw_dataset, prepared_corpus_provenance = _authenticated_prepared_corpus(
        prepared, comparison0, profile
    )
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
    if any(not record.expected_output for record in schedule.records):
        raise BridgeError("prepared exact-match targets must be non-empty")
    prepared_policy = prepared / "evaluation/inputs/acceptance.json"
    policy = _read_regular_file(prepared_policy, label="prepared policy")
    expected_policy = profile.acceptance_policy()
    if json.loads(policy) != expected_policy:
        raise BridgeError("prepared exact-match policy is not the fixed example policy")
    for role in ("baseline", "subject"):
        output = root / f"upstream/{role}/result"
        _run_verified_worker(
            engine=container_engine,
            image=image,
            role=role,
            prepared=prepared,
            output=output,
            profile=profile,
            device=device_selector,
        )
    runs = {
        role: load_run(
            root / f"upstream/{role}/result/run-manifest.json",
            role,
            image,
            profile,
        )
        for role in ("baseline", "subject")
    }
    if (
        runs["baseline"][0]["task_config_sha256"]
        != runs["subject"][0]["task_config_sha256"]
        or runs["baseline"][0]["execution_config_sha256"]
        != runs["subject"][0]["execution_config_sha256"]
        or any(
            run["dataset_sha256"] != dataset["sha256"]
            for run, _samples in runs.values()
        )
    ):
        raise BridgeError(
            "fresh worker runs used different Harness configurations or dataset"
        )
    (root / "inputs/acceptance.json").write_bytes(policy)
    provenance = canonical_json_bytes(
        {
            "format": "invarlock/lm-evaluation-harness-provenance-v2",
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
                    "samples": load_upstream_samples(runs[role][1], role=role),
                }
                for role in ("baseline", "subject")
            },
        }
    )
    (root / "inputs/harness-provenance.json").write_bytes(provenance)
    provider = HFTransformersProvider()
    sides: dict[str, Any] = {}
    anchors: dict[str, str] = {}
    for role in ("baseline", "subject"):
        records_path = root / f"imports/{role}-records.jsonl"
        adapt(runs[role][1], schedule, records_path)
        original = comparison0[role]
        settings = original["runtime"]["settings"]
        spec = ModelRuntimeSpec(
            "hf_transformers", original["artifact"]["model_id"], settings
        )
        checkpoint = prepared / f"evaluation/models/{role}"
        if (checkpoint / "generation_config.json").exists() or (
            checkpoint / "generation_config.json"
        ).is_symlink():
            raise BridgeError(
                f"{role} snapshot does not leave generation defaults to the task"
            )
        try:
            observed_checkpoint_digest = checkpoint_tree_sha256(checkpoint)
        except (OSError, RuntimeError, ValueError) as exc:
            raise BridgeError(f"{role} checkpoint could not be authenticated") from exc
        if observed_checkpoint_digest != settings["checkpoint_tree_sha256"]:
            raise BridgeError(f"{role} checkpoint tree digest does not match")
        identity = provider.authenticate_artifact(spec, checkpoint)
        if runs[role][0]["model_tree_sha256"] != settings.get("checkpoint_tree_sha256"):
            raise BridgeError(f"{role} run used a different authenticated checkpoint")
        observed_tokenizer_digest = _tokenizer_metadata_digest(checkpoint)
        if settings["tokenizer_metadata_sha256"] != observed_tokenizer_digest:
            raise BridgeError(
                f"{role} tokenizer identity does not match the checkpoint"
            )
        snapshot = selected_models.snapshot(role)
        if (
            snapshot.checkpoint_tree_sha256 is not None
            and observed_checkpoint_digest != snapshot.checkpoint_tree_sha256
        ):
            raise BridgeError(f"{role} checkpoint tree does not match its pin")
        if (
            snapshot.tokenizer_contract_sha256 is not None
            and observed_tokenizer_digest != snapshot.tokenizer_contract_sha256
        ):
            raise BridgeError(f"{role} tokenizer contract does not match its pin")
        execution = runs[role][0]["execution_config"]
        if (
            settings.get("seed") != execution["seed"]
            or settings.get("batch_size") != execution["batch_size"]
            or settings.get("max_output_tokens") != execution["max_generation_tokens"]
        ):
            raise BridgeError(f"{role} runtime settings do not match Harness execution")
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
                "lm-evaluation-harness-hf",
                VERSION,
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

    def side(role: str) -> dict[str, Any]:
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
            "baseline": side("baseline"),
            "subject": side("subject"),
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
                "id": "lm-evaluation-harness-provenance",
                "kind": "harness_provenance",
                "scope": "comparison",
                "path": "inputs/harness-provenance.json",
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
        verifier_identity="invarlock-example/lm-evaluation-harness-verifier",
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
        completed = _run_bounded_command(
            [sys.executable, "-m", "invarlock", *arguments],
            timeout_seconds=CLI_TIMEOUT_SECONDS,
            label="InvarLock transaction command",
        )
        if completed.returncode:
            raise BridgeError(completed.stderr or completed.stdout)
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
    bridge_parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            worker(args.role, args.model, args.dataset, args.output)
        else:
            evidence, receipt, report = complete(
                Path(os.path.abspath(args.workspace.expanduser())),
                Path(os.path.abspath(args.prepared.expanduser())),
                args.runtime_image,
                container_engine=args.container_engine,
                evidence_signing_key=(
                    Path(os.path.abspath(args.evidence_signing_key.expanduser()))
                    if args.evidence_signing_key is not None
                    else None
                ),
                verifier_signing_key=(
                    Path(os.path.abspath(args.verifier_signing_key.expanduser()))
                    if args.verifier_signing_key is not None
                    else None
                ),
                trust_root=(
                    Path(os.path.abspath(args.trust_root.expanduser()))
                    if args.trust_root is not None
                    else None
                ),
                source_commit=args.source_commit,
                base_image_id=args.base_image_id,
                builder_public_key=(
                    Path(os.path.abspath(args.builder_public_key.expanduser()))
                    if args.builder_public_key is not None
                    else None
                ),
                build_attestation=(
                    Path(os.path.abspath(args.build_attestation.expanduser()))
                    if args.build_attestation is not None
                    else None
                ),
                device=args.device,
            )
            print(f"Evidence: {evidence}\nReceipt: {receipt}\nReport: {report}")
    except (BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
