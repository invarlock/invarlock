#!/usr/bin/env python3
"""Compare a pinned BF16 checkpoint with its source-derived Q5_K_M GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from examples.integrations import launch
from examples.integrations.bounded_command import run_bounded_command
from examples.integrations.evaluator_transaction.corpora import records_jsonl
from examples.integrations.evaluator_transaction.image_cleanup import (
    OwnedImageTag,
    record_owned_image_tag,
    remove_owned_image_tags,
    temporary_image_tag,
)
from examples.integrations.evaluator_transaction.model_profiles import Snapshot
from examples.integrations.gguf_deployment_profiles import (
    DEFAULT_DEPLOYMENT_PROFILE,
    DeploymentProfile,
    deployment_profile,
    deployment_profile_keys,
    deployment_records,
)
from examples.integrations.gguf_llama_cpp import (
    PendingTrust,
    _build_runtime_image,
    _container_command,
    _inspect_image_id,
    _inspect_spec,
    _materialize_trust,
    _mount_source,
    _sha256_file,
    _stage_backend,
)
from examples.integrations.run import ExamplePaths, _paths, _write_private_key
from examples.integrations.trust_material import (
    load_external_key,
    validate_new_trust_root,
)
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import ModelRuntimeSpec, artifact_identity_sha256
from invarlock.core.runtime_provider.request_bindings import (
    LLAMA_CPP_REQUEST_SETTINGS,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_support import EvidencePackStatus
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
    load_hf_text_tokenizer,
)

_TRANSFORMATION_FORMAT = "invarlock/example-gguf-deployment-transformation-v1"
_LLAMA_SOURCE_COMMIT = "12127defda4f41b7679cb2477a4b0d65ee6a0c8f"
_LLAMA_SOURCE_SHA256 = (
    "5ab75e394f4c71425ecce64a213dab3b8e3e9cfe0f19d0dcda4d5a4f7733da83"
)
_SEED = 20_260_716
_BASELINE_TIMEOUT_SECONDS = 300
_SUBJECT_TIMEOUT_SECONDS = 600
_WORKER_CPUS = "16"
_SUBJECT_CPU_THREADS = 16
_SUBJECT_PROMPT_BATCH_SIZE = 512
_SUBJECT_PROMPT_MICROBATCH_SIZE = 512
_VERIFIER_IDENTITY = "invarlock-example/gguf-deployment-verifier"


@dataclass(frozen=True, slots=True)
class ConversionResult:
    subject: Path
    intermediate_sha256: str
    intermediate_byte_length: int
    subject_sha256: str
    subject_byte_length: int


def _conversion_command(
    profile: DeploymentProfile,
    engine: str,
    image_id: str,
    *,
    source_checkpoint: Path,
    source_archive: Path,
    output_root: Path,
) -> list[str]:
    code = f"""\
import os
import runpy
import sys
import tarfile

with tarfile.open('/inputs/llama.cpp.tar.gz', mode='r:gz') as archive:
    archive.extractall('/tmp', filter='data')
source = '/tmp/llama.cpp-b10015'
os.chdir(source)
sys.path.insert(0, source + '/gguf-py')
sys.argv = [
    'convert_hf_to_gguf.py',
    '/inputs/model',
    '--outfile',
    '/output/{profile.intermediate_name}',
    '--outtype',
    'bf16',
]
runpy.run_path(source + '/convert_hf_to_gguf.py', run_name='__main__')
"""
    return [
        *_container_command(
            engine,
            image_id,
            user="65532:65532",
            entrypoint="python",
            mounts=(
                f"type=bind,src={_mount_source(source_checkpoint)},dst=/inputs/model,readonly",
                f"type=bind,src={_mount_source(source_archive)},dst=/inputs/llama.cpp.tar.gz,readonly",
                f"type=bind,src={_mount_source(output_root)},dst=/output",
            ),
            environment=("HOME=/tmp", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"),
        ),
        "-c",
        code,
    ]


def _quantization_command(
    profile: DeploymentProfile,
    engine: str,
    image_id: str,
    *,
    intermediate: Path,
    output_root: Path,
) -> list[str]:
    return [
        *_container_command(
            engine,
            image_id,
            user="65532:65532",
            entrypoint="/opt/llama.cpp/llama-quantize",
            mounts=(
                f"type=bind,src={_mount_source(intermediate)},dst=/inputs/source.gguf,readonly",
                f"type=bind,src={_mount_source(output_root)},dst=/output",
            ),
        ),
        "/inputs/source.gguf",
        f"/output/{profile.subject_name}",
        profile.quantization,
    ]


def _new_nonempty_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"pinned llama.cpp did not create the {label}")


def _convert_and_quantize(
    repository: Path,
    *,
    profile: DeploymentProfile,
    source_checkpoint: Path,
    source_archive: Path,
    output_root: Path,
    container_engine: str,
    conversion_image_id: str,
    gguf_image_id: str,
) -> ConversionResult:
    intermediate = output_root / profile.intermediate_name
    subject = output_root / profile.subject_name
    for path in (intermediate, subject):
        if path.exists() or path.is_symlink():
            raise RuntimeError("GGUF conversion destinations must be new")
    if _sha256_file(source_archive) != _LLAMA_SOURCE_SHA256:
        raise RuntimeError("GGUF conversion source archive is not pinned")
    original_mode = output_root.stat().st_mode & 0o777
    output_root.chmod(0o733)
    try:
        launch._run(
            _conversion_command(
                profile,
                container_engine,
                conversion_image_id,
                source_checkpoint=source_checkpoint,
                source_archive=source_archive,
                output_root=output_root,
            ),
            cwd=repository,
        )
        _new_nonempty_file(intermediate, label="BF16 GGUF")
        intermediate_sha256 = _sha256_file(intermediate)
        intermediate_byte_length = intermediate.stat().st_size
        launch._run(
            _quantization_command(
                profile,
                container_engine,
                gguf_image_id,
                intermediate=intermediate,
                output_root=output_root,
            ),
            cwd=repository,
        )
        _new_nonempty_file(subject, label="Q5_K_M GGUF")
        subject_sha256 = _sha256_file(subject)
        if subject_sha256 == intermediate_sha256:
            raise RuntimeError("quantization did not create a distinct GGUF artifact")
        subject.chmod(0o644)
        result = ConversionResult(
            subject=subject,
            intermediate_sha256=intermediate_sha256,
            intermediate_byte_length=intermediate_byte_length,
            subject_sha256=subject_sha256,
            subject_byte_length=subject.stat().st_size,
        )
    except Exception:
        subject.unlink(missing_ok=True)
        raise
    finally:
        intermediate.unlink(missing_ok=True)
        output_root.chmod(original_mode)
    return result


def _stage_source_checkpoint(destination_root: Path, snapshot: Snapshot) -> Path:
    staged_snapshot = replace(snapshot, role="source")
    staged = destination_root / staged_snapshot.role
    staged.mkdir(mode=0o755)
    try:
        for item in staged_snapshot.files:
            destination = staged / item.name
            partial = destination.with_suffix(destination.suffix + ".partial")
            digest = hashlib.sha256()
            length = 0
            request = urllib.request.Request(
                staged_snapshot.url(item.name),
                headers={"User-Agent": "invarlock-gguf-deployment-example/1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                    with partial.open("xb") as output:
                        while chunk := response.read(1024 * 1024):
                            if not isinstance(chunk, bytes):
                                raise RuntimeError(
                                    "snapshot download did not return bytes"
                                )
                            length += len(chunk)
                            if length > item.byte_length:
                                raise RuntimeError(
                                    "snapshot download exceeds its pinned size"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                if length != item.byte_length or digest.hexdigest() != item.sha256:
                    raise RuntimeError(
                        f"downloaded source/{item.name} is not byte-pinned"
                    )
                partial.chmod(0o644)
                partial.replace(destination)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        config = json.loads((staged / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != staged_snapshot.model_type:
            raise RuntimeError("staged source checkpoint architecture is not pinned")
        observed_tree = checkpoint_tree_sha256(staged)
        if observed_tree != snapshot.checkpoint_tree_sha256:
            raise RuntimeError("staged source checkpoint tree is not pinned")
    except Exception:
        shutil.rmtree(staged)
        raise
    return staged


def _baseline_spec(
    source_checkpoint: Path, profile: DeploymentProfile
) -> dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer = load_hf_text_tokenizer(
        AutoTokenizer.from_pretrained,
        source_checkpoint,
    )
    tokenizer_digest = hf_tokenizer_contract_sha256(tokenizer)
    if tokenizer_digest != profile.source.tokenizer_contract_sha256:
        raise RuntimeError("staged source tokenizer contract is not pinned")
    tree = checkpoint_tree_sha256(source_checkpoint)
    if tree != profile.source.checkpoint_tree_sha256:
        raise RuntimeError("staged source checkpoint tree is not pinned")
    for record in deployment_records(profile):
        target = tokenizer(record["expected"], add_special_tokens=False)["input_ids"]
        prompt = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
        if len(target) != 1 or len(prompt) + 1 > profile.corpus.context_length:
            raise RuntimeError("deployment corpus exceeds the pinned token bounds")
    return {
        "model_id": profile.source.repository,
        "settings": {
            "batch_size": 1,
            "checkpoint_tree_sha256": tree,
            "context_length": profile.corpus.context_length,
            "immutable_revision": profile.source.revision,
            "max_output_tokens": 1,
            "offline": True,
            "seed": _SEED,
            "timeout_seconds": _BASELINE_TIMEOUT_SECONDS,
            "tokenizer_metadata_sha256": tokenizer_digest,
        },
    }


def _transformation_document(
    *,
    profile: DeploymentProfile,
    conversion: ConversionResult,
    conversion_image_id: str,
    gguf_image_id: str,
) -> dict[str, object]:
    return {
        "format": _TRANSFORMATION_FORMAT,
        "source": {
            "repository": profile.source.repository,
            "revision": profile.source.revision,
            "checkpoint_tree_sha256": profile.source.checkpoint_tree_sha256,
            "tokenizer_contract_sha256": profile.source.tokenizer_contract_sha256,
        },
        "conversion": {
            "runtime_image_digest": conversion_image_id,
            "tool": {
                "name": "llama.cpp/convert_hf_to_gguf.py",
                "source_commit": _LLAMA_SOURCE_COMMIT,
                "source_sha256": _LLAMA_SOURCE_SHA256,
                "source_tag": "b10015",
            },
            "output": {
                "filename": profile.intermediate_name,
                "sha256": conversion.intermediate_sha256,
                "byte_length": conversion.intermediate_byte_length,
                "type": "BF16",
            },
        },
        "quantization": {
            "runtime_image_digest": gguf_image_id,
            "tool": "llama-quantize",
            "type": profile.quantization,
        },
        "subject": {
            "filename": conversion.subject.name,
            "sha256": conversion.subject_sha256,
            "byte_length": conversion.subject_byte_length,
        },
    }


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_transformation_binding(
    transformation: Mapping[str, object],
    *,
    profile: DeploymentProfile,
    subject: Path,
    subject_sha256: str,
    baseline_image_id: str,
    subject_image_id: str,
) -> None:
    if (
        set(transformation)
        != {
            "conversion",
            "format",
            "quantization",
            "source",
            "subject",
        }
        or transformation.get("format") != _TRANSFORMATION_FORMAT
    ):
        raise RuntimeError("deployment transformation document is invalid")
    if transformation.get("source") != {
        "repository": profile.source.repository,
        "revision": profile.source.revision,
        "checkpoint_tree_sha256": profile.source.checkpoint_tree_sha256,
        "tokenizer_contract_sha256": profile.source.tokenizer_contract_sha256,
    }:
        raise RuntimeError("deployment transformation source identity is invalid")

    conversion = transformation.get("conversion")
    if not isinstance(conversion, dict) or set(conversion) != {
        "output",
        "runtime_image_digest",
        "tool",
    }:
        raise RuntimeError("deployment transformation conversion is invalid")
    if conversion.get("runtime_image_digest") != baseline_image_id or conversion.get(
        "tool"
    ) != {
        "name": "llama.cpp/convert_hf_to_gguf.py",
        "source_commit": _LLAMA_SOURCE_COMMIT,
        "source_sha256": _LLAMA_SOURCE_SHA256,
        "source_tag": "b10015",
    }:
        raise RuntimeError("deployment transformation conversion identity is invalid")
    intermediate = conversion.get("output")
    if (
        not isinstance(intermediate, dict)
        or set(intermediate) != {"byte_length", "filename", "sha256", "type"}
        or intermediate.get("filename") != profile.intermediate_name
        or intermediate.get("type") != "BF16"
        or not _is_sha256_hex(intermediate.get("sha256"))
        or isinstance(intermediate.get("byte_length"), bool)
        or not isinstance(intermediate.get("byte_length"), int)
        or intermediate["byte_length"] <= 0
    ):
        raise RuntimeError("deployment transformation intermediate identity is invalid")

    if transformation.get("quantization") != {
        "runtime_image_digest": subject_image_id,
        "tool": "llama-quantize",
        "type": profile.quantization,
    }:
        raise RuntimeError("deployment transformation quantization identity is invalid")
    if transformation.get("subject") != {
        "filename": subject.name,
        "sha256": subject_sha256,
        "byte_length": subject.stat().st_size,
    }:
        raise RuntimeError("deployment transformation subject identity is invalid")


def _validate_runtime_spec_bindings(
    *,
    profile: DeploymentProfile,
    baseline_spec: Mapping[str, object],
    subject_spec: Mapping[str, object],
    subject_sha256: str,
    subject_byte_length: int,
) -> None:
    baseline_settings = baseline_spec.get("settings")
    expected_baseline_settings = {
        "batch_size": 1,
        "checkpoint_tree_sha256": profile.source.checkpoint_tree_sha256,
        "context_length": profile.corpus.context_length,
        "immutable_revision": profile.source.revision,
        "max_output_tokens": 1,
        "offline": True,
        "seed": _SEED,
        "timeout_seconds": _BASELINE_TIMEOUT_SECONDS,
        "tokenizer_metadata_sha256": profile.source.tokenizer_contract_sha256,
    }
    if baseline_spec.get("model_id") != profile.source.repository or (
        baseline_settings != expected_baseline_settings
    ):
        raise RuntimeError("deployment baseline runtime specification is not pinned")

    subject_settings = subject_spec.get("settings")
    if (
        subject_spec.get("model_id") != f"gguf-sha256-{subject_sha256}.gguf"
        or not isinstance(subject_settings, dict)
        or set(subject_settings) != LLAMA_CPP_REQUEST_SETTINGS
        or subject_settings.get("artifact_sha256") != subject_sha256
        or subject_settings.get("artifact_byte_length") != subject_byte_length
        or subject_settings.get("batch_size") != 1
        or subject_settings.get("context_length") != profile.corpus.context_length
        or subject_settings.get("cpu_threads") != _SUBJECT_CPU_THREADS
        or subject_settings.get("max_output_tokens") != 1
        or subject_settings.get("prompt_batch_size") != _SUBJECT_PROMPT_BATCH_SIZE
        or subject_settings.get("prompt_microbatch_size")
        != _SUBJECT_PROMPT_MICROBATCH_SIZE
        or subject_settings.get("seed") != _SEED
        or subject_settings.get("timeout_seconds") != _SUBJECT_TIMEOUT_SECONDS
    ):
        raise RuntimeError(
            "deployment subject runtime specification is not bound to the GGUF bytes"
        )


def _artifact_anchor(
    provider_name: str,
    model_id: str,
    settings: Mapping[str, Any],
    artifact_path: Path,
) -> str:
    spec = ModelRuntimeSpec(
        provider_name=provider_name,
        model_id=model_id,
        settings=settings,
    )
    if provider_name == "hf_transformers":
        return "sha256:" + artifact_identity_sha256(
            HFTransformersProvider().authenticate_artifact(spec, artifact_path)
        )
    if provider_name == "llama_cpp":
        from invarlock_addins.gguf.provider import LlamaCppProvider

        return "sha256:" + artifact_identity_sha256(
            LlamaCppProvider().authenticate_artifact(spec, artifact_path)
        )
    raise ValueError("deployment artifact provider is unsupported")


def _prepare_transaction(
    root: Path,
    *,
    profile: DeploymentProfile,
    runtime_root: Path,
    source_checkpoint: Path,
    subject: Path,
    baseline_spec: Mapping[str, object],
    subject_spec: Mapping[str, object],
    transformation: Mapping[str, object],
    baseline_image_id: str,
    subject_image_id: str,
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    ephemeral_trust_root: bool = True,
) -> tuple[ExamplePaths, PendingTrust]:
    _new_nonempty_file(subject, label="Q5_K_M GGUF")
    subject_sha256 = _sha256_file(subject)
    _validate_runtime_spec_bindings(
        profile=profile,
        baseline_spec=baseline_spec,
        subject_spec=subject_spec,
        subject_sha256=subject_sha256,
        subject_byte_length=subject.stat().st_size,
    )
    _validate_transformation_binding(
        transformation,
        profile=profile,
        subject=subject,
        subject_sha256=subject_sha256,
        baseline_image_id=baseline_image_id,
        subject_image_id=subject_image_id,
    )
    external_trust = any(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    )
    if external_trust and not all(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    ):
        raise ValueError(
            "evidence key, verifier key, and trust root must be supplied together"
        )
    if external_trust and ephemeral_trust_root:
        raise ValueError("external trust material cannot use ephemeral mode")
    if not external_trust and not ephemeral_trust_root:
        raise ValueError(
            "caller-owned evidence/verifier keys and trust root are required"
        )
    paths = _paths(
        root,
        evidence_key=(
            Path(os.path.abspath(evidence_signing_key.expanduser()))
            if external_trust and evidence_signing_key is not None
            else None
        ),
        trust_root=(
            Path(os.path.abspath(trust_root.expanduser()))
            if external_trust and trust_root is not None
            else None
        ),
    )
    inputs = paths.evaluation / "inputs"
    inputs.mkdir(parents=True)
    evidence_key_bytes: bytes | None = None
    verifier_key_bytes: bytes | None = None
    if external_trust:
        assert evidence_signing_key is not None
        assert verifier_signing_key is not None
        assert trust_root is not None
        trust_root = validate_new_trust_root(trust_root, transaction_root=root)
        evidence_key_path, evidence_key_bytes, evidence_signer = load_external_key(
            evidence_signing_key,
            transaction_root=root,
            label="evidence signing key",
        )
        _verifier_key_path, verifier_key_bytes, verifier = load_external_key(
            verifier_signing_key,
            transaction_root=root,
            label="verifier signing key",
        )
        if evidence_signer == verifier:
            raise ValueError("evidence and verifier signing keys must be distinct")
        paths = _paths(root, evidence_key=evidence_key_path, trust_root=trust_root)
    else:
        paths.independent_policy.parent.mkdir(parents=True)
        paths.evidence_key.parent.mkdir(parents=True)
        paths.verifier_key.parent.mkdir(parents=True)
    paths.receipt.parent.mkdir(parents=True, exist_ok=True)

    records = deployment_records(profile)
    dataset_bytes = records_jsonl(records, compact=True)
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_sha256 != profile.corpus.dataset_sha256:
        raise RuntimeError("deployment dataset does not match the pinned corpus")
    dataset = inputs / "records.jsonl"
    dataset.write_bytes(dataset_bytes)
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset,
            sha256=dataset_sha256,
            name=profile.corpus.dataset_name,
            split=profile.corpus.split,
            input_field="prompt",
            expected_output_field="expected",
            id_field="id",
        ),
        dataset_bytes,
    )
    policy_bytes = canonical_json_bytes(profile.corpus.acceptance_policy())
    (inputs / "acceptance.json").write_bytes(policy_bytes)
    if not external_trust:
        paths.independent_policy.write_bytes(policy_bytes)

    def request_side(
        *,
        provider: str,
        artifact: Path,
        spec: Mapping[str, object],
        locator: str,
    ) -> dict[str, object]:
        model_id = spec.get("model_id")
        settings = spec.get("settings")
        if not isinstance(model_id, str) or not isinstance(settings, dict):
            raise RuntimeError(f"{provider} inspection payload is invalid")
        return {
            "artifact": {
                "path": str(artifact.relative_to(paths.evaluation)),
                "model_id": model_id,
                "locator": locator,
            },
            "runtime": {"provider": provider, "settings": settings},
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": request_side(
                provider="hf_transformers",
                artifact=source_checkpoint,
                spec=baseline_spec,
                locator=profile.source.locator,
            ),
            "subject": request_side(
                provider="llama_cpp",
                artifact=subject,
                spec=subject_spec,
                locator=(
                    f"derived://{profile.source.repository}@{profile.source.revision}#"
                    f"llama.cpp-b10015-{profile.quantization.lower()}@sha256:"
                    f"{subject_sha256}"
                ),
            ),
            "dataset": profile.corpus.dataset_descriptor(),
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "observations": [
            {
                "id": profile.observation_id,
                "kind": "artifact_transformation",
                "scope": "subject",
                "path": "inputs/subject-transformation.json",
            }
        ],
        "output": {"evidence": "evidence"},
    }
    (inputs / "subject-transformation.json").write_bytes(
        canonical_json_bytes(dict(transformation))
    )
    paths.request.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    if external_trust:
        assert evidence_key_bytes is not None
        assert verifier_key_bytes is not None
    else:
        evidence_signer = _write_private_key(paths.evidence_key)
        verifier = _write_private_key(paths.verifier_key)
        paths.evidence_key.with_suffix(".fingerprint").write_text(
            evidence_signer + "\n", encoding="ascii"
        )
        paths.verifier_key.with_suffix(".fingerprint").write_text(
            verifier + "\n", encoding="ascii"
        )
    if not source_checkpoint.is_relative_to(runtime_root) or not subject.is_relative_to(
        runtime_root
    ):
        raise RuntimeError("deployment artifacts do not share one closed resource root")
    baseline_settings = baseline_spec.get("settings")
    baseline_model_id = baseline_spec.get("model_id")
    subject_settings = subject_spec.get("settings")
    subject_model_id = subject_spec.get("model_id")
    if (
        not isinstance(baseline_model_id, str)
        or not isinstance(subject_model_id, str)
        or not isinstance(baseline_settings, dict)
        or not isinstance(subject_settings, dict)
    ):
        raise RuntimeError("deployment runtime specifications are invalid")
    anchors = {
        "baseline_artifact_digest": _artifact_anchor(
            "hf_transformers",
            baseline_model_id,
            baseline_settings,
            source_checkpoint,
        ),
        "subject_artifact_digest": _artifact_anchor(
            "llama_cpp", subject_model_id, subject_settings, subject
        ),
        "schedule_digest": f"sha256:{schedule.schedule_sha256}",
        "baseline_runtime_digest": baseline_image_id,
        "subject_runtime_digest": subject_image_id,
        "evidence_signer_fingerprint": evidence_signer,
    }
    return paths, PendingTrust(
        anchors=anchors,
        policy_bytes=policy_bytes,
        external=external_trust,
        trust_root=trust_root,
        verifier_key_bytes=verifier_key_bytes,
        evidence_fingerprint=evidence_signer,
        verifier_fingerprint=verifier,
    )


def _execute(
    repository: Path,
    paths: ExamplePaths,
    *,
    profile: DeploymentProfile,
    runtime_root: Path,
    container_engine: str,
    baseline_image_id: str,
    subject_image_id: str,
    pending_trust: PendingTrust,
    allow_policy_fail: bool = False,
) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "INVARLOCK_GGUF_RESOURCE_ROOT": str(runtime_root),
            "INVARLOCK_GGUF_BACKEND_EXECUTABLE": "backend/llama-completion",
            "INVARLOCK_GGUF_BACKEND_SOURCE": "backend/llama.cpp-b10015.tar.gz",
        }
    )
    base = [sys.executable, "-m", "invarlock"]
    evaluation = [
        *base,
        "evaluate",
        str(paths.request),
        "--signing-key",
        str(paths.evidence_key),
        "--container-engine",
        container_engine,
        "--baseline-runtime-image",
        baseline_image_id,
        "--baseline-runtime-image-digest",
        baseline_image_id,
        "--baseline-runtime-device",
        profile.baseline_device,
        "--subject-runtime-image",
        subject_image_id,
        "--subject-runtime-image-digest",
        subject_image_id,
        "--subject-runtime-device",
        profile.subject_device,
        "--runtime-cpus",
        _WORKER_CPUS,
        "--json",
    ]
    preflight = launch._run(
        [*evaluation, "--preflight"],
        cwd=repository,
        capture_output=True,
        environment=environment,
    )
    try:
        request_digest = json.loads(preflight.stdout)["request_digest"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "deployment preflight did not return a request identity"
        ) from exc
    if (
        not isinstance(request_digest, str)
        or not request_digest.startswith("sha256:")
        or len(request_digest) != 71
    ):
        raise RuntimeError("deployment preflight returned an invalid request identity")
    _materialize_trust(
        paths,
        pending_trust,
        request_digest,
        verifier_identity=_VERIFIER_IDENTITY,
    )
    launch._run(evaluation, cwd=repository, environment=environment)
    verify_command = [
        *base,
        "verify",
        str(paths.evidence),
        "--trust-profile",
        str(paths.trusted_inputs),
        "--receipt",
        str(paths.receipt),
        "--json",
    ]
    if allow_policy_fail:
        completed = run_bounded_command(
            verify_command,
            cwd=repository,
            capture_output=True,
            check=False,
            timeout_seconds=24 * 60 * 60,
            stdout_limit=4 * 1024 * 1024,
            stderr_limit=4 * 1024 * 1024,
            label="GGUF deployment verification command",
        )
        if completed.returncode not in {0, int(EvidencePackStatus.REPORTS)}:
            diagnostic = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                diagnostic or f"verification exited with status {completed.returncode}"
            )
    else:
        launch._run(verify_command, cwd=repository)
    launch._run(
        [*base, "report", str(paths.evidence), "--html", str(paths.html_report)],
        cwd=repository,
    )
    try:
        report = json.loads(
            (paths.evidence / "reports/evaluation.report.json").read_text(
                encoding="utf-8"
            )
        )
        receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "deployment comparison outputs are missing or invalid"
        ) from exc
    comparison = report.get("comparison") if isinstance(report, dict) else None
    value = comparison.get("value") if isinstance(comparison, dict) else None
    statement = receipt.get("statement") if isinstance(receipt, dict) else None
    verdict = statement.get("verdict") if isinstance(statement, dict) else None
    policy_verdict = (
        verdict.get("policy_verdict") if isinstance(verdict, dict) else None
    )
    expected_report_verdict = "pass" if policy_verdict == "pass" else "fail"
    if (
        not isinstance(report, dict)
        or report.get("verdict") != expected_report_verdict
        or report.get("metric") != "exact_match"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isinstance(verdict, dict)
        or verdict.get("ok") is not (policy_verdict == "pass")
        or verdict.get("integrity_ok") is not True
        or policy_verdict not in {"pass", "fail"}
        or (policy_verdict == "fail" and not allow_policy_fail)
        or not paths.html_report.is_file()
        or paths.html_report.stat().st_size == 0
    ):
        raise RuntimeError("deployment comparison did not produce verified evidence")
    print(
        f"COMPLETE policy {policy_verdict}; subject exact-match delta: "
        f"{value:.2f} percentage points"
    )
    print(f"Evidence: {paths.evidence}")
    print(f"Receipt: {paths.receipt}")
    print(f"Report: {paths.html_report}")


def _image_runner(command: list[str], *, cwd: Path) -> str:
    completed = launch._run(command, cwd=cwd, capture_output=True)
    return completed.stdout or ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--profile",
        choices=deployment_profile_keys(),
        default=DEFAULT_DEPLOYMENT_PROFILE,
        help="Closed source checkpoint and corpus profile.",
    )
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument("--baseline-runtime-image")
    parser.add_argument("--subject-runtime-image")
    parser.add_argument("--evidence-signing-key", type=Path)
    parser.add_argument("--verifier-signing-key", type=Path)
    parser.add_argument("--trust-root", type=Path)
    parser.add_argument(
        "--allow-policy-fail",
        action="store_true",
        help="Retain an independently verified policy rejection as evidence.",
    )
    parser.add_argument(
        "--ephemeral-trust-root",
        action="store_true",
        help="Use disposable generated keys; never use this mode for acceptance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    trust_values = (
        arguments.evidence_signing_key,
        arguments.verifier_signing_key,
        arguments.trust_root,
    )
    provided_trust = any(value is not None for value in trust_values)
    external_trust = all(value is not None for value in trust_values)
    if provided_trust and not external_trust:
        print(
            "FAIL --evidence-signing-key, --verifier-signing-key, and "
            "--trust-root must be supplied together",
            file=sys.stderr,
        )
        return 2
    if not external_trust and not arguments.ephemeral_trust_root:
        print(
            "FAIL caller-owned evidence/verifier keys and trust root are required; "
            "use --ephemeral-trust-root only for a disposable non-acceptance demo",
            file=sys.stderr,
        )
        return 2
    if external_trust and arguments.ephemeral_trust_root:
        print("FAIL external trust material cannot use ephemeral mode", file=sys.stderr)
        return 2
    repository = Path(__file__).resolve().parents[2]
    workspace = (
        Path(tempfile.mkdtemp(prefix="invarlock-gguf-deployment-")).resolve(strict=True)
        if arguments.workspace is None
        else Path(os.path.abspath(arguments.workspace.expanduser()))
    )
    if arguments.workspace is not None:
        if workspace.exists() or workspace.is_symlink():
            print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
            return 2
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(mode=0o700)
    cleanup_tags: list[OwnedImageTag] = []
    result = 2
    try:
        transaction = workspace / "transaction"
        transaction.mkdir()
        if external_trust:
            assert arguments.trust_root is not None
            validate_new_trust_root(
                arguments.trust_root,
                transaction_root=transaction,
            )
        profile = deployment_profile(arguments.profile)
        commit = launch._require_committed_checkout(repository)
        build_root = workspace / "build"
        build_root.mkdir()
        if arguments.baseline_runtime_image is None:
            baseline_tag = temporary_image_tag("invarlock-gguf-deployment-hf", commit)
            hf_build_root = build_root / "hf"
            hf_build_root.mkdir()
            baseline_image_id, _ = launch._runtime_image(
                repository=repository,
                build_root=hf_build_root,
                container_engine=arguments.container_engine,
                dockerfile="runtime/Dockerfile.cuda",
                image_tag=baseline_tag,
                build_arguments=("CUDA_PROFILE=cu129",),
            )
            cleanup_tags.append(
                record_owned_image_tag(
                    _image_runner,
                    arguments.container_engine,
                    baseline_tag,
                    baseline_image_id,
                    repository,
                )
            )
        else:
            baseline_image_id = _inspect_image_id(
                repository,
                container_engine=arguments.container_engine,
                image=arguments.baseline_runtime_image,
            )
        if arguments.subject_runtime_image is None:
            subject_tag = temporary_image_tag("invarlock-gguf-deployment-llama", commit)
            baseline_build = build_root / "gguf"
            baseline_build.mkdir()
            subject_image_id = _build_runtime_image(
                repository,
                baseline_build,
                container_engine=arguments.container_engine,
                image_tag=subject_tag,
            )
            cleanup_tags.append(
                record_owned_image_tag(
                    _image_runner,
                    arguments.container_engine,
                    subject_tag,
                    subject_image_id,
                    repository,
                )
            )
        else:
            subject_image_id = _inspect_image_id(
                repository,
                container_engine=arguments.container_engine,
                image=arguments.subject_runtime_image,
            )
        runtime_root = transaction / "evaluation/runtime"
        model_root = runtime_root / "models"
        model_root.mkdir(parents=True)
        _stage_backend(
            repository,
            runtime_root,
            container_engine=arguments.container_engine,
            image_id=subject_image_id,
        )
        source_checkpoint = _stage_source_checkpoint(model_root, profile.source)
        conversion = _convert_and_quantize(
            repository,
            profile=profile,
            source_checkpoint=source_checkpoint,
            source_archive=runtime_root / "backend/llama.cpp-b10015.tar.gz",
            output_root=model_root,
            container_engine=arguments.container_engine,
            conversion_image_id=baseline_image_id,
            gguf_image_id=subject_image_id,
        )
        baseline_spec = _baseline_spec(source_checkpoint, profile)
        subject_spec = _inspect_spec(
            repository,
            conversion.subject,
            container_engine=arguments.container_engine,
            image_id=subject_image_id,
            seed=_SEED,
            context_length=profile.corpus.context_length,
            batch_size=1,
            cpu_threads=_SUBJECT_CPU_THREADS,
            prompt_batch_size=_SUBJECT_PROMPT_BATCH_SIZE,
            prompt_microbatch_size=_SUBJECT_PROMPT_MICROBATCH_SIZE,
            max_output_tokens=1,
            timeout_seconds=_SUBJECT_TIMEOUT_SECONDS,
        )
        transformation = _transformation_document(
            profile=profile,
            conversion=conversion,
            conversion_image_id=baseline_image_id,
            gguf_image_id=subject_image_id,
        )
        paths, pending = _prepare_transaction(
            transaction,
            profile=profile,
            runtime_root=runtime_root,
            source_checkpoint=source_checkpoint,
            subject=conversion.subject,
            baseline_spec=baseline_spec,
            subject_spec=subject_spec,
            transformation=transformation,
            baseline_image_id=baseline_image_id,
            subject_image_id=subject_image_id,
            evidence_signing_key=arguments.evidence_signing_key,
            verifier_signing_key=arguments.verifier_signing_key,
            trust_root=arguments.trust_root,
            ephemeral_trust_root=arguments.ephemeral_trust_root,
        )
        _execute(
            repository,
            paths,
            profile=profile,
            runtime_root=runtime_root,
            container_engine=arguments.container_engine,
            baseline_image_id=baseline_image_id,
            subject_image_id=subject_image_id,
            pending_trust=pending,
            allow_policy_fail=arguments.allow_policy_fail,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
    else:
        result = 0
    finally:
        try:
            remove_owned_image_tags(
                _image_runner,
                arguments.container_engine,
                repository,
                cleanup_tags,
            )
        except RuntimeError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            result = 2
    if result == 0:
        print(f"Complete GGUF deployment workspace: {workspace}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
