from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import invarlock.runtime_provider_evidence as evidence_module
from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    canonical_artifact_identity_json,
)
from invarlock.runtime_provider_evidence import (
    ARTIFACT_IDENTITY_FILENAME,
    MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
    PROVIDER_RECEIPT_FILENAME,
    SCORING_OBSERVATION_FILENAME,
    RuntimeProviderEvidenceError,
    RuntimeProviderEvidencePaths,
    decode_artifact_identity,
    decode_runtime_provider_capabilities,
    decode_runtime_provider_receipt,
    decode_scoring_observation,
    encode_artifact_identity,
    encode_runtime_provider_capabilities,
    encode_runtime_provider_receipt,
    encode_scoring_observation,
    load_runtime_provider_evidence,
    runtime_provider_evidence_errors,
    runtime_request_binding_errors,
    write_runtime_provider_evidence,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact() -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name="tiny-model.gguf",
        sha256="1" * 64,
        byte_length=123,
        gguf_metadata_sha256="2" * 64,
        tensor_inventory_sha256="3" * 64,
        tokenizer_metadata_sha256="4" * 64,
    )


def _capabilities(*, provider_name: str = "llama_cpp") -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=provider_name,
        artifact_formats=("gguf",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image="ghcr.io/invarlock/runtime-llama-cpp@sha256:" + "b" * 64,
    )


def _observation(
    artifact: GGUFArtifactIdentity,
    *,
    provider_name: str = "llama_cpp",
) -> ScoringObservation:
    output = "A"
    record = RuntimeScoringRecord(
        record_id="sample-1",
        input_sha256="5" * 64,
        status="ok",
        output_text=output,
        output_sha256=_sha256(output.encode("utf-8")),
    )
    records_payload = [asdict(record)]
    aggregate = _sha256(
        json.dumps(
            records_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return ScoringObservation(
        provider_name=provider_name,
        artifact_identity_sha256=artifact_identity_sha256(artifact),
        schedule_sha256="6" * 64,
        records=(record,),
        aggregate_source_sha256=aggregate,
    )


def _receipt(
    artifact: GGUFArtifactIdentity,
    observation: ScoringObservation,
    *,
    capabilities: RuntimeProviderCapabilities | None = None,
    observation_bytes: bytes | None = None,
    outer_image_digest: str | None = _IMAGE_DIGEST,
) -> RuntimeProviderReceipt:
    encoded_observation = observation_bytes or encode_scoring_observation(observation)
    return RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name="llama_cpp",
            distribution="invarlock",
            distribution_version="0.13.0",
        ),
        backend=RuntimeBackendIdentity(
            name="llama.cpp",
            version="b1234",
            source_sha256="7" * 64,
            binary_sha256="8" * 64,
            build_sha256="9" * 64,
        ),
        capabilities=capabilities or _capabilities(),
        artifact_identity=artifact,
        execution_settings=RuntimeExecutionSettings(
            seed=42,
            context_length=512,
            batch_size=1,
            max_output_tokens=32,
            timeout_seconds=120,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(
            device_kind="cpu",
            device_name="x86_64",
        ),
        outer_image_digest=outer_image_digest,
        scoring_observation_sha256=_sha256(encoded_observation),
    )


def _bundle_values() -> tuple[
    GGUFArtifactIdentity, ScoringObservation, RuntimeProviderReceipt
]:
    artifact = _artifact()
    observation = _observation(artifact)
    return artifact, observation, _receipt(artifact, observation)


def _llama_cpp_settings(
    artifact: GGUFArtifactIdentity,
    receipt: RuntimeProviderReceipt,
) -> dict[str, object]:
    execution = receipt.execution_settings
    return {
        "artifact_byte_length": artifact.byte_length,
        "artifact_sha256": artifact.sha256,
        "backend_binary_sha256": receipt.backend.binary_sha256,
        "backend_source_sha256": receipt.backend.source_sha256,
        "backend_version": receipt.backend.version,
        "batch_size": execution.batch_size,
        "context_length": execution.context_length,
        "gguf_metadata_sha256": artifact.gguf_metadata_sha256,
        "max_output_tokens": execution.max_output_tokens,
        "seed": execution.seed,
        "tensor_inventory_sha256": artifact.tensor_inventory_sha256,
        "timeout_seconds": execution.timeout_seconds,
        "tokenizer_metadata_sha256": artifact.tokenizer_metadata_sha256,
    }


def _hf_request_values() -> tuple[
    HFSnapshotArtifactIdentity, dict[str, object], RuntimeProviderReceipt
]:
    artifact = HFSnapshotArtifactIdentity(
        model_id="org/model",
        immutable_revision="a" * 40,
        checkpoint_tree_sha256="b" * 64,
        tokenizer_metadata_sha256="c" * 64,
    )
    receipt = RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name="hf_transformers",
            distribution="invarlock",
            distribution_version="0.13.0",
        ),
        backend=RuntimeBackendIdentity(
            name="transformers+torch",
            version="transformers=5.14.1;torch=2.11.0+cu128",
            source_sha256="d" * 64,
            binary_sha256="e" * 64,
            build_sha256="f" * 64,
        ),
        capabilities=RuntimeProviderCapabilities(
            provider_name="hf_transformers",
            artifact_formats=("hf_snapshot",),
            tasks=("text_causal",),
            metrics=("exact_match",),
            execution_modes=("container",),
            required_extra="hf",
            required_image=None,
        ),
        artifact_identity=artifact,
        execution_settings=RuntimeExecutionSettings(
            seed=7,
            context_length=1024,
            batch_size=1,
            max_output_tokens=32,
            timeout_seconds=900,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(
            device_kind="cuda",
            device_name="test-gpu",
            compute_capability="9.0",
        ),
        outer_image_digest=_IMAGE_DIGEST,
        scoring_observation_sha256="9" * 64,
    )
    settings: dict[str, object] = {
        "batch_size": 1,
        "checkpoint_tree_sha256": "sha256:" + "b" * 64,
        "context_length": 1024,
        "immutable_revision": "a" * 40,
        "max_output_tokens": 32,
        "offline": True,
        "seed": 7,
        "timeout_seconds": 900,
        "tokenizer_metadata_sha256": "c" * 64,
    }
    return artifact, settings, receipt


def _tensorrt_request_values() -> tuple[
    TensorRTLLMArtifactIdentity, dict[str, object], RuntimeProviderReceipt
]:
    artifact = TensorRTLLMArtifactIdentity(
        bundle_name="tensorrt-llm-sha256-" + "1" * 64,
        engine_bundle_tree_sha256="1" * 64,
        file_inventory_sha256="2" * 64,
        builder_config_sha256="3" * 64,
        tokenizer_metadata_sha256="4" * 64,
        engine_metadata_sha256="5" * 64,
        target_compute_capability="9.0",
    )
    receipt = RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name="tensorrt_llm",
            distribution="invarlock-runtime-tensorrt-llm",
            distribution_version="0.13.0",
        ),
        backend=RuntimeBackendIdentity(
            name="TensorRT-LLM",
            version="1.2.1",
            source_sha256=None,
            binary_sha256="6" * 64,
            build_sha256="7" * 64,
        ),
        capabilities=RuntimeProviderCapabilities(
            provider_name="tensorrt_llm",
            artifact_formats=("tensorrt_llm_engine",),
            tasks=("text_causal",),
            metrics=("exact_match",),
            execution_modes=("container",),
            required_extra=None,
            required_image=None,
        ),
        artifact_identity=artifact,
        execution_settings=RuntimeExecutionSettings(
            seed=11,
            context_length=2048,
            batch_size=2,
            max_output_tokens=64,
            timeout_seconds=600,
            allow_network=False,
        ),
        device=RuntimeDeviceFacts(
            device_kind="cuda",
            device_name="test-gpu",
            compute_capability="9.0",
        ),
        outer_image_digest=_IMAGE_DIGEST,
        scoring_observation_sha256="8" * 64,
    )
    settings: dict[str, object] = {
        "backend_build_sha256": "7" * 64,
        "backend_version": "1.2.1",
        "batch_size": 2,
        "builder_config_sha256": "3" * 64,
        "context_length": 2048,
        "engine_bundle_tree_sha256": "1" * 64,
        "engine_metadata_sha256": "5" * 64,
        "file_inventory_sha256": "2" * 64,
        "max_output_tokens": 64,
        "runner_binary_sha256": "6" * 64,
        "seed": 11,
        "target_compute_capability": "9.0",
        "timeout_seconds": 600,
        "tokenizer_metadata_sha256": "4" * 64,
    }
    return artifact, settings, receipt


def _vision_text_request_values() -> tuple[
    HFSnapshotArtifactIdentity, dict[str, object], RuntimeProviderReceipt
]:
    artifact, settings, receipt = _hf_request_values()
    settings["processor_metadata_sha256"] = "9" * 64
    return (
        artifact,
        settings,
        replace(
            receipt,
            plugin=replace(receipt.plugin, name="hf_vision_text"),
            capabilities=replace(
                receipt.capabilities,
                provider_name="hf_vision_text",
                tasks=("vision_text_generation",),
            ),
        ),
    )


def test_typed_codecs_round_trip_canonical_contract_values() -> None:
    artifact, observation, receipt = _bundle_values()
    capabilities = receipt.capabilities

    artifact_bytes = encode_artifact_identity(artifact)

    assert artifact_bytes == canonical_artifact_identity_json(artifact)
    assert decode_artifact_identity(artifact_bytes) == artifact
    assert (
        decode_scoring_observation(encode_scoring_observation(observation))
        == observation
    )
    assert (
        decode_runtime_provider_capabilities(
            encode_runtime_provider_capabilities(capabilities)
        )
        == capabilities
    )
    assert (
        decode_runtime_provider_receipt(encode_runtime_provider_receipt(receipt))
        == receipt
    )


def test_gguf_request_binding_authenticates_backend_artifact_and_execution() -> None:
    artifact, _observation_value, receipt = _bundle_values()
    receipt = replace(
        receipt,
        backend=replace(receipt.backend, build_sha256=None),
    )
    settings = _llama_cpp_settings(artifact, receipt)

    assert (
        runtime_request_binding_errors(
            provider_name="llama_cpp",
            settings=settings,
            artifact_identity=artifact,
            receipt=receipt,
        )
        == ()
    )

    errors = runtime_request_binding_errors(
        provider_name="llama_cpp",
        settings={**settings, "backend_binary_sha256": "0" * 64},
        artifact_identity=artifact,
        receipt=receipt,
    )
    assert errors == (
        "llama_cpp provider receipt does not match request setting "
        "'backend_binary_sha256'",
    )


@pytest.mark.parametrize("provider_name", [None, ""], ids=("non-string", "empty"))
def test_request_binding_rejects_invalid_normalized_provider(
    provider_name: object,
) -> None:
    artifact, _observation_value, receipt = _bundle_values()

    assert runtime_request_binding_errors(
        provider_name=provider_name,
        settings={},
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("normalized request provider is invalid",)


def test_request_binding_rejects_non_mapping_settings() -> None:
    artifact, _observation_value, receipt = _bundle_values()

    assert runtime_request_binding_errors(
        provider_name="llama_cpp",
        settings=(("seed", 42),),
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("normalized request runtime settings are invalid",)


def test_generic_request_binding_reports_provider_and_execution_drift() -> None:
    artifact = HFSnapshotArtifactIdentity(
        model_id="org/model",
        immutable_revision="a" * 40,
        checkpoint_tree_sha256=None,
        tokenizer_metadata_sha256="b" * 64,
    )
    _gguf_artifact, _observation_value, receipt = _bundle_values()

    assert runtime_request_binding_errors(
        provider_name="hf_transformers",
        settings={"seed": receipt.execution_settings.seed + 1},
        artifact_identity=artifact,
        receipt=receipt,
    ) == (
        "request provider does not match provider receipt",
        "provider receipt does not match request runtime setting 'seed'",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("batch_size", 3),
        ("checkpoint_tree_sha256", "0" * 64),
        ("context_length", 2048),
        ("immutable_revision", "0" * 40),
        ("max_output_tokens", 64),
        ("offline", False),
        ("seed", 8),
        ("timeout_seconds", 901),
        ("tokenizer_metadata_sha256", "0" * 64),
    ],
)
def test_hf_request_binding_rejects_every_closed_setting_substitution(
    field: str,
    replacement: object,
) -> None:
    artifact, settings, receipt = _hf_request_values()

    errors = runtime_request_binding_errors(
        provider_name="hf_transformers",
        settings={**settings, field: replacement},
        artifact_identity=artifact,
        receipt=receipt,
    )

    assert any(field in error or field == "offline" for error in errors)


def test_hf_request_binding_accepts_digest_spellings_and_optional_absence() -> None:
    artifact, settings, receipt = _hf_request_values()

    assert (
        runtime_request_binding_errors(
            provider_name="hf_transformers",
            settings=settings,
            artifact_identity=artifact,
            receipt=receipt,
        )
        == ()
    )

    local_artifact = replace(artifact, immutable_revision=None)
    local_receipt = replace(receipt, artifact_identity=local_artifact)
    local_settings = {**settings, "tokenizer_metadata_sha256": "sha256:" + "c" * 64}
    del local_settings["immutable_revision"]
    assert (
        runtime_request_binding_errors(
            provider_name="hf_transformers",
            settings=local_settings,
            artifact_identity=local_artifact,
            receipt=local_receipt,
        )
        == ()
    )


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_hf_request_binding_rejects_nonclosed_settings(mutation: str) -> None:
    artifact, settings, receipt = _hf_request_values()
    if mutation == "missing":
        del settings["tokenizer_metadata_sha256"]
    else:
        settings["unreviewed"] = "value"

    assert runtime_request_binding_errors(
        provider_name="hf_transformers",
        settings=settings,
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("hf_transformers request settings are not the closed supported set",)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("batch_size", 3),
        ("checkpoint_tree_sha256", "0" * 64),
        ("context_length", 2048),
        ("immutable_revision", "0" * 40),
        ("max_output_tokens", 64),
        ("offline", False),
        ("seed", 8),
        ("timeout_seconds", 901),
        ("tokenizer_metadata_sha256", "0" * 64),
    ],
)
def test_vision_text_request_binding_rejects_bound_setting_substitution(
    field: str,
    replacement: object,
) -> None:
    artifact, settings, receipt = _vision_text_request_values()

    errors = runtime_request_binding_errors(
        provider_name="hf_vision_text",
        settings={**settings, field: replacement},
        artifact_identity=artifact,
        receipt=receipt,
    )

    assert any(field in error or field == "offline" for error in errors)


def test_vision_text_processor_digest_has_bounded_v1_validation() -> None:
    artifact, settings, receipt = _vision_text_request_values()

    assert (
        runtime_request_binding_errors(
            provider_name="hf_vision_text",
            settings={**settings, "processor_metadata_sha256": "sha256:" + "9" * 64},
            artifact_identity=artifact,
            receipt=receipt,
        )
        == ()
    )
    assert runtime_request_binding_errors(
        provider_name="hf_vision_text",
        settings={**settings, "processor_metadata_sha256": "not-a-digest"},
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("hf_vision_text request processor metadata digest is invalid",)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_vision_text_request_binding_rejects_nonclosed_settings(
    mutation: str,
) -> None:
    artifact, settings, receipt = _vision_text_request_values()
    if mutation == "missing":
        del settings["processor_metadata_sha256"]
    else:
        settings["unreviewed"] = "value"

    assert runtime_request_binding_errors(
        provider_name="hf_vision_text",
        settings=settings,
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("hf_vision_text request settings are not the closed supported set",)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("backend_build_sha256", "0" * 64),
        ("backend_version", "2.0.0"),
        ("batch_size", 3),
        ("builder_config_sha256", "0" * 64),
        ("context_length", 4096),
        ("engine_bundle_tree_sha256", "0" * 64),
        ("engine_metadata_sha256", "0" * 64),
        ("file_inventory_sha256", "0" * 64),
        ("max_output_tokens", 65),
        ("runner_binary_sha256", "0" * 64),
        ("seed", 12),
        ("target_compute_capability", "8.0"),
        ("timeout_seconds", 601),
        ("tokenizer_metadata_sha256", "0" * 64),
    ],
)
def test_tensorrt_request_binding_rejects_every_closed_setting_substitution(
    field: str,
    replacement: object,
) -> None:
    artifact, settings, receipt = _tensorrt_request_values()

    errors = runtime_request_binding_errors(
        provider_name="tensorrt_llm",
        settings={**settings, field: replacement},
        artifact_identity=artifact,
        receipt=receipt,
    )

    assert any(field in error for error in errors)


def test_tensorrt_request_binding_accepts_complete_bound_request() -> None:
    artifact, settings, receipt = _tensorrt_request_values()

    assert (
        runtime_request_binding_errors(
            provider_name="tensorrt_llm",
            settings=settings,
            artifact_identity=artifact,
            receipt=receipt,
        )
        == ()
    )


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_tensorrt_request_binding_rejects_nonclosed_settings(mutation: str) -> None:
    artifact, settings, receipt = _tensorrt_request_values()
    if mutation == "missing":
        del settings["runner_binary_sha256"]
    else:
        settings["unreviewed"] = "value"

    assert runtime_request_binding_errors(
        provider_name="tensorrt_llm",
        settings=settings,
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("tensorrt_llm request settings are not the closed supported set",)


def test_provider_request_binding_rejects_runtime_boundary_substitutions() -> None:
    hf_artifact, hf_settings, hf_receipt = _hf_request_values()
    hf_errors = runtime_request_binding_errors(
        provider_name="hf_transformers",
        settings=hf_settings,
        artifact_identity=hf_artifact,
        receipt=replace(
            hf_receipt,
            execution_settings=replace(
                hf_receipt.execution_settings,
                allow_network=True,
            ),
        ),
    )
    assert hf_errors == (
        "hf_transformers provider receipt must disable network access",
    )

    trt_artifact, trt_settings, trt_receipt = _tensorrt_request_values()
    trt_errors = runtime_request_binding_errors(
        provider_name="tensorrt_llm",
        settings=trt_settings,
        artifact_identity=trt_artifact,
        receipt=replace(
            trt_receipt,
            backend=replace(
                trt_receipt.backend,
                name="TensorRT",
                source_sha256="9" * 64,
            ),
            execution_settings=replace(
                trt_receipt.execution_settings,
                allow_network=True,
            ),
            device=replace(trt_receipt.device, compute_capability="8.0"),
        ),
    )
    assert trt_errors == (
        "tensorrt_llm provider receipt backend name is invalid",
        "tensorrt_llm provider receipt backend source digest must be null",
        "tensorrt_llm provider receipt must disable network access",
        "tensorrt_llm provider receipt device compute capability does not match "
        "the artifact target",
    )


def test_request_binding_rejects_provider_artifact_type_substitution() -> None:
    artifact, _observation_value, receipt = _bundle_values()
    hf_artifact = HFSnapshotArtifactIdentity(
        model_id="org/model",
        immutable_revision="a" * 40,
        checkpoint_tree_sha256=None,
        tokenizer_metadata_sha256="b" * 64,
    )
    closed_settings = _llama_cpp_settings(artifact, receipt)

    assert runtime_request_binding_errors(
        provider_name="llama_cpp",
        settings=closed_settings,
        artifact_identity=hf_artifact,
        receipt=receipt,
    ) == ("llama_cpp request and GGUF artifact identity do not agree",)
    assert runtime_request_binding_errors(
        provider_name="hf_transformers",
        settings={},
        artifact_identity=artifact,
        receipt=replace(
            receipt,
            plugin=replace(receipt.plugin, name="hf_transformers"),
            capabilities=replace(
                receipt.capabilities,
                provider_name="hf_transformers",
            ),
        ),
    ) == ("hf_transformers request and HF artifact identity do not agree",)


def test_gguf_request_binding_rejects_open_settings_and_runtime_substitution() -> None:
    artifact, _observation_value, receipt = _bundle_values()
    valid_receipt = replace(
        receipt,
        backend=replace(receipt.backend, build_sha256=None),
    )
    settings = _llama_cpp_settings(artifact, valid_receipt)

    assert runtime_request_binding_errors(
        provider_name="llama_cpp",
        settings={**settings, "unreviewed": "value"},
        artifact_identity=artifact,
        receipt=valid_receipt,
    ) == ("llama_cpp request settings are not the closed supported set",)

    substituted_receipt = replace(
        valid_receipt,
        backend=replace(
            valid_receipt.backend,
            name="substituted-backend",
            build_sha256="9" * 64,
        ),
        execution_settings=replace(
            valid_receipt.execution_settings,
            allow_network=True,
        ),
    )
    substituted_settings = _llama_cpp_settings(artifact, substituted_receipt)
    assert runtime_request_binding_errors(
        provider_name="llama_cpp",
        settings=substituted_settings,
        artifact_identity=artifact,
        receipt=substituted_receipt,
    ) == (
        "llama_cpp provider receipt backend name is invalid",
        "llama_cpp provider receipt backend build digest must be null",
        "llama_cpp provider receipt must disable network access",
    )


@pytest.mark.parametrize(
    "identity",
    [
        HFSnapshotArtifactIdentity(
            model_id="org/model",
            immutable_revision="a" * 40,
            checkpoint_tree_sha256=None,
            tokenizer_metadata_sha256="b" * 64,
        ),
        _artifact(),
        TensorRTLLMArtifactIdentity(
            bundle_name="engine-bundle",
            engine_bundle_tree_sha256="c" * 64,
            file_inventory_sha256="d" * 64,
            builder_config_sha256="e" * 64,
            tokenizer_metadata_sha256="a" * 64,
            engine_metadata_sha256="f" * 64,
            target_compute_capability="9.0",
        ),
    ],
    ids=("hf-snapshot", "gguf", "tensorrt-llm"),
)
def test_artifact_identity_codec_reconstructs_every_supported_type(
    identity: HFSnapshotArtifactIdentity
    | GGUFArtifactIdentity
    | TensorRTLLMArtifactIdentity,
) -> None:
    encoded = encode_artifact_identity(identity)

    assert encoded == canonical_artifact_identity_json(identity)
    assert decode_artifact_identity(encoded) == identity
    assert type(decode_artifact_identity(encoded)) is type(identity)


def test_artifact_identity_dispatch_rejects_unknown_internal_format() -> None:
    with pytest.raises(RuntimeProviderEvidenceError, match="unsupported artifact"):
        evidence_module._artifact_from_payload({"artifact_format": "future"})


def test_typed_decoder_enforces_sidecar_bound_before_json_parsing() -> None:
    oversized = b"{" + b" " * MAX_RUNTIME_PROVIDER_SIDECAR_BYTES + b"}"

    with pytest.raises(RuntimeProviderEvidenceError, match="size limit"):
        decode_artifact_identity(oversized)


def test_unknown_provider_cannot_claim_a_gguf_artifact_contract() -> None:
    artifact, _observation_value, receipt = _bundle_values()
    receipt = replace(
        receipt,
        plugin=replace(receipt.plugin, name="future_provider"),
        capabilities=replace(
            receipt.capabilities,
            provider_name="future_provider",
        ),
    )

    assert runtime_request_binding_errors(
        provider_name="future_provider",
        settings={},
        artifact_identity=artifact,
        receipt=receipt,
    ) == ("llama_cpp request and GGUF artifact identity do not agree",)


def test_write_and_reload_produces_cross_bound_canonical_sidecars(
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()

    bundle = write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
        expected_outer_image_digest=_IMAGE_DIGEST,
    )

    assert bundle.artifact_identity == artifact
    assert bundle.scoring_observation == observation
    assert bundle.receipt == receipt
    assert bundle.capabilities == receipt.capabilities
    assert bundle.artifact_identity_bytes == encode_artifact_identity(artifact)
    assert bundle.scoring_observation_bytes == encode_scoring_observation(observation)
    assert bundle.receipt_bytes == encode_runtime_provider_receipt(receipt)
    assert bundle.artifact_identity_sha256 == _sha256(bundle.artifact_identity_bytes)
    assert bundle.scoring_observation_sha256 == receipt.scoring_observation_sha256
    assert bundle.receipt_sha256 == _sha256(bundle.receipt_bytes)
    assert {path.name for path in tmp_path.iterdir()} == {
        ARTIFACT_IDENTITY_FILENAME,
        SCORING_OBSERVATION_FILENAME,
        PROVIDER_RECEIPT_FILENAME,
    }


def test_write_rejects_cross_binding_drift_without_creating_sidecars(
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    mismatched = replace(receipt, scoring_observation_sha256="0" * 64)

    with pytest.raises(
        RuntimeProviderEvidenceError,
        match="scoring_observation_sha256 does not match",
    ):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=mismatched,
        )

    assert list(tmp_path.iterdir()) == []


def test_cross_binding_reports_provider_artifact_capability_and_image_drift() -> None:
    artifact, observation, receipt = _bundle_values()
    other_artifact = replace(artifact, sha256="0" * 64)
    other_observation = replace(
        observation,
        provider_name="other_provider",
        artifact_identity_sha256="d" * 64,
    )
    unsupported_capabilities = replace(
        receipt.capabilities,
        provider_name="llama_cpp",
        artifact_formats=("hf_snapshot",),
    )
    drifted_receipt = replace(
        receipt,
        artifact_identity=other_artifact,
        capabilities=unsupported_capabilities,
    )

    errors = runtime_provider_evidence_errors(
        artifact_identity=artifact,
        scoring_observation=other_observation,
        receipt=drifted_receipt,
        expected_outer_image_digest="sha256:" + "c" * 64,
    )

    assert errors == (
        "receipt artifact_identity does not match the bound artifact",
        "receipt scoring_observation_sha256 does not match observation bytes",
        "receipt and observation provider names do not agree",
        "observation artifact_identity_sha256 does not match bound artifact",
        "bound artifact format is not declared by provider capabilities",
        "receipt outer_image_digest does not match expected runtime image",
    )


def test_cross_binding_rejects_noncanonical_expected_image_digest() -> None:
    artifact, observation, receipt = _bundle_values()

    errors = runtime_provider_evidence_errors(
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
        expected_outer_image_digest="a" * 64,
    )

    assert errors == ("expected outer image digest is not canonical",)


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        (
            SCORING_OBSERVATION_FILENAME,
            b'{"provider_name":"a","provider_name":"b"}',
            "duplicate key",
        ),
        (PROVIDER_RECEIPT_FILENAME, b'{"value":NaN}', "non-standard constant"),
        (ARTIFACT_IDENTITY_FILENAME, b"[]", "must be a JSON object"),
    ],
)
def test_reload_rejects_ambiguous_or_nonobject_json(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    message: str,
) -> None:
    artifact, observation, receipt = _bundle_values()
    write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
    )
    (tmp_path / filename).write_bytes(payload)

    with pytest.raises(RuntimeProviderEvidenceError, match=message):
        load_runtime_provider_evidence(
            RuntimeProviderEvidencePaths.in_directory(tmp_path)
        )


def test_reload_rejects_schema_invalid_sidecar(tmp_path: Path) -> None:
    artifact, observation, receipt = _bundle_values()
    write_runtime_provider_evidence(
        tmp_path,
        artifact_identity=artifact,
        scoring_observation=observation,
        receipt=receipt,
    )
    (tmp_path / PROVIDER_RECEIPT_FILENAME).write_bytes(b"{}")

    with pytest.raises(RuntimeProviderEvidenceError, match="schema validation failed"):
        load_runtime_provider_evidence(
            RuntimeProviderEvidencePaths.in_directory(tmp_path)
        )


def test_reload_rejects_symlink_and_hardlink_role_aliases(tmp_path: Path) -> None:
    paths = RuntimeProviderEvidencePaths.in_directory(tmp_path)
    paths.artifact_identity.write_bytes(encode_artifact_identity(_artifact()))
    paths.receipt.write_bytes(b"{}")
    paths.scoring_observation.symlink_to(paths.artifact_identity)

    with pytest.raises(RuntimeProviderEvidenceError, match="regular file"):
        load_runtime_provider_evidence(paths)

    paths.scoring_observation.unlink()
    os.link(paths.artifact_identity, paths.scoring_observation)
    with pytest.raises(RuntimeProviderEvidenceError, match="must not alias"):
        load_runtime_provider_evidence(paths)


def test_reload_rejects_oversized_regular_sidecar(tmp_path: Path) -> None:
    paths = RuntimeProviderEvidencePaths.in_directory(tmp_path)
    paths.artifact_identity.write_bytes(b"{}")
    paths.scoring_observation.write_bytes(b"{}")
    paths.receipt.write_bytes(b"{}")
    with paths.artifact_identity.open("r+b") as handle:
        handle.truncate(MAX_RUNTIME_PROVIDER_SIDECAR_BYTES + 1)

    with pytest.raises(RuntimeProviderEvidenceError, match="exceeds"):
        load_runtime_provider_evidence(paths)


def test_atomic_write_failure_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()

    def _fail_link(_source: Path, _target: Path, *, follow_symlinks: bool) -> None:
        assert follow_symlinks is False
        raise OSError("injected link failure")

    monkeypatch.setattr("invarlock.runtime_provider_evidence.os.link", _fail_link)

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert list(tmp_path.iterdir()) == []


def test_late_publication_failure_rolls_back_earlier_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    real_link = os.link
    calls = 0

    def _fail_second_link(source: Path, target: Path, *, follow_symlinks: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-link failure")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        "invarlock.runtime_provider_evidence.os.link", _fail_second_link
    )

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert list(tmp_path.iterdir()) == []


def test_publication_race_does_not_replace_existing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact, observation, receipt = _bundle_values()
    real_link = os.link
    raced_target = tmp_path / ARTIFACT_IDENTITY_FILENAME

    def _race_link(source: Path, target: Path, *, follow_symlinks: bool) -> None:
        raced_target.write_text("created concurrently", encoding="utf-8")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("invarlock.runtime_provider_evidence.os.link", _race_link)

    with pytest.raises(RuntimeProviderEvidenceError, match="atomically write"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert raced_target.read_text(encoding="utf-8") == "created concurrently"


def test_write_rejects_existing_canonical_sidecar(tmp_path: Path) -> None:
    artifact, observation, receipt = _bundle_values()
    (tmp_path / ARTIFACT_IDENTITY_FILENAME).write_text("do not replace")

    with pytest.raises(RuntimeProviderEvidenceError, match="already exist"):
        write_runtime_provider_evidence(
            tmp_path,
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
        )

    assert (tmp_path / ARTIFACT_IDENTITY_FILENAME).read_text() == "do not replace"


def test_reload_rejects_noncanonical_sidecar_filenames(tmp_path: Path) -> None:
    paths = RuntimeProviderEvidencePaths(
        artifact_identity=tmp_path / "artifact.json",
        scoring_observation=tmp_path / "observation.json",
        receipt=tmp_path / "receipt.json",
    )

    with pytest.raises(RuntimeProviderEvidenceError, match="canonical filenames"):
        load_runtime_provider_evidence(paths)
