from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest

from invarlock.core.runtime_provider.types import (
    EvaluationBatch,
    EvaluationRecord,
    GGUFArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    canonical_artifact_identity_json,
    runtime_execution_settings_from_mapping,
)

_SHA256 = "a" * 64
_IMAGE_DIGEST = "sha256:" + "b" * 64


def _capabilities() -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name="llama_cpp",
        artifact_formats=("gguf",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("local_process",),
        required_extra=None,
        required_image="ghcr.io/example/runtime@" + _IMAGE_DIGEST,
        platform_constraints=("linux",),
        evidence_surfaces=("behavior", "tokenizer"),
        supported_claim_sets=("runtime-behavioral",),
    )


def _artifact() -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name="model.gguf",
        sha256=_SHA256,
        byte_length=1,
        gguf_metadata_sha256="b" * 64,
        tensor_inventory_sha256="c" * 64,
        tokenizer_metadata_sha256="d" * 64,
    )


def _evaluation_record(record_id: str = "record-1") -> EvaluationRecord:
    return EvaluationRecord(
        record_id=record_id,
        input_text="prompt",
        input_sha256=_SHA256,
        expected_output="answer",
    )


def _scoring_record(record_id: str = "record-1") -> RuntimeScoringRecord:
    return RuntimeScoringRecord(
        record_id=record_id,
        input_sha256=_SHA256,
        status="ok",
        output_text="answer",
        output_sha256="b" * 64,
    )


def test_string_and_path_contracts_reject_control_and_path_material() -> None:
    with pytest.raises(ValueError, match="control characters"):
        RuntimeProviderPluginIdentity(
            name="llama_cpp",
            distribution="invar\x00lock",
            distribution_version="1.0",
        )

    with pytest.raises(ValueError, match="absolute or traversal path"):
        dataclasses.replace(_artifact(), artifact_name="C:model.gguf")
    with pytest.raises(ValueError, match="absolute or traversal path"):
        dataclasses.replace(_artifact(), artifact_name="models/model.gguf")


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_positive_integer_contract_rejects_nonpositive_or_nonnumeric(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="byte_length must be a positive integer"):
        dataclasses.replace(_artifact(), byte_length=value)  # type: ignore[arg-type]


def test_capabilities_reject_empty_nontuple_duplicate_and_whitespace() -> None:
    capabilities = _capabilities()

    with pytest.raises(ValueError, match="artifact_formats must be a non-empty tuple"):
        dataclasses.replace(capabilities, artifact_formats=cast(Any, []))
    with pytest.raises(ValueError, match="supported_claim_sets must be a tuple"):
        dataclasses.replace(capabilities, supported_claim_sets=())
    with pytest.raises(
        ValueError, match="platform_constraints must not contain duplicate"
    ):
        dataclasses.replace(capabilities, platform_constraints=("linux", "linux"))
    with pytest.raises(ValueError, match="required_image must not contain whitespace"):
        dataclasses.replace(capabilities, required_image="image with-space")


def test_canonical_artifact_identity_rejects_arbitrary_objects() -> None:
    with pytest.raises(TypeError, match="supported model artifact identity"):
        canonical_artifact_identity_json(cast(Any, object()))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (cast(Any, []), "settings must be a mapping"),
        ({"Upper": 1}, "keys must be canonical names"),
        ({"temperature": float("inf")}, "finite JSON scalar"),
    ],
)
def test_model_runtime_spec_rejects_noncanonical_settings(
    settings: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelRuntimeSpec(
            provider_name="llama_cpp",
            model_id="model",
            settings=settings,
        )


def test_evaluation_record_rejects_nontext_fields() -> None:
    record = _evaluation_record()

    with pytest.raises(ValueError, match="input_text must be a string"):
        dataclasses.replace(record, input_text=cast(Any, 7))
    with pytest.raises(ValueError, match="expected_output must be a string or null"):
        dataclasses.replace(record, expected_output=cast(Any, 7))


@pytest.mark.parametrize("records", [(), cast(Any, [])])
def test_evaluation_batch_requires_a_nonempty_tuple(records: Any) -> None:
    with pytest.raises(ValueError, match="records must be a non-empty tuple"):
        EvaluationBatch(schedule_sha256=_SHA256, records=records)


def test_scoring_record_rejects_incoherent_status_and_output_facts() -> None:
    record = _scoring_record()

    with pytest.raises(ValueError, match="status must be ok or error"):
        dataclasses.replace(record, status=cast(Any, "skipped"))
    with pytest.raises(ValueError, match="output_text must be a string or null"):
        dataclasses.replace(record, output_text=cast(Any, 7))
    with pytest.raises(ValueError, match="must be present together"):
        dataclasses.replace(record, output_sha256=None)
    with pytest.raises(ValueError, match="positive token_count and utf8_byte_count"):
        dataclasses.replace(
            record,
            logprob_sum=-1.0,
            token_count=0,
            utf8_byte_count=1,
        )
    with pytest.raises(ValueError, match="error_code is required"):
        RuntimeScoringRecord(
            record_id="record-1",
            input_sha256=_SHA256,
            status="error",
        )
    with pytest.raises(ValueError, match="must not contain measured output facts"):
        RuntimeScoringRecord(
            record_id="record-1",
            input_sha256=_SHA256,
            status="error",
            output_text="answer",
            output_sha256="b" * 64,
            error_code="backend_error",
        )
    with pytest.raises(ValueError, match="error_code must be null"):
        dataclasses.replace(record, error_code="backend_error")
    with pytest.raises(ValueError, match="require output or logprob facts"):
        RuntimeScoringRecord(
            record_id="record-1",
            input_sha256=_SHA256,
            status="ok",
        )


def test_scoring_observation_requires_nonempty_unique_records() -> None:
    record = _scoring_record()

    with pytest.raises(ValueError, match="records must be a non-empty tuple"):
        ScoringObservation(
            provider_name="llama_cpp",
            artifact_identity_sha256=_SHA256,
            schedule_sha256="b" * 64,
            records=(),
            aggregate_source_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="record IDs must be unique"):
        ScoringObservation(
            provider_name="llama_cpp",
            artifact_identity_sha256=_SHA256,
            schedule_sha256="b" * 64,
            records=(record, record),
            aggregate_source_sha256="c" * 64,
        )


def test_execution_settings_and_mapping_reject_invalid_types() -> None:
    valid = RuntimeExecutionSettings(
        seed=0,
        context_length=1,
        batch_size=1,
        max_output_tokens=1,
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        dataclasses.replace(valid, seed=cast(Any, "0"))
    with pytest.raises(ValueError, match="allow_network must be boolean"):
        dataclasses.replace(valid, allow_network=cast(Any, 0))
    with pytest.raises(TypeError, match="settings must be a mapping"):
        runtime_execution_settings_from_mapping(cast(Any, []), allow_network=False)


def test_execution_context_rejects_invalid_ephemeral_bindings() -> None:
    with pytest.raises(ValueError, match="strict must be boolean"):
        RuntimeExecutionContext(
            strict=cast(Any, 1),
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
        )
    with pytest.raises(ValueError, match="allow_network must be boolean"):
        RuntimeExecutionContext(
            strict=True,
            allow_network=cast(Any, 0),
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
        )
    with pytest.raises(ValueError, match="sha256 image digest"):
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest="latest",
            device_kind="cpu",
        )
    with pytest.raises(ValueError, match="scorer must be callable"):
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            scorer=cast(Any, object()),
        )
    with pytest.raises(ValueError, match="close_callback must be callable"):
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            close_callback=cast(Any, object()),
        )


def test_backend_device_and_receipt_fail_closed_on_missing_identity() -> None:
    with pytest.raises(ValueError, match="at least one backend digest"):
        RuntimeBackendIdentity(
            name="llama.cpp",
            version="1.0",
            source_sha256=None,
            binary_sha256=None,
            build_sha256=None,
        )
    with pytest.raises(ValueError, match="major.minor notation"):
        RuntimeDeviceFacts(
            device_kind="cuda",
            device_name="GPU",
            compute_capability="9",
        )

    with pytest.raises(ValueError, match="provider names must match"):
        RuntimeProviderReceipt(
            plugin=RuntimeProviderPluginIdentity(
                name="tensorrt_llm",
                distribution="invarlock",
                distribution_version="1.0",
            ),
            backend=RuntimeBackendIdentity(
                name="llama.cpp",
                version="1.0",
                source_sha256=_SHA256,
                binary_sha256=None,
                build_sha256=None,
            ),
            capabilities=_capabilities(),
            artifact_identity=_artifact(),
            execution_settings=RuntimeExecutionSettings(
                seed=0,
                context_length=1,
                batch_size=1,
                max_output_tokens=1,
                timeout_seconds=1,
            ),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="CPU"),
            outer_image_digest=None,
            scoring_observation_sha256=_SHA256,
        )
