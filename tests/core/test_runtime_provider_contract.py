from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sys
from pathlib import Path

import pytest


def test_runtime_provider_core_import_is_torch_free() -> None:
    before = set(sys.modules)

    module = importlib.import_module("invarlock.core.runtime_provider")

    assert module.INVARLOCK_RUNTIME_PROVIDER_ABI == "1"
    assert "torch" not in set(sys.modules) - before


def test_runtime_provider_protocol_accepts_complete_structural_implementations() -> (
    None
):
    from invarlock.core.runtime_provider import RuntimeProvider, RuntimeSession

    class Session:
        def score(self, batch):
            return batch

        def runtime_receipt(self):
            return None

        def close(self):
            return None

    class Provider:
        name = "test_provider"
        abi_version = "1"

        def validate_config(self, spec):
            return None

        def capabilities(self):
            return None

        def identify_artifact(self, spec):
            return None

        def prepare_execution(self, spec, resources):
            return resources

        def open(self, spec, context):
            return Session()

    assert isinstance(Session(), RuntimeSession)
    assert isinstance(Provider(), RuntimeProvider)


def test_runtime_execution_context_binds_opaque_provider_state_and_callbacks() -> None:
    from invarlock.core.runtime_provider.types import RuntimeExecutionContext

    provider_state = object()
    scorer = lambda batch, settings: (batch, settings)  # noqa: E731
    close_callback = lambda: None  # noqa: E731

    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "a" * 64,
        device_kind="cuda",
        artifact_identity_sha256="b" * 64,
        provider_state=provider_state,
        scorer=scorer,
        close_callback=close_callback,
    )

    assert context.provider_state is provider_state
    assert context.scorer is scorer
    assert context.close_callback is close_callback
    assert context.artifact_identity_sha256 == "b" * 64
    assert dataclasses.is_dataclass(context)
    assert context.__dataclass_params__.frozen is True


def test_runtime_execution_settings_mapping_uses_shared_validation() -> None:
    from invarlock.core.runtime_provider.types import (
        RuntimeExecutionSettings,
        runtime_execution_settings_from_mapping,
    )

    settings = runtime_execution_settings_from_mapping(
        {
            "seed": 7,
            "context_length": 2048,
            "batch_size": 1,
            "max_output_tokens": 64,
            "timeout_seconds": 30,
        },
        allow_network=False,
    )

    assert settings == RuntimeExecutionSettings(
        seed=7,
        context_length=2048,
        batch_size=1,
        max_output_tokens=64,
        timeout_seconds=30,
        allow_network=False,
    )
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        runtime_execution_settings_from_mapping(
            {
                "seed": True,
                "context_length": 2048,
                "batch_size": 1,
                "max_output_tokens": 64,
                "timeout_seconds": 30,
            },
            allow_network=False,
        )


def test_runtime_provider_capabilities_are_closed_and_canonical() -> None:
    from invarlock.core.runtime_provider.types import RuntimeProviderCapabilities

    capabilities = RuntimeProviderCapabilities(
        provider_name="hf_transformers",
        artifact_formats=("hf_snapshot",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("in_process",),
        required_extra="hf",
        required_image=None,
    )

    assert capabilities.format_version == "runtime-provider-capabilities-v1"
    assert capabilities.provider_abi == "1"

    with pytest.raises(ValueError, match="duplicate"):
        dataclasses.replace(capabilities, metrics=("exact_match", "exact_match"))
    with pytest.raises(ValueError, match="unsupported artifact format"):
        dataclasses.replace(capabilities, artifact_formats=("pickle",))
    with pytest.raises(ValueError, match="provider_name"):
        dataclasses.replace(capabilities, provider_name="../provider")


def test_runtime_provider_capabilities_accept_future_canonical_tasks() -> None:
    from invarlock.core.runtime_provider.types import RuntimeProviderCapabilities

    capabilities = RuntimeProviderCapabilities(
        provider_name="future_runtime",
        artifact_formats=("hf_snapshot",),
        tasks=(
            "text_causal",
            "text_seq2seq",
            "masked_language",
            "vision_text_generation",
            "audio_text_generation",
        ),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image=None,
    )

    assert capabilities.tasks[-1] == "audio_text_generation"
    with pytest.raises(ValueError, match="canonical task identifier"):
        dataclasses.replace(capabilities, tasks=("audio/text",))


def test_model_artifact_identity_variants_reject_paths_and_bad_digests() -> None:
    from invarlock.core.runtime_provider.types import (
        GGUFArtifactIdentity,
        HFSnapshotArtifactIdentity,
        TensorRTLLMArtifactIdentity,
        artifact_identity_sha256,
        canonical_artifact_identity_json,
    )

    hf = HFSnapshotArtifactIdentity(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        immutable_revision="1" * 40,
        checkpoint_tree_sha256="a" * 64,
        tokenizer_metadata_sha256="b" * 64,
    )
    gguf = GGUFArtifactIdentity(
        artifact_name="tinyllama-q4.gguf",
        sha256="a" * 64,
        byte_length=123,
        gguf_metadata_sha256="b" * 64,
        tensor_inventory_sha256="c" * 64,
        tokenizer_metadata_sha256="d" * 64,
    )
    engine = TensorRTLLMArtifactIdentity(
        bundle_name="tinyllama-engine",
        engine_bundle_tree_sha256="a" * 64,
        file_inventory_sha256="b" * 64,
        builder_config_sha256="c" * 64,
        tokenizer_metadata_sha256="e" * 64,
        engine_metadata_sha256="d" * 64,
        target_compute_capability="9.0",
    )

    assert hf.artifact_format == "hf_snapshot"
    assert gguf.artifact_format == "gguf"
    assert engine.artifact_format == "tensorrt_llm_engine"
    canonical = canonical_artifact_identity_json(gguf)
    assert canonical == (
        b'{"artifact_format":"gguf","artifact_name":"tinyllama-q4.gguf",'
        b'"byte_length":123,"format_version":"invarlock/model-artifact-identity-v1",'
        b'"gguf_metadata_sha256":"'
        + b"b" * 64
        + b'","sha256":"'
        + b"a" * 64
        + b'","tensor_inventory_sha256":"'
        + b"c" * 64
        + b'","tokenizer_metadata_sha256":"'
        + b"d" * 64
        + b'"}'
    )
    assert artifact_identity_sha256(gguf) == hashlib.sha256(canonical).hexdigest()

    with pytest.raises(ValueError, match="absolute or traversal path"):
        dataclasses.replace(gguf, artifact_name="/models/model.gguf")
    with pytest.raises(ValueError, match="40-64 character"):
        dataclasses.replace(hf, immutable_revision="main")
    with pytest.raises(ValueError, match="absolute or traversal path"):
        dataclasses.replace(engine, bundle_name="../engine")
    with pytest.raises(ValueError, match="sha256"):
        dataclasses.replace(hf, checkpoint_tree_sha256="bad")
    with pytest.raises(ValueError, match="tokenizer_metadata_sha256"):
        dataclasses.replace(engine, tokenizer_metadata_sha256="bad")
    with pytest.raises(ValueError, match="major.minor"):
        dataclasses.replace(engine, target_compute_capability="09.0")
    with pytest.raises(ValueError, match="at least one"):
        dataclasses.replace(hf, immutable_revision=None, checkpoint_tree_sha256=None)


def test_runtime_device_facts_keep_driver_and_cuda_runtime_distinct() -> None:
    from invarlock.core.runtime_provider.types import RuntimeDeviceFacts

    facts = RuntimeDeviceFacts(
        device_kind="cuda",
        device_name="NVIDIA H200",
        compute_capability="9.0",
        driver_version="570.00",
        cuda_runtime_version="12.8",
    )

    assert facts.driver_version == "570.00"
    assert facts.cuda_runtime_version == "12.8"
    with pytest.raises(ValueError, match="cuda_runtime_version"):
        dataclasses.replace(facts, cuda_runtime_version=" 12.8 ")


def test_evaluation_batch_and_scoring_observation_validate_pairing() -> None:
    from invarlock.core.runtime_provider.types import (
        EvaluationBatch,
        EvaluationRecord,
        RuntimeScoringRecord,
        ScoringObservation,
    )

    record = EvaluationRecord(
        record_id="sample-1",
        input_text="hello",
        input_sha256="a" * 64,
        expected_output="world",
    )
    batch = EvaluationBatch(schedule_sha256="b" * 64, records=(record,))
    scored = RuntimeScoringRecord(
        record_id="sample-1",
        input_sha256="a" * 64,
        status="ok",
        output_text="world",
        output_sha256="c" * 64,
        logprob_sum=-1.25,
        token_count=2,
        utf8_byte_count=5,
    )
    observation = ScoringObservation(
        provider_name="hf_transformers",
        artifact_identity_sha256="d" * 64,
        schedule_sha256=batch.schedule_sha256,
        records=(scored,),
        aggregate_source_sha256="e" * 64,
    )

    assert observation.format_version == "invarlock/runtime-scoring-observation-v1"
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(batch, records=(record, record))
    with pytest.raises(ValueError, match="error_code"):
        dataclasses.replace(scored, status="error")
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(scored, logprob_sum=float("nan"))


def test_model_runtime_spec_freezes_provider_settings() -> None:
    from invarlock.core.runtime_provider.types import ModelRuntimeSpec

    source = {"seed": 43, "offline": True}
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        settings=source,
    )
    source["seed"] = 99

    assert dict(spec.settings) == {"seed": 43, "offline": True}
    with pytest.raises(TypeError):
        spec.settings["seed"] = 100  # type: ignore[index]
    with pytest.raises(ValueError, match="JSON scalar"):
        dataclasses.replace(spec, settings={"nested": {"bad": True}})


def test_runtime_provider_type_module_has_no_backend_imports() -> None:
    from invarlock.core.runtime_provider import types as runtime_provider_types

    source = Path(runtime_provider_types.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import transformers" not in source
    assert "import tensorrt" not in source
    assert "import llama_cpp" not in source
