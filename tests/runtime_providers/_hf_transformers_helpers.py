from __future__ import annotations

import pytest

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

_IMAGE_DIGEST = "sha256:" + "2" * 64
_REAL_BACKEND_IDENTITY = hf_transformers._installed_backend_identity
_REAL_DEVICE_FACTS = hf_transformers._observed_device_facts
_REAL_STRICT_EXECUTION_BINDING = hf_transformers._require_strict_execution_binding


class _BindingTokenizer:
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = None
    clean_up_tokenization_spaces = False
    model_max_length = 128
    padding_side = "right"
    truncation_side = "right"

    def get_vocab(self) -> dict[str, int]:
        return {"<eos>": 0, "hello": 1}

    def get_added_vocab(self) -> dict[str, int]:
        return {}


@pytest.fixture(autouse=True)
def _authenticated_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hf_transformers, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(hf_transformers, "network_allowed", lambda: False)
    monkeypatch.setattr(hf_transformers, "remote_code_allowed", lambda: False)
    monkeypatch.setattr(hf_transformers, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        hf_transformers, "resolve_runtime_image_digest", lambda: _IMAGE_DIGEST
    )
    monkeypatch.setattr(
        hf_transformers,
        "resolve_runtime_image",
        lambda: "registry.invalid/invarlock@" + _IMAGE_DIGEST,
    )
    monkeypatch.setattr(
        hf_transformers,
        "_installed_backend_identity",
        lambda _model: RuntimeBackendIdentity(
            name="transformers+torch",
            version="transformers=5.12.0;torch=2.11.0",
            source_sha256="3" * 64,
            binary_sha256="4" * 64,
            build_sha256="5" * 64,
        ),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_observed_device_facts",
        lambda _model, *, expected_device_kind: RuntimeDeviceFacts(
            device_kind=expected_device_kind,
            device_name="authenticated test device",
        ),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        lambda *, spec, identity, context: context.scorer,
    )


def _spec(**settings: JSONScalar) -> ModelRuntimeSpec:
    merged_settings: dict[str, JSONScalar] = {
        "batch_size": 1,
        "immutable_revision": "1" * 40,
        "checkpoint_tree_sha256": "a" * 64,
        "context_length": 128,
        "max_output_tokens": 16,
        "offline": True,
        "seed": 7,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": "b" * 64,
    }
    merged_settings.update(settings)
    return ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_name="hf_causal",
        settings=merged_settings,
    )


def _batch() -> EvaluationBatch:
    return EvaluationBatch(
        schedule_sha256="c" * 64,
        records=(
            EvaluationRecord(
                record_id="sample-1",
                input_text="hello",
                input_sha256="d" * 64,
                expected_output="world",
            ),
        ),
    )


def _artifact_sha256(spec: ModelRuntimeSpec | None = None) -> str:
    provider = HFTransformersProvider()
    return artifact_identity_sha256(provider.identify_artifact(spec or _spec()))


def _observation(
    batch: EvaluationBatch,
    _settings: RuntimeExecutionSettings | None = None,
    *,
    artifact_sha256: str = "e" * 64,
) -> ScoringObservation:
    return ScoringObservation(
        provider_name="hf_transformers",
        artifact_identity_sha256=artifact_sha256,
        schedule_sha256=batch.schedule_sha256,
        records=(
            RuntimeScoringRecord(
                record_id="sample-1",
                input_sha256="d" * 64,
                status="ok",
                output_text="world",
                output_sha256="f" * 64,
                logprob_sum=-1.25,
                token_count=2,
                utf8_byte_count=5,
            ),
        ),
        aggregate_source_sha256="1" * 64,
    )
