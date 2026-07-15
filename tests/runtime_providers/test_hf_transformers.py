from __future__ import annotations

import subprocess
import sys

import pytest

from invarlock.core.config_runtime import InvarLockConfig, RuntimeProviderConfig
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProvider,
    RuntimeScoringRecord,
    RuntimeSession,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersProvider,
    HFTransformersSessionFactory,
)


def _spec(**settings: JSONScalar) -> ModelRuntimeSpec:
    merged_settings: dict[str, JSONScalar] = {
        "immutable_revision": "1" * 40,
        "checkpoint_tree_sha256": "a" * 64,
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
    batch: EvaluationBatch, *, artifact_sha256: str = "e" * 64
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


def test_hf_provider_module_imports_no_backend() -> None:
    code = """
import importlib
import sys
before = set(sys.modules)
module = importlib.import_module('invarlock.runtime_providers.hf_transformers')
imported = set(sys.modules) - before
assert module.INVARLOCK_RUNTIME_PROVIDER_ABI == '1'
assert 'torch' not in imported
assert 'transformers' not in imported
assert not any(name.startswith('invarlock.adapters') for name in imported)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hf_provider_declares_full_in_process_capabilities() -> None:
    provider = HFTransformersProvider()

    capabilities = provider.capabilities()

    assert isinstance(provider, RuntimeProvider)
    assert capabilities.provider_name == "hf_transformers"
    assert capabilities.artifact_formats == ("hf_snapshot",)
    assert capabilities.execution_modes == ("in_process",)
    assert capabilities.required_extra == "hf"
    assert capabilities.required_image is None
    assert capabilities.evidence_surfaces == (
        "behavior",
        "tokenizer",
        "weights",
        "modules",
        "activations",
    )
    assert capabilities.metrics == ("exact_match", "multiple_choice_accuracy")
    assert capabilities.supported_claim_sets == (
        "invarlock-weight-edit-regression-v2",
        "invarlock-runtime-behavioral-regression-v1",
    )


def test_registry_instantiates_hf_provider_with_dedicated_abi() -> None:
    provider = CoreRegistry().get_runtime_provider("hf_transformers")

    assert isinstance(provider, HFTransformersProvider)
    assert provider.abi_version == "1"


def test_implicit_and_explicit_hf_config_produce_identical_provider_spec() -> None:
    model = {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_causal",
    }
    implicit = InvarLockConfig({"model": model})
    explicit = InvarLockConfig(
        {
            "model": {
                **model,
                "runtime_provider": {
                    "name": "hf_transformers",
                    "settings": {},
                },
            }
        }
    )

    def _provider_spec(config: InvarLockConfig) -> ModelRuntimeSpec:
        runtime_provider = config.model.runtime_provider
        assert isinstance(runtime_provider, RuntimeProviderConfig)
        return ModelRuntimeSpec(
            provider_name=runtime_provider.name,
            model_id=str(config.model.id),
            settings=runtime_provider.settings,
            adapter_name=config.model.adapter,
        )

    implicit_spec = _provider_spec(implicit)
    explicit_spec = _provider_spec(explicit)

    assert implicit_spec == explicit_spec
    HFTransformersProvider().validate_config(implicit_spec)
    HFTransformersProvider().validate_config(explicit_spec)


def test_hf_provider_reuses_bound_adapter_model_and_scorer_without_loading() -> None:
    adapter = object()
    native_model = object()
    batch = _batch()
    spec = _spec()
    artifact_sha256 = artifact_identity_sha256(
        HFTransformersProvider().identify_artifact(spec)
    )
    observation = _observation(batch, artifact_sha256=artifact_sha256)
    scored: list[EvaluationBatch] = []

    def scorer(candidate: EvaluationBatch) -> ScoringObservation:
        scored.append(candidate)
        return observation

    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256=artifact_sha256,
        model_adapter=adapter,
        native_model=native_model,
        scorer=scorer,
        close_callback=None,
    )

    session = HFTransformersProvider().open(spec, context)

    assert isinstance(session, RuntimeSession)
    assert session.model_adapter() is adapter
    assert session.native_model() is native_model
    assert session.score(batch) is observation
    assert scored == [batch]
    with pytest.raises(RuntimeError, match="provenance facts"):
        session.runtime_receipt()


def test_hf_session_factory_defers_scorer_binding_without_duplicate_load() -> None:
    adapter = object()
    native_model = object()
    batch = _batch()
    spec = _spec()
    scored: list[EvaluationBatch] = []
    factory = HFTransformersSessionFactory(
        spec=spec,
        authenticated_artifact_identity=(
            HFTransformersProvider().identify_artifact(spec)
        ),
        model_adapter=adapter,
        native_model=native_model,
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
    )

    def scorer(candidate: EvaluationBatch) -> ScoringObservation:
        scored.append(candidate)
        return _observation(
            candidate,
            artifact_sha256=factory.artifact_identity_sha256,
        )

    session = factory.open(scorer)

    assert session.model_adapter() is adapter
    assert session.native_model() is native_model
    assert session.score(batch).artifact_identity_sha256 == (
        factory.artifact_identity_sha256
    )
    assert scored == [batch]


def test_hf_session_factory_rejects_loaded_artifact_identity_mismatch() -> None:
    spec = _spec()
    mismatched_identity = HFSnapshotArtifactIdentity(
        model_id=spec.model_id,
        immutable_revision="2" * 40,
        checkpoint_tree_sha256="a" * 64,
        tokenizer_metadata_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="authenticated artifact identity"):
        HFTransformersSessionFactory(
            spec=spec,
            authenticated_artifact_identity=mismatched_identity,
            model_adapter=object(),
            native_model=object(),
            strict=True,
            allow_network=False,
            container_image_digest="sha256:" + "2" * 64,
            device_kind="cpu",
        )


def test_hf_session_closes_bound_resources_exactly_once() -> None:
    calls: list[str] = []
    batch = _batch()
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
        close_callback=lambda: calls.append("closed"),
    )
    session = HFTransformersProvider().open(_spec(), context)

    session.close()
    session.close()

    assert calls == ["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        session.score(batch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_adapter", None),
        ("native_model", None),
        ("scorer", None),
    ],
)
def test_hf_provider_requires_prebound_execution_objects(
    field: str, value: object
) -> None:
    values = {
        "strict": True,
        "allow_network": False,
        "container_image_digest": "sha256:" + "2" * 64,
        "device_kind": "cpu",
        "artifact_identity_sha256": _artifact_sha256(),
        "model_adapter": object(),
        "native_model": object(),
        "scorer": _observation,
        "close_callback": None,
    }
    values[field] = value
    context = RuntimeExecutionContext(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=field):
        HFTransformersProvider().open(_spec(), context)


def test_hf_provider_rejects_network_enabled_strict_context() -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=True,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
        close_callback=None,
    )

    with pytest.raises(ValueError, match="network"):
        HFTransformersProvider().open(_spec(), context)


def test_hf_session_rejects_scorer_pairing_drift() -> None:
    batch = _batch()
    mismatched = ScoringObservation(
        provider_name="hf_transformers",
        artifact_identity_sha256=_artifact_sha256(),
        schedule_sha256="9" * 64,
        records=_observation(batch).records,
        aggregate_source_sha256="1" * 64,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        model_adapter=object(),
        native_model=object(),
        scorer=lambda _batch: mismatched,
        close_callback=None,
    )
    session = HFTransformersProvider().open(_spec(), context)

    with pytest.raises(ValueError, match="schedule"):
        session.score(batch)


def test_hf_session_rejects_scorer_artifact_identity_drift() -> None:
    batch = _batch()
    mismatched = _observation(batch, artifact_sha256="9" * 64)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        model_adapter=object(),
        native_model=object(),
        scorer=lambda _batch: mismatched,
        close_callback=None,
    )
    session = HFTransformersProvider().open(_spec(), context)

    with pytest.raises(ValueError, match="artifact identity"):
        session.score(batch)


def test_hf_provider_rejects_context_artifact_identity_drift() -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        artifact_identity_sha256="9" * 64,
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
        close_callback=None,
    )

    with pytest.raises(ValueError, match="context artifact identity"):
        HFTransformersProvider().open(_spec(), context)


def test_hf_provider_requires_artifact_identity_in_strict_mode() -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest="sha256:" + "2" * 64,
        device_kind="cpu",
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
        close_callback=None,
    )

    with pytest.raises(ValueError, match="requires artifact_identity_sha256"):
        HFTransformersProvider().open(_spec(), context)


def test_hf_provider_identifies_bound_snapshot_without_exposing_path() -> None:
    identity = HFTransformersProvider().identify_artifact(_spec())

    assert isinstance(identity, HFSnapshotArtifactIdentity)
    assert identity.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert identity.immutable_revision == "1" * 40
    assert identity.checkpoint_tree_sha256 == "a" * 64
    assert identity.tokenizer_metadata_sha256 == "b" * 64


def test_hf_provider_pseudonymizes_local_path_from_authenticated_tree_digest(
    tmp_path,
) -> None:
    private_checkpoint = tmp_path / "private-customer" / "checkpoint"
    private_checkpoint.mkdir(parents=True)
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id=str(private_checkpoint),
        adapter_name="hf_causal",
        settings={
            "checkpoint_tree_sha256": "sha256:" + "a" * 64,
            "tokenizer_metadata_sha256": "sha256:" + "b" * 64,
        },
    )

    identity = HFTransformersProvider().identify_artifact(spec)

    assert identity.model_id == "local-checkpoint-aaaaaaaaaaaa"
    assert identity.checkpoint_tree_sha256 == "a" * 64
    assert identity.tokenizer_metadata_sha256 == "b" * 64
    assert str(tmp_path) not in repr(identity)


def test_hf_provider_rejects_unbound_local_path_instead_of_leaking_it(tmp_path) -> None:
    private_checkpoint = tmp_path / "private-customer" / "checkpoint"
    private_checkpoint.mkdir(parents=True)
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id=str(private_checkpoint),
        adapter_name="hf_causal",
        settings={
            "immutable_revision": "1" * 40,
            "tokenizer_metadata_sha256": "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="local.*checkpoint_tree_sha256"):
        HFTransformersProvider().identify_artifact(spec)


def test_hf_provider_rejects_wrong_provider_unknown_settings_and_unbound_identity() -> (
    None
):
    provider = HFTransformersProvider()
    wrong_provider = ModelRuntimeSpec(
        provider_name="llama_cpp",
        model_id="model.gguf",
        settings={},
        adapter_name=None,
    )
    unknown_setting = _spec(unexpected=True)
    unbound = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        settings={},
        adapter_name="hf_causal",
    )

    with pytest.raises(ValueError, match="provider_name"):
        provider.validate_config(wrong_provider)
    with pytest.raises(ValueError, match="unexpected"):
        provider.validate_config(unknown_setting)
    with pytest.raises(ValueError, match="immutable identity"):
        provider.identify_artifact(unbound)


def test_hf_provider_rejects_mutable_revision_even_with_tree_digest() -> None:
    with pytest.raises(ValueError, match="40-64 character"):
        HFTransformersProvider().identify_artifact(_spec(immutable_revision="main"))
