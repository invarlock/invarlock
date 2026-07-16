from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProvider,
    RuntimeSession,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation
from invarlock.runtime_providers.hf_transformers import HFTransformersProvider
from tests.runtime_providers._hf_transformers_helpers import (
    _IMAGE_DIGEST,
    _artifact_sha256,
    _authenticated_test_runtime,  # noqa: F401
    _batch,
    _observation,
    _spec,
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
    assert capabilities.metrics == (
        "exact_match",
        "normalized_nll_per_utf8_byte",
    )


def test_registry_instantiates_hf_provider_with_dedicated_abi() -> None:
    provider = CoreRegistry().get_runtime_provider("hf_transformers")

    assert isinstance(provider, HFTransformersProvider)
    assert provider.abi_version == "1"


def test_hf_provider_reuses_bound_provider_state_and_scorer_without_loading() -> None:
    provider_state = object()
    batch = _batch()
    spec = _spec()
    artifact_sha256 = artifact_identity_sha256(
        HFTransformersProvider().identify_artifact(spec)
    )
    observation = _observation(batch, artifact_sha256=artifact_sha256)
    scored: list[EvaluationBatch] = []

    def scorer(
        candidate: EvaluationBatch, _settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        scored.append(candidate)
        return observation

    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=artifact_sha256,
        provider_state=provider_state,
        scorer=scorer,
        close_callback=None,
    )

    session = HFTransformersProvider().open(spec, context)

    assert isinstance(session, RuntimeSession)
    assert session.score(batch) is observation
    assert scored == [batch]
    receipt = session.runtime_receipt()
    assert receipt.plugin.name == "hf_transformers"
    assert receipt.backend.name == "transformers+torch"
    assert receipt.artifact_identity == HFTransformersProvider().identify_artifact(spec)
    assert receipt.execution_settings == RuntimeExecutionSettings(
        seed=7,
        context_length=128,
        batch_size=1,
        max_output_tokens=16,
        timeout_seconds=30,
        allow_network=False,
    )
    assert receipt.device == RuntimeDeviceFacts(
        device_kind="cpu", device_name="authenticated test device"
    )
    assert receipt.outer_image_digest == _IMAGE_DIGEST
    assert (
        receipt.scoring_observation_sha256
        == hashlib.sha256(encode_scoring_observation(observation)).hexdigest()
    )


def test_hf_receipt_is_unavailable_before_complete_scoring() -> None:
    spec = _spec()
    session = HFTransformersProvider().open(
        spec,
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=_artifact_sha256(spec),
            provider_state=object(),
            scorer=_observation,
        ),
    )

    with pytest.raises(RuntimeError, match="before scoring"):
        session.runtime_receipt()


def test_hf_strict_scorer_receives_exact_provider_settings_object() -> None:
    spec = _spec()
    batch = _batch()
    received: list[RuntimeExecutionSettings] = []

    def settings_observer(
        candidate: EvaluationBatch, settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        received.append(settings)
        return _observation(
            candidate,
            settings,
            artifact_sha256=_artifact_sha256(spec),
        )

    session = HFTransformersProvider().open(
        spec,
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=_artifact_sha256(spec),
            provider_state=object(),
            scorer=settings_observer,
        ),
    )

    session.score(batch)
    receipt = session.runtime_receipt()

    assert received == [receipt.execution_settings]
    assert received[0] is receipt.execution_settings


def test_hf_strict_scorer_rejects_legacy_one_argument_contract() -> None:
    spec = _spec()
    batch = _batch()

    def legacy_scorer(candidate: EvaluationBatch) -> ScoringObservation:
        return _observation(candidate, artifact_sha256=_artifact_sha256(spec))

    session = HFTransformersProvider().open(
        spec,
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=_artifact_sha256(spec),
            provider_state=object(),
            scorer=legacy_scorer,  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(RuntimeError, match="exact runtime execution settings"):
        session.score(batch)


def test_hf_session_closes_bound_resources_exactly_once() -> None:
    calls: list[str] = []
    batch = _batch()
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        provider_state=object(),
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
        ("provider_state", None),
        ("scorer", None),
    ],
)
def test_hf_provider_requires_prebound_execution_objects(
    field: str, value: object
) -> None:
    values = {
        "strict": True,
        "allow_network": False,
        "container_image_digest": _IMAGE_DIGEST,
        "device_kind": "cpu",
        "artifact_identity_sha256": _artifact_sha256(),
        "provider_state": object(),
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
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        provider_state=object(),
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
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        provider_state=object(),
        scorer=lambda _batch, _settings: mismatched,
        close_callback=None,
    )
    session = HFTransformersProvider().open(_spec(), context)

    with pytest.raises(ValueError, match="schedule"):
        session.score(batch)


def test_hf_failed_rescore_invalidates_previous_observation_receipt() -> None:
    batch = _batch()
    artifact_sha256 = _artifact_sha256()
    valid = _observation(batch, artifact_sha256=artifact_sha256)
    invalid = ScoringObservation(
        provider_name="hf_transformers",
        artifact_identity_sha256=artifact_sha256,
        schedule_sha256="9" * 64,
        records=valid.records,
        aggregate_source_sha256="1" * 64,
    )
    observations = iter((valid, invalid))
    session = HFTransformersProvider().open(
        _spec(),
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=artifact_sha256,
            provider_state=object(),
            scorer=lambda _batch, _settings: next(observations),
        ),
    )
    session.score(batch)
    session.runtime_receipt()

    with pytest.raises(ValueError, match="schedule"):
        session.score(batch)
    with pytest.raises(RuntimeError, match="before scoring"):
        session.runtime_receipt()


def test_hf_session_rejects_scorer_artifact_identity_drift() -> None:
    batch = _batch()
    mismatched = _observation(batch, artifact_sha256="9" * 64)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(),
        provider_state=object(),
        scorer=lambda _batch, _settings: mismatched,
        close_callback=None,
    )
    session = HFTransformersProvider().open(_spec(), context)

    with pytest.raises(ValueError, match="artifact identity"):
        session.score(batch)


def test_hf_provider_rejects_context_artifact_identity_drift() -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256="9" * 64,
        provider_state=object(),
        scorer=_observation,
        close_callback=None,
    )

    with pytest.raises(ValueError, match="context artifact identity"):
        HFTransformersProvider().open(_spec(), context)


def test_hf_provider_requires_artifact_identity_in_strict_mode() -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        provider_state=object(),
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
    )
    unknown_setting = _spec(unexpected=True)
    unbound = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        settings={},
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
