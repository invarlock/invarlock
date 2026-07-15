from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest

from invarlock.core.runtime_provider import (
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_provider_evidence import (
    encode_runtime_provider_receipt,
    encode_scoring_observation,
)
from invarlock.runtime_providers import tensorrt_llm_canary as canary
from invarlock.runtime_providers.tensorrt_llm import TensorRTLLMProvider
from invarlock.runtime_providers.tensorrt_llm_session import (
    TensorRTLLMRuntimeBindings,
)

_IMAGE_DIGEST = "sha256:" + "9" * 64


def _runner_info() -> dict[str, str]:
    return {
        "backend_build_sha256": "a" * 64,
        "backend_name": "TensorRT-LLM",
        "backend_version": "1.2.1",
        "cuda_compute_capability": "9.0",
        "cuda_device_name": "NVIDIA H200",
        "cuda_driver_version": "575.57.08",
        "cuda_runtime_version": "12.9",
        "device_kind": "cuda",
        "format_version": "invarlock/tensorrt-llm-runner-info-v1",
        "protocol_version": "invarlock/tensorrt-llm-runner-v1",
    }


def _identity(tokenizer_sha256: str) -> TensorRTLLMArtifactIdentity:
    return TensorRTLLMArtifactIdentity(
        bundle_name="tensorrt-llm-sha256-" + "b" * 64,
        engine_bundle_tree_sha256="b" * 64,
        file_inventory_sha256="c" * 64,
        builder_config_sha256="d" * 64,
        tokenizer_metadata_sha256=tokenizer_sha256,
        engine_metadata_sha256="e" * 64,
        target_compute_capability="9.0",
    )


def test_runner_info_contract_rejects_unknown_or_unpinned_values() -> None:
    assert canary._validate_runner_info(_runner_info()) == _runner_info()  # noqa: SLF001

    unknown = {**_runner_info(), "unexpected": "value"}
    with pytest.raises(canary.TensorRTLLMCanaryError, match="unexpected schema"):
        canary._validate_runner_info(unknown)  # noqa: SLF001

    wrong_version = {**_runner_info(), "backend_version": "1.3.0"}
    with pytest.raises(canary.TensorRTLLMCanaryError, match="pinned contract"):
        canary._validate_runner_info(wrong_version)  # noqa: SLF001


def test_image_binding_requires_exact_digest_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv(
        "INVARLOCK_RUNTIME_IMAGE", "registry.invalid/invarlock@" + _IMAGE_DIGEST
    )
    assert canary._require_image_binding() == _IMAGE_DIGEST  # noqa: SLF001

    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "sha256:" + "8" * 64)
    with pytest.raises(canary.TensorRTLLMCanaryError, match="exact candidate"):
        canary._require_image_binding()  # noqa: SLF001


@pytest.mark.parametrize("value", ["", "sha256:" + "a" * 64, "A" * 64, "a" * 63])
def test_expected_fixture_digests_must_be_canonical(value: str) -> None:
    with pytest.raises(canary.TensorRTLLMCanaryError, match="lowercase sha256"):
        canary._required_expected_sha256(  # noqa: SLF001
            value, label="expected fixture digest"
        )


def test_candidate_qualification_uses_real_provider_session_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", _IMAGE_DIGEST)
    engine = tmp_path / "engine"
    engine.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b'{"tokenizer":"fixture"}')
    runner = tmp_path / "runner"
    runner.write_bytes(b"#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
    identity = _identity(tokenizer_sha256)
    info = _runner_info()
    calls: list[str] = []
    receipt_versions: list[str] = []
    emitted_receipts: list[RuntimeProviderReceipt] = []

    monkeypatch.setattr(canary, "_raw_runner_info", lambda _runner: info)

    def identify(
        path: Path, *, target_compute_capability: str, tokenizer_metadata_sha256: str
    ) -> TensorRTLLMArtifactIdentity:
        assert path == engine
        assert target_compute_capability == "9.0"
        assert tokenizer_metadata_sha256 == tokenizer_sha256
        return identity

    monkeypatch.setattr(canary, "read_tensorrt_llm_artifact_identity", identify)
    capabilities = TensorRTLLMProvider().capabilities()

    class Session:
        observation: ScoringObservation | None = None

        def __init__(self) -> None:
            self.distribution_version = (
                receipt_versions.pop(0) if receipt_versions else "0.0.0"
            )

        def score(self, batch):  # noqa: ANN001, ANN201
            calls.append("score")
            record = RuntimeScoringRecord(
                record_id=batch.records[0].record_id,
                input_sha256=batch.records[0].input_sha256,
                status="ok",
                output_text="qualified",
                output_sha256=hashlib.sha256(b"qualified").hexdigest(),
            )
            self.observation = ScoringObservation(
                provider_name="tensorrt_llm",
                artifact_identity_sha256=artifact_identity_sha256(identity),
                schedule_sha256=batch.schedule_sha256,
                records=(record,),
                aggregate_source_sha256=runtime_scoring_records_sha256(
                    [asdict(record)]
                ),
            )
            return self.observation

        def runtime_receipt(self) -> RuntimeProviderReceipt:
            calls.append("receipt")
            assert self.observation is not None
            receipt = RuntimeProviderReceipt(
                plugin=RuntimeProviderPluginIdentity(
                    name="tensorrt_llm",
                    distribution="invarlock",
                    distribution_version=self.distribution_version,
                ),
                backend=RuntimeBackendIdentity(
                    name="TensorRT-LLM",
                    version="1.2.1",
                    source_sha256=None,
                    binary_sha256=runner_sha256,
                    build_sha256="a" * 64,
                ),
                capabilities=capabilities,
                artifact_identity=identity,
                execution_settings=RuntimeExecutionSettings(
                    seed=0,
                    context_length=8,
                    batch_size=1,
                    max_output_tokens=1,
                    timeout_seconds=300,
                    allow_network=False,
                ),
                device=RuntimeDeviceFacts(
                    device_kind="cuda",
                    device_name="NVIDIA H200",
                    compute_capability="9.0",
                    driver_version="575.57.08",
                    cuda_runtime_version="12.9",
                ),
                outer_image_digest=_IMAGE_DIGEST,
                scoring_observation_sha256=hashlib.sha256(
                    encode_scoring_observation(self.observation)
                ).hexdigest(),
            )
            emitted_receipts.append(receipt)
            return receipt

        def close(self) -> None:
            calls.append("close")

    class Provider:
        def validate_config(self, spec):  # noqa: ANN001, ANN201
            calls.append("validate")
            assert spec.model_id == identity.bundle_name
            assert spec.settings["runner_binary_sha256"] == runner_sha256

        def open(self, spec, context):  # noqa: ANN001, ANN201
            calls.append("open")
            assert spec.settings["backend_build_sha256"] == "a" * 64
            assert context.strict is True
            assert context.container_image_digest == _IMAGE_DIGEST
            assert context.artifact_identity_sha256 == artifact_identity_sha256(
                identity
            )
            assert isinstance(context.native_model, TensorRTLLMRuntimeBindings)
            return Session()

    monkeypatch.setattr(canary, "TensorRTLLMProvider", Provider)

    qualification_args = {
        "engine_bundle": engine,
        "tokenizer_contract": tokenizer,
        "runner": runner,
        "expected_engine_tree_sha256": identity.engine_bundle_tree_sha256,
        "expected_tokenizer_sha256": tokenizer_sha256,
    }
    with pytest.raises(canary.TensorRTLLMCanaryError, match="reviewed digest"):
        canary.qualify_candidate(
            **qualification_args,
            expected_output_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )

    calls.clear()
    receipt_versions[:] = ["0.0.0", "0.0.1"]
    with pytest.raises(canary.TensorRTLLMCanaryError, match="not deterministic"):
        canary.qualify_candidate(
            **qualification_args,
            expected_output_sha256=hashlib.sha256(b"qualified").hexdigest(),
        )

    calls.clear()
    receipt_versions[:] = ["0.0.0", "0.0.0"]
    emitted_receipts.clear()
    result = canary.qualify_candidate(
        **qualification_args,
        expected_output_sha256=hashlib.sha256(b"qualified").hexdigest(),
    )

    assert result["ok"] is True
    assert result["artifact_identity_sha256"] == artifact_identity_sha256(identity)
    assert len(emitted_receipts) == 2
    assert (
        result["runtime_provider_receipt_sha256"]
        == hashlib.sha256(
            encode_runtime_provider_receipt(emitted_receipts[-1])
        ).hexdigest()
    )
    assert calls == [
        "validate",
        "open",
        "score",
        "receipt",
        "close",
        "open",
        "score",
        "receipt",
        "close",
    ]


def test_candidate_qualification_rejects_unreviewed_fixture_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", _IMAGE_DIGEST)
    engine = tmp_path / "engine"
    engine.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    runner = tmp_path / "runner"
    runner.write_bytes(b"runner")
    runner.chmod(0o755)
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    identity = _identity(tokenizer_sha256)
    monkeypatch.setattr(canary, "_raw_runner_info", lambda _runner: _runner_info())
    monkeypatch.setattr(
        canary,
        "read_tensorrt_llm_artifact_identity",
        lambda *_args, **_kwargs: identity,
    )

    with pytest.raises(canary.TensorRTLLMCanaryError, match="tokenizer contract"):
        canary.qualify_candidate(
            engine_bundle=engine,
            tokenizer_contract=tokenizer,
            runner=runner,
            expected_engine_tree_sha256=identity.engine_bundle_tree_sha256,
            expected_tokenizer_sha256="f" * 64,
            expected_output_sha256=hashlib.sha256(b"qualified").hexdigest(),
        )

    with pytest.raises(canary.TensorRTLLMCanaryError, match="engine bundle"):
        canary.qualify_candidate(
            engine_bundle=engine,
            tokenizer_contract=tokenizer,
            runner=runner,
            expected_engine_tree_sha256="f" * 64,
            expected_tokenizer_sha256=tokenizer_sha256,
            expected_output_sha256=hashlib.sha256(b"qualified").hexdigest(),
        )


def test_candidate_qualification_fails_closed_on_session_device_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", _IMAGE_DIGEST)
    engine = tmp_path / "engine"
    engine.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    runner = tmp_path / "runner"
    runner.write_bytes(b"runner")
    runner.chmod(0o755)
    tokenizer_sha256 = hashlib.sha256(b"{}").hexdigest()
    identity = _identity(tokenizer_sha256)
    monkeypatch.setattr(canary, "_raw_runner_info", lambda _runner: _runner_info())
    monkeypatch.setattr(
        canary,
        "read_tensorrt_llm_artifact_identity",
        lambda *_args, **_kwargs: identity,
    )

    class Session:
        def score(self, batch):  # noqa: ANN001, ANN201
            record = RuntimeScoringRecord(
                record_id=batch.records[0].record_id,
                input_sha256=batch.records[0].input_sha256,
                status="ok",
                output_text="qualified",
                output_sha256=hashlib.sha256(b"qualified").hexdigest(),
            )
            self.observation = ScoringObservation(
                provider_name="tensorrt_llm",
                artifact_identity_sha256=artifact_identity_sha256(identity),
                schedule_sha256=batch.schedule_sha256,
                records=(record,),
                aggregate_source_sha256=runtime_scoring_records_sha256(
                    [asdict(record)]
                ),
            )
            return self.observation

        def runtime_receipt(self) -> RuntimeProviderReceipt:
            return RuntimeProviderReceipt(
                plugin=RuntimeProviderPluginIdentity(
                    name="tensorrt_llm",
                    distribution="invarlock",
                    distribution_version="0.0.0",
                ),
                backend=RuntimeBackendIdentity(
                    name="TensorRT-LLM",
                    version="1.2.1",
                    source_sha256=None,
                    binary_sha256=hashlib.sha256(b"runner").hexdigest(),
                    build_sha256="a" * 64,
                ),
                capabilities=TensorRTLLMProvider().capabilities(),
                artifact_identity=identity,
                execution_settings=RuntimeExecutionSettings(
                    seed=0,
                    context_length=8,
                    batch_size=1,
                    max_output_tokens=1,
                    timeout_seconds=300,
                ),
                device=RuntimeDeviceFacts(
                    device_kind="cuda",
                    device_name="different GPU",
                    compute_capability="9.0",
                    driver_version="575.57.08",
                    cuda_runtime_version="12.9",
                ),
                outer_image_digest=_IMAGE_DIGEST,
                scoring_observation_sha256=hashlib.sha256(
                    encode_scoring_observation(self.observation)
                ).hexdigest(),
            )

        def close(self) -> None:
            return None

    class Provider:
        def validate_config(self, _spec):  # noqa: ANN001, ANN201
            return None

        def open(self, _spec, _context):  # noqa: ANN001, ANN201
            return Session()

    monkeypatch.setattr(canary, "TensorRTLLMProvider", Provider)

    with pytest.raises(canary.TensorRTLLMCanaryError, match="device facts changed"):
        canary.qualify_candidate(
            engine_bundle=engine,
            tokenizer_contract=tokenizer,
            runner=runner,
            expected_engine_tree_sha256=identity.engine_bundle_tree_sha256,
            expected_tokenizer_sha256=tokenizer_sha256,
            expected_output_sha256=hashlib.sha256(b"qualified").hexdigest(),
        )
