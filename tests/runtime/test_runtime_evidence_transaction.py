from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

import invarlock.runtime_behavior.transaction as runtime_transaction
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    HFSnapshotArtifactIdentity,
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
    artifact_identity_sha256,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_behavior import run_evidence_side
from invarlock.runtime_provider_evidence import encode_scoring_observation
from invarlock.runtime_verify import verify_runtime_manifest

_IMAGE_DIGEST = "sha256:" + "a" * 64
_IMAGE_REF = f"registry.example/runtime@{_IMAGE_DIGEST}"


def _identity() -> HFSnapshotArtifactIdentity:
    return HFSnapshotArtifactIdentity(
        model_id="portable-model",
        immutable_revision="b" * 40,
        checkpoint_tree_sha256="c" * 64,
        tokenizer_metadata_sha256="d" * 64,
    )


def _capabilities() -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name="fixture_provider",
        artifact_formats=("hf_snapshot",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image=None,
    )


@dataclass
class _Session:
    observation: ScoringObservation | None = None
    closed: bool = False

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        records = tuple(
            RuntimeScoringRecord(
                record_id=record.record_id,
                input_sha256=record.input_sha256,
                status="ok",
                output_text=record.expected_output,
                output_sha256=hashlib.sha256(
                    (record.expected_output or "").encode("utf-8")
                ).hexdigest(),
            )
            for record in batch.records
        )
        self.observation = ScoringObservation(
            provider_name="fixture_provider",
            artifact_identity_sha256=artifact_identity_sha256(_identity()),
            schedule_sha256=batch.schedule_sha256,
            records=records,
            aggregate_source_sha256=runtime_scoring_records_sha256(
                [asdict(record) for record in records]
            ),
        )
        return self.observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        assert self.observation is not None
        return RuntimeProviderReceipt(
            plugin=RuntimeProviderPluginIdentity(
                name="fixture_provider",
                distribution="fixture-provider",
                distribution_version="1",
            ),
            backend=RuntimeBackendIdentity(
                name="fixture",
                version="1",
                source_sha256="e" * 64,
                binary_sha256=None,
                build_sha256=None,
            ),
            capabilities=_capabilities(),
            artifact_identity=_identity(),
            execution_settings=RuntimeExecutionSettings(
                seed=0,
                context_length=8,
                batch_size=1,
                max_output_tokens=1,
                timeout_seconds=30,
                allow_network=False,
            ),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="fixture"),
            outer_image_digest=_IMAGE_DIGEST,
            scoring_observation_sha256=hashlib.sha256(
                encode_scoring_observation(self.observation)
            ).hexdigest(),
        )

    def close(self) -> None:
        self.closed = True


class _Provider:
    name = "fixture_provider"
    abi_version = "1"

    def __init__(self) -> None:
        self.session = _Session()

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError("wrong provider")

    def capabilities(self) -> RuntimeProviderCapabilities:
        return _capabilities()

    def identify_artifact(self, spec: ModelRuntimeSpec) -> HFSnapshotArtifactIdentity:
        self.validate_config(spec)
        return _identity()

    def open(
        self, spec: ModelRuntimeSpec, context: RuntimeExecutionContext
    ) -> _Session:
        self.validate_config(spec)
        assert context.strict
        return self.session


def test_runtime_evidence_side_contract_publishes_typed_strict_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # This is a contract test. The real HF prepare/open/score test is separate;
    # release trust still requires the unmocked Docker journey.
    monkeypatch.setattr(
        runtime_transaction, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(
        runtime_transaction, "resolve_runtime_image_digest", lambda: _IMAGE_DIGEST
    )
    monkeypatch.setattr(
        runtime_transaction, "resolve_runtime_image", lambda: _IMAGE_REF
    )
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "contract",
            "config_name": None,
            "revision": "f" * 40,
            "split": "validation",
        },
        records=[
            {"record_id": "one", "input_text": "Return A", "expected_output": "A"}
        ],
    )
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    provider = _Provider()
    spec = ModelRuntimeSpec(
        provider_name=provider.name,
        model_id="portable-model",
        settings={
            "batch_size": 1,
            "context_length": 8,
            "max_output_tokens": 1,
            "seed": 0,
            "timeout_seconds": 30,
        },
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=artifact_identity_sha256(_identity()),
    )

    bundle = run_evidence_side(
        role="baseline",
        provider=provider,
        spec=spec,
        context=context,
        schedule_path=schedule_path,
        policy_digest="sha256:" + "1" * 64,
        output_directory=tmp_path / "baseline",
    )

    assert provider.session.closed is True
    assert json.loads(bundle.report_path.read_bytes())["format"] == (
        "invarlock/runtime-side-report-v1"
    )
    assert json.loads(bundle.config_path.read_bytes())["policy_digest"] == (
        "sha256:" + "1" * 64
    )
    verified = verify_runtime_manifest(
        bundle.report_path,
        bundle.manifest_path,
        expected_image_digest=_IMAGE_DIGEST,
        require_strict_runtime=True,
    )
    assert verified.ok, verified.errors
