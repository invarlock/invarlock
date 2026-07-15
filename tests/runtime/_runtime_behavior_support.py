from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import invarlock.runtime_behavior.side as runtime_behavior_side
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    ModelArtifactIdentity,
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
    build_runtime_behavioral_schedule,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.policy_pack import build_behavioral_policy_pack
from invarlock.reporting.validation.runtime_behavioral_claim import (
    runtime_execution_settings_sha256,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_behavior import (
    run_side,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation

_IMAGE_DIGEST = "sha256:" + "a" * 64
_IMAGE_REF = "registry.example/invarlock-runtime@" + _IMAGE_DIGEST


@pytest.fixture(autouse=True)
def _strict_container_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", _IMAGE_REF)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _IMAGE_DIGEST)
    monkeypatch.setattr(
        runtime_behavior_side,
        "strict_container_boundary_present",
        lambda: True,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_payload(record: RuntimeScoringRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "status": record.status,
        "output_text": record.output_text,
        "output_sha256": record.output_sha256,
        "logprob_sum": record.logprob_sum,
        "token_count": record.token_count,
        "utf8_byte_count": record.utf8_byte_count,
        "error_code": record.error_code,
    }


def _execution_settings(*, allow_network: bool = False) -> RuntimeExecutionSettings:
    return RuntimeExecutionSettings(
        seed=7,
        context_length=128,
        batch_size=1,
        max_output_tokens=8,
        timeout_seconds=30,
        allow_network=allow_network,
    )


@dataclass
class _FakeSession:
    provider_name: str
    capabilities: RuntimeProviderCapabilities
    artifact_identity: ModelArtifactIdentity
    outputs: tuple[str, ...]
    reverse_records: bool = False
    score_error: bool = False
    closed: bool = False
    observation: ScoringObservation | None = None
    receipt_allow_network: bool = False
    receipt_image_digest: str = _IMAGE_DIGEST

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        if self.score_error:
            raise RuntimeError("backend failed")
        records = tuple(
            RuntimeScoringRecord(
                record_id=record.record_id,
                input_sha256=record.input_sha256,
                status="ok",
                output_text=output,
                output_sha256=_sha256(output.encode("utf-8")),
            )
            for record, output in zip(batch.records, self.outputs, strict=True)
        )
        if self.reverse_records:
            records = tuple(reversed(records))
        record_payloads = [_record_payload(record) for record in records]
        self.observation = ScoringObservation(
            provider_name=self.provider_name,
            artifact_identity_sha256=artifact_identity_sha256(self.artifact_identity),
            schedule_sha256=batch.schedule_sha256,
            records=records,
            aggregate_source_sha256=runtime_scoring_records_sha256(record_payloads),
        )
        return self.observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        assert self.observation is not None
        return RuntimeProviderReceipt(
            plugin=RuntimeProviderPluginIdentity(
                name=self.provider_name,
                distribution="invarlock-test-provider",
                distribution_version="1.0.0",
            ),
            backend=RuntimeBackendIdentity(
                name="test-backend",
                version="1",
                source_sha256="b" * 64,
                binary_sha256=None,
                build_sha256=None,
            ),
            capabilities=self.capabilities,
            artifact_identity=self.artifact_identity,
            execution_settings=_execution_settings(
                allow_network=self.receipt_allow_network
            ),
            device=RuntimeDeviceFacts(device_kind="cpu", device_name="test-cpu"),
            outer_image_digest=self.receipt_image_digest,
            scoring_observation_sha256=_sha256(
                encode_scoring_observation(self.observation)
            ),
        )

    def model_adapter(self) -> None:
        return None

    def native_model(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProvider:
    abi_version = "1"

    def __init__(
        self,
        *,
        name: str,
        identity: ModelArtifactIdentity,
        outputs: tuple[str, ...],
        reverse_records: bool = False,
        score_error: bool = False,
        receipt_allow_network: bool = False,
        receipt_image_digest: str = _IMAGE_DIGEST,
    ) -> None:
        self.name = name
        self.identity = identity
        self.outputs = outputs
        self.reverse_records = reverse_records
        self.score_error = score_error
        self.receipt_allow_network = receipt_allow_network
        self.receipt_image_digest = receipt_image_digest
        self.session: _FakeSession | None = None

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name or spec.model_id != "portable-model":
            raise ValueError("invalid test provider spec")

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=(self.identity.artifact_format,),
            tasks=("text_causal",),
            metrics=("exact_match",),
            execution_modes=("container",),
            required_extra=None,
            required_image=None,
            platform_constraints=("test",),
            evidence_surfaces=("behavior", "tokenizer", "build"),
            supported_claim_sets=("invarlock-runtime-behavioral-regression-v1",),
        )

    def identify_artifact(self, spec: ModelRuntimeSpec) -> ModelArtifactIdentity:
        self.validate_config(spec)
        return self.identity

    def open(
        self, spec: ModelRuntimeSpec, context: RuntimeExecutionContext
    ) -> _FakeSession:
        self.validate_config(spec)
        assert context.strict
        self.session = _FakeSession(
            provider_name=self.name,
            capabilities=self.capabilities(),
            artifact_identity=self.identity,
            outputs=self.outputs,
            reverse_records=self.reverse_records,
            score_error=self.score_error,
            receipt_allow_network=self.receipt_allow_network,
            receipt_image_digest=self.receipt_image_digest,
        )
        return self.session


def _baseline_identity() -> HFSnapshotArtifactIdentity:
    return HFSnapshotArtifactIdentity(
        model_id="portable-model",
        immutable_revision="1" * 40,
        checkpoint_tree_sha256=None,
        tokenizer_metadata_sha256="2" * 64,
    )


def _subject_identity() -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name="portable-model.gguf",
        sha256="3" * 64,
        byte_length=256,
        gguf_metadata_sha256="4" * 64,
        tensor_inventory_sha256="5" * 64,
        tokenizer_metadata_sha256="6" * 64,
    )


def _binding(
    *,
    provider_name: str,
    identity: ModelArtifactIdentity,
    settings: RuntimeExecutionSettings | None = None,
) -> dict[str, object]:
    return {
        "provider_name": provider_name,
        "artifact_format": identity.artifact_format,
        "artifact_identity_sha256": artifact_identity_sha256(identity),
        "outer_image_digest": _IMAGE_DIGEST,
        "execution_settings_sha256": runtime_execution_settings_sha256(
            settings or _execution_settings()
        ),
    }


def _behavioral_policy(
    schedule,
    *,
    minimum_subject_score: float = 0.5,
    maximum_regression: float = 0.5,
    baseline_binding: dict[str, object] | None = None,
    subject_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_behavioral_policy_pack(
        tier="conservative",
        schedule_sha256=schedule.schedule_sha256,
        baseline=baseline_binding
        or _binding(
            provider_name="hf_transformers",
            identity=_baseline_identity(),
        ),
        subject=subject_binding
        or _binding(provider_name="llama_cpp", identity=_subject_identity()),
        metric_kind="exact_match",
        minimum_subject_score=minimum_subject_score,
        maximum_regression=maximum_regression,
        dataset_identity=schedule.dataset_identity.to_payload(),
        required_evidence_surfaces=["behavior", "build", "tokenizer"],
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    prompt_a = "Choose A"
    prompt_b = "Choose B"
    schedule = build_runtime_behavioral_schedule(
        {
            "format_version": "invarlock/runtime-behavioral-schedule-v1",
            "dataset_identity": {
                "provider": "synthetic",
                "dataset_name": None,
                "config_name": None,
                "revision": None,
                "split": "test",
            },
            "records": [
                {
                    "record_id": "sample-a",
                    "input_text": prompt_a,
                    "input_sha256": _sha256(prompt_a.encode("utf-8")),
                    "expected_output": "A",
                },
                {
                    "record_id": "sample-b",
                    "input_text": prompt_b,
                    "input_sha256": _sha256(prompt_b.encode("utf-8")),
                    "expected_output": "B",
                },
            ],
        }
    )
    policy = _behavioral_policy(schedule)
    schedule_path = tmp_path / "schedule.json"
    policy_path = tmp_path / "policy.json"
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))
    policy_path.write_text(
        json.dumps(policy, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return schedule_path, policy_path


def _rewrite_policy(
    policy_path: Path,
    *,
    schedule_sha256: str | None = None,
    baseline: dict[str, object] | None = None,
    subject: dict[str, object] | None = None,
    minimum_subject_score: float = 0.5,
) -> None:
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    claim = original["behavioral_claim"]
    changed = build_behavioral_policy_pack(
        tier="conservative",
        schedule_sha256=schedule_sha256 or claim["schedule_sha256"],
        baseline=baseline or claim["baseline"],
        subject=subject or claim["subject"],
        metric_kind="exact_match",
        minimum_subject_score=minimum_subject_score,
        maximum_regression=0.5,
        dataset_identity=original["compatibility"]["dataset_identity"],
        required_evidence_surfaces=["behavior", "build", "tokenizer"],
    )
    policy_path.write_text(json.dumps(changed), encoding="utf-8")


def _context(identity: ModelArtifactIdentity) -> RuntimeExecutionContext:
    return RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=artifact_identity_sha256(identity),
    )


def _run(
    tmp_path: Path,
    *,
    provider: _FakeProvider,
    role: str,
    directory_name: str,
    schedule_path: Path,
    policy_path: Path,
):
    return run_side(
        role=role,  # type: ignore[arg-type]
        provider=provider,
        spec=ModelRuntimeSpec(
            provider_name=provider.name,
            model_id="portable-model",
            settings={"batch_size": 1},
        ),
        context=_context(provider.identity),
        schedule_path=schedule_path,
        policy_pack_path=policy_path,
        output_directory=tmp_path / directory_name,
    )
